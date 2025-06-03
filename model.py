import re
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
import cv2
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
from sam2.utils.dcpp import DetectionCellPostProcessor, calculate_instances
from skimage.color import rgba2rgb
from skimage import io
from monai.networks import one_hot
from sam2.utils.metrics import get_fast_pq, remap_label, get_fast_aji
from typing import List, Tuple, Type, Optional
import os
from tqdm import tqdm
from functools import partial
from typing import List, Literal, Tuple, Union
from collections import OrderedDict
from torchmetrics.functional.classification import binary_jaccard_index
from sam2.utils.tools import remove_small_objects

class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                # print(lateral_features.shape, top_down_features.shape)
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos

def binarize(x):
    """
    convert multichannel (multiclass) instance segmetation tensor
    to binary instance segmentation (bg and nuclei),

    :param x: B*B*C (for PanNuke 256*256*5 )
    :return: Instance segmentation
    """
    out = np.zeros([x.shape[0], x.shape[1]])
    count = 1
    for i in range(x.shape[2]):
        x_ch = x[:, :, i]
        unique_vals = np.unique(x_ch)
        unique_vals = unique_vals.tolist()
        unique_vals.remove(0)
        for j in unique_vals:
            x_tmp = x_ch == j
            x_tmp_c = 1 - x_tmp
            out *= x_tmp_c
            out += count * x_tmp
            count += 1
    out = out.astype("int32")
    return out


def InsGenerator(mask, area=2500):

    targets = np.zeros(mask.shape, dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8)*255)
    for i in range(1, num_labels):
        targets[labels == i] = i

    targets = remove_small_objects(targets, area)
    return targets

class CoSeg(nn.Module):
    def __init__(self, sam_model = SAM2Base, img_size=1024, sem_cls=6, ins_cls=4, model_name=None):
        super(CoSeg, self).__init__()

        self.num_ins_classes = ins_cls + 3
        self.num_sem_classes = sem_cls
        self.device = "cuda"
        self.img_size = img_size
        self._transforms = SAM2Transforms(
            resolution=img_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        self.model = sam_model
        self._features = None
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        
    def calculate_step_metric_validation(self, num_classes: int, predictions: dict, gt: dict) -> dict:
        """Calculate the metrics for the training step

        Args:
            predictions (DataclassHVStorage): OrderedDict: Processed network output
            gt (DataclassHVStorage): Ground truth values
        Returns:
            dict: Dictionary with metrics. Keys:
                binary_dice_scores, binary_jaccard_scores, tissue_pred, tissue_gt
        """
        # predictions = predictions.get_dict()
        # gt = gt.get_dict()

        # Tissue Tpyes logits to probs and argmax to get class

        predictions["nuclei_binary_map"][predictions["nuclei_binary_map"] >= 0.5] = 1
        predictions["nuclei_binary_map"][predictions["nuclei_binary_map"] < 0.5] = 0
    
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        predictions["instance_types_nuclei"] = (
            predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        instance_maps_gt = gt["instance_map"].detach().cpu()[0]
        # gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
        #     torch.uint8
        # )
        gt["instance_types_nuclei"] = (
            gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )

        binary_dice_scores = []
        binary_jaccard_scores = []
        cell_type_pq_scores = []
        pq_scores = []
        dq_scores = []
        sq_scores = []
        aji_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # binary dice score: Score for cell detection per image, without background
        pred_binary_map = predictions["nuclei_binary_map"]
        target_binary_map = gt["nuclei_binary_map"]

        # binary aji
        cell_jaccard = (
            binary_jaccard_index(
                preds=pred_binary_map,
                target=target_binary_map,
            )
            .detach()
            .cpu()
        )
        binary_jaccard_scores.append(float(cell_jaccard))
        # pq values
        # print(predictions["instance_types_nuclei"][0][1:].shape)
        remapped_instance_pred = binarize(
                    predictions["instance_types_nuclei"][0][1:].transpose(1, 2, 0))
        remapped_instance_pred = np.expand_dims(remapped_instance_pred, 0)
        remapped_gt = remap_label(instance_maps_gt)
       
        # print(remapped_instance_pred.shape, remapped_gt.shape)
        [dq, sq, pq], _, precision, recall, f1 = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
        pq_scores.append(pq)
        dq_scores.append(dq)
        sq_scores.append(sq)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        aji = get_fast_aji(true=remapped_gt, pred=remapped_instance_pred)
        aji_scores.append(aji)

        # pq values per class (with class 0 beeing background -> should be skipped in the future)
        nuclei_type_pq = []
        nuclei_type_dq = []
        nuclei_type_sq = []
        nuclei_type_prec = []
        nuclei_type_recall = []
        nuclei_type_f1 = []
        nuclei_type_aji = []

        # gt["instance_types_nuclei"] = one_hot(torch.tensor(gt["instance_types_nuclei"]), num_classes=num_classes)
        numpy_types_nuclei = gt["instance_types_nuclei"][0][0].copy()
        target_instances_numpy = instance_maps_gt.cpu().numpy()
        for j in range(0, num_classes):
                temp_map = target_instances_numpy.copy()
                temp_map[0][numpy_types_nuclei!=j] = 0     

                pred_nuclei_instance_class = remap_label(
                    np.expand_dims(predictions["instance_types_nuclei"][0][j, ...], 0)
                )
                target_nuclei_instance_class = remap_label(
                    temp_map
                )
                if len(np.unique(target_nuclei_instance_class)) == 1:
                    pq_tmp = np.nan
                    dq_tmp = np.nan
                    sq_tmp = np.nan
                    precision = np.nan
                    recall = np.nan
                    f1 = np.nan
                    aji = np.nan
                else:
                    [dq_tmp, sq_tmp, pq_tmp], _, precision, recall, f1 = get_fast_pq(
                        target_nuclei_instance_class,
                        pred_nuclei_instance_class,
                        match_iou=0.5,
                    )
                    aji = get_fast_aji(true=target_nuclei_instance_class, pred=pred_nuclei_instance_class)
                    
                nuclei_type_pq.append(pq_tmp)
                nuclei_type_dq.append(dq_tmp)
                nuclei_type_sq.append(sq_tmp)
                nuclei_type_prec.append(precision)
                nuclei_type_recall.append(recall)
                nuclei_type_f1.append(f1)
                nuclei_type_aji.append(aji)

        batch_metrics = {
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
            "dq_scores": dq_scores,
            "sq_scores": sq_scores,
            "aji_scores": aji_scores,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "f1_scores": f1_scores,
            "nuclei_type_pq": nuclei_type_pq,
            "nuclei_type_dq": nuclei_type_dq,
            "nuclei_type_prec": nuclei_type_prec,
            "nuclei_type_recall": nuclei_type_recall,
            "nuclei_type_f1": nuclei_type_f1,
            "nuclei_type_sq": nuclei_type_sq,
            "nuclei_type_aji": nuclei_type_aji,
        }
        return batch_metrics
    
    def calculate_tissue_metric_validation(self, num_classes, predictions, gt):
        predictions_oh = one_hot(predictions, num_classes=num_classes).detach().cpu().numpy()[0]
        gt_oh = one_hot(gt, num_classes=num_classes).detach().cpu().numpy()[0]
        for i in range(predictions_oh.shape[0]):  # Iterate over each class mask
            pred_mask = predictions_oh[i, :, :].copy()
            gt_mask = gt_oh[i, :, :].copy()
            # Remove small objects
            pred_mask = InsGenerator(pred_mask.astype(np.uint8), area=200)
            gt_mask = InsGenerator(gt_mask.astype(np.uint8), area=0)
            # Fill holes
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
            predictions_oh[i, :, :] = pred_mask
            gt_oh[i, :, :] = gt_mask

        binary_jaccard_scores = []
        pq_scores = []
        dq_scores = []
        sq_scores = []
        aji_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        #print(num_classes)
        for i in range(0, num_classes):
            pred_tissue_class = remap_label(
                np.expand_dims(predictions_oh[i, ...], 0)
            )
            target_tissue_class = remap_label(
                np.expand_dims(gt_oh[i, ...], 0)
            )
            if len(np.unique(target_tissue_class)) == 1:
                pq = np.nan
                dq = np.nan
                sq = np.nan
                precision = np.nan
                recall = np.nan
                f1 = np.nan
                aji = np.nan
            else:
                [dq, sq, pq], _, precision, recall, f1 = get_fast_pq(
                    target_tissue_class,
                    pred_tissue_class,
                    match_iou=0.5,
                )
                #print(dq, sq, pq)
            pq_scores.append(pq)
            dq_scores.append(dq)
            sq_scores.append(sq)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
        batch_metrics = {
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
            "dq_scores": dq_scores,
            "sq_scores": sq_scores,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "f1_scores": f1_scores,
        }


        return batch_metrics, predictions_oh
    
    def calculate_instance_map(
        self, num_classes: int, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Calculate Instance Map from network predictions (after Softmax output)

        Args:
            predictions (dict): Dictionary with the following required keys:
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, self.num_ins_classes, H, W)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, 2, H, W)
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
        """
        # reshape to B, H, W, C
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        cell_post_processor = DetectionCellPostProcessor(
            nr_types=num_classes, magnification=magnification, gt=False
        )
        instance_preds = []
        type_preds = []

        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            predictions_["nuclei_binary_map"][i][predictions_["nuclei_binary_map"][i] >= 0.5] = 1
            predictions_["nuclei_binary_map"][i][predictions_["nuclei_binary_map"][i] < 0.5] = 0
            # print(torch.argmax(predictions_["nuclei_type_map"], dim=-1, keepdim=True)[i]
            #         .permute(1, 2, 0).shape)
            # print(predictions_["nuclei_binary_map"][i].shape)
            pred_map = np.concatenate(
                [
                    torch.argmax(predictions_["nuclei_type_map"], dim=-1, keepdim=True)[i]
                    .detach()
                    .cpu(),
                    predictions_["nuclei_binary_map"][i]
                    .detach()
                    .cpu(),
                    predictions_["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds

    def generate_instance_nuclei_map(
        self, num_classes: int, instance_maps: torch.Tensor, type_preds: List[dict]
    ) -> torch.Tensor:
        """Convert instance map (binary) to nuclei type instance map

        Args:
            instance_maps (torch.Tensor): Binary instance map, each instance has own integer. Shape: (B, H, W)
            type_preds (List[dict]): List (len=B) of dictionary with instance type information (compare post_process_hovernet function for more details)

        Returns:
            torch.Tensor: Nuclei type instance map. Shape: (B, self.num_ins_classes, H, W)
        """

        batch_size, h, w = instance_maps.shape
        instance_type_nuclei_maps = torch.zeros(
            (batch_size, h, w, num_classes)
        )
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, num_classes))
            instance_map = instance_maps[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][
                    instance_map == nuclei
                ] = nuclei

            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

        instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
        return torch.Tensor(instance_type_nuclei_maps)
    

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=(1024, 1024) # fixed
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=(1024, 1024)
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_sem: Optional[torch.Tensor] = None,
        mask_ins: Optional[torch.Tensor] = None,
        img_idx: int = -1,
        first_fwd: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)


        sparse_embeddings_sem, dense_embeddings_sem = self.model.stp_encoder_ins(
            points=None,
            boxes=None,
            masks=mask_ins,
            image=self._features["image_embed"][img_idx].unsqueeze(0),
        )

        sparse_embeddings_ins, dense_embeddings_ins = self.model.stp_encoder_sem(
            points=None,
            boxes=None,
            masks=mask_sem,
            image=self._features["image_embed"][img_idx].unsqueeze(0),
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        if first_fwd:
            low_res_masks_ins, prob_ins = self.model.mct_decoder_ins(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.stp_encoder_sem.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_ins,
                dense_prompt_embeddings=dense_embeddings_ins,
                multimask_output=False,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
                first_fwds = first_fwd,
            )

            low_res_masks_sem, prob_sem = self.model.mct_decoder_sem(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.stp_encoder_ins.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_sem,
                dense_prompt_embeddings=dense_embeddings_sem,
                multimask_output=False,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
                first_fwds = first_fwd,
            )
        else:
            low_res_masks_ins, prob_ins = self.model.mct_decoder_ins(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.stp_encoder_sem.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_ins,
                dense_prompt_embeddings=dense_embeddings_ins,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
                first_fwds = first_fwd,
            )

            low_res_masks_sem, prob_sem = self.model.mct_decoder_sem(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.stp_encoder_ins.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_sem,
                dense_prompt_embeddings=dense_embeddings_sem,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
                first_fwds = first_fwd,
            )
        if first_fwd:
            mask_ins = low_res_masks_ins
            mask_sem = low_res_masks_sem
        else:
            mask_ins = F.interpolate(low_res_masks_ins, (1024, 1024), mode="bilinear", align_corners=False)
            mask_sem = F.interpolate(low_res_masks_sem, (1024, 1024), mode="bilinear", align_corners=False)
        return mask_ins, mask_sem, prob_sem, prob_ins


    def forward(self, x, prob_ins=None, prob_sem=None, img_id=None):

        batch_size = x.shape[0]
        backbone_out = self.model.forward_image(x)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]} 


        num_images = len(self._features["image_embed"])

        outputs_mask_ins = []
        outputs_mask_sem = []
        outputs_prob_ins = []
        outputs_prob_sem = []
        normalize_coords = True
        for img_idx in range(num_images):

            point_coords = None
            point_labels = None
            box = None
            mask_input = None
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )

            if (prob_sem != None) and (prob_ins != None):
                masks_ins, masks_sem, _, _ = self._predict(
                    unnorm_coords,
                    labels,
                    unnorm_box,
                    prob_sem[img_idx].unsqueeze(0),
                    prob_ins[img_idx].unsqueeze(0),
                    img_idx=img_idx,
                    first_fwd = False
                )
                outputs_mask_ins.append(masks_ins.squeeze(0))
                outputs_mask_sem.append(masks_sem.squeeze(0))

            else:
                masks_ins, masks_sem, kl_sem, kl_ins = self._predict(
                    unnorm_coords,
                    labels,
                    unnorm_box,
                    prob_sem,
                    prob_ins,
                    img_idx=img_idx,
                    first_fwd = True
                )
                outputs_mask_ins.append(masks_ins.squeeze(0))
                outputs_mask_sem.append(masks_sem.squeeze(0))
                outputs_prob_sem.append(kl_sem.squeeze(0))
                outputs_prob_ins.append(kl_ins.squeeze(0))
        if (prob_sem != None) and (prob_ins != None):
            return torch.stack(outputs_mask_ins, dim=0), torch.stack(outputs_mask_sem, dim=0)
        else:
            return torch.stack(outputs_mask_ins, dim=0), torch.stack(outputs_mask_sem, dim=0), torch.stack(outputs_prob_ins, dim=0), torch.stack(outputs_prob_sem, dim=0)
        



