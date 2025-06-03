import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataloader import PUMALoader
from skimage import measure, morphology
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image
import json
from tqdm import tqdm
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from model import CoSeg
from functools import partial
from scipy import ndimage as ndi
from monai.metrics import compute_hausdorff_distance, compute_percent_hausdorff_distance, HausdorffDistanceMetric
from monai.metrics import DiceHelper, ConfusionMatrixMetric, get_confusion_matrix, compute_confusion_matrix_metric, MeanIoU
from sam2.build_sam import build_sam2
from monai.networks import one_hot
import hydra
import matplotlib.pyplot as plt


def hd_score(p, y):

    tmp_hd = compute_hausdorff_distance(p, y, True)
    return tmp_hd



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='puma',type=str, help='data')
    parser.add_argument('--mode', default='2d',type=str, help='2d, 3d')
    parser.add_argument('--gt_path', default='mask_1024',type=str, help='')
    parser.add_argument('--jsonfile', default='data_split.json',type=str, help='')
    parser.add_argument('--size', type=int, default=1024, help='epoches')
    parser.add_argument('--model',default='pretrain/sam2_hiera_large.pt', type=str, help='')
    parser.add_argument('--checkpoint',default='outputs/coseg_puma_39.pth', type=str, help='')
    parser.add_argument('--ins_cls', type=int, default=4, help='number of instance classes (including background)')
    parser.add_argument('--sem_cls', type=int, default=3, help='number of semantic classes (including background)')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory')
    args = parser.parse_args()
    
    save_png = os.path.join(args.output_dir, args.dataset)
    os.makedirs(save_png,exist_ok=True)


    print(args.dataset)
    print("------------------------------------------")

    args.jsonfile = f'datasets/{args.dataset}/data_split.json'

    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    test_files = df['test']


    test_dataset = PUMALoader(args.dataset, test_files, A.Compose([
                                        A.Resize(256, 256),
                                        ], additional_targets={'mask2': 'mask','mask3': 'mask','mask4': 'mask','mask5': 'mask'}))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)
    model_cfg = "sam2_hiera_l.yaml"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    model = CoSeg(build_sam2(model_cfg, args.model, mode=None))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    total_params = sum(
	param.numel() for param in model.parameters()
    )
    print('Total Params = ' + str(total_params/1000**2) + 'M')
    model.load_state_dict(torch.load(args.checkpoint), strict=True)
    model = model.cuda()

    idx = 0
    
    TestAcc = Accuracy()
    TestPrecision = Precision()
    TestDiceB_tissue = DiceHelper(include_background=True)
    TestDiceM_tissue = DiceHelper(include_background=True, num_classes=args.sem_cls, softmax=True, get_not_nans=True, ignore_empty=False)
    TestDiceB_nuclei = DiceHelper(include_background=True)
    TestDiceM_nuclei = DiceHelper(include_background=True, num_classes=args.ins_cls, softmax=True, get_not_nans=True, ignore_empty=False)
    TestRecall = Recall()
    TestF1 = ConfusionMatrixMetric(metric_name='f1 score')
    TestIoU = MeanIoU(get_not_nans=True, ignore_empty=False)

    FPS_score = []
    Dice_tissue1 = []
    Dice_tissue2 = []
    Dice_tissue_avg = []
    Dice_nuclei1 = []
    Dice_nuclei2 = []
    Dice_nuclei3 = []
    Dice_nuclei_avg = []
    miou_tissue1 = []
    miou_tissue2 = []
    miou_tissue_avg = []
    miou_nuclei1 = []
    miou_nuclei2 = []
    miou_nuclei3 = []
    miou_nuclei_avg = []
    hd_tissue1 = []
    hd_tissue2 = []
    hd_tissue_avg = []
    f1_nuclei1 = []
    f1_nuclei2 = []
    f1_nuclei3 = []
    f1_nuclei_avg = []
    prec_nuclei1 = []
    prec_nuclei2 = []
    prec_nuclei3 = []
    prec_nuclei_avg = []
    recall_nuclei1 = []
    recall_nuclei2 = []
    recall_nuclei3 = []
    recall_nuclei_avg = []
    aji_nuclei1 = []
    aji_nuclei2 = []
    aji_nuclei3 = []
    aji_nuclei_avg = []
    pq_nuclei_avg = []
    dq_nuclei_avg = []
    sq_nuclei_avg = []
    pq_tissue_avg = []
    dq_tissue_avg = []
    sq_tissue_avg = []
    pq_nuclei1 = []
    pq_nuclei2 = []
    pq_nuclei3 = []
    dq_nuclei1 = []
    dq_nuclei2 = []
    dq_nuclei3 = []
    sq_nuclei1 = []
    sq_nuclei2 = []
    sq_nuclei3 = []
    pq_tissue1 = []
    pq_tissue2 = []
    dq_tissue1 = []
    dq_tissue2 = []
    sq_tissue1 = []
    sq_tissue2 = []
    image_ids = []
    
    since = time.time()

    model.train(False)  
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for data_dict in tqdm(test_loader):
            
            img = Variable(data_dict["image"].cuda())
            tissue_map = Variable(data_dict["tissue_map"].cuda()).unsqueeze(1)
            nuclei_tissue_map_res = Variable(data_dict["nuclei_tissue_map_res"].cuda()).unsqueeze(1)
            nuclei_binary_map = Variable(data_dict["nuclei_binary_map"].cuda()).unsqueeze(1)
            nuclei_type_map = Variable(data_dict["nuclei_type_map"].cuda()).unsqueeze(1)
            nuclei_inst_map = Variable(data_dict["nuclei_inst_map"].cuda()).unsqueeze(1)
            nuclei_hv_map = Variable(data_dict["nuclei_hv_map"].cuda())
            nuclei_type_map_re = Variable(data_dict["nuclei_type_map_res"].cuda()).unsqueeze(1)
            img_id = data_dict["image_id"]

            torch.cuda.synchronize()
            start = time.time()

            mask_pred_ins, mask_pred_sem, _, _ = model(x=img)
            mask_pred_ins, mask_pred_sem = model(x=img, prob_ins=mask_pred_ins, prob_sem=mask_pred_sem)

            nuclei_binary_map_pred = mask_pred_ins[:,0:1,:,:]
            nuclei_type_map_pred = mask_pred_ins[:,1:1+args.ins_cls,:,:]
            nuclei_hv_map_pred = mask_pred_ins[:,1+args.ins_cls:3+args.ins_cls,:,:]

            mask_pred_sem = torch.softmax(mask_pred_sem, 1)
            pred_sem = torch.argmax(mask_pred_sem, dim=1, keepdim=True)
            nuclei_type_map_pred = torch.softmax(nuclei_type_map_pred, 1)
            nuclei_binary_map_pred = torch.sigmoid(nuclei_binary_map_pred)

            predictions = {"nuclei_binary_map": nuclei_binary_map_pred,
                "nuclei_type_map":  nuclei_type_map_pred,
                "hv_map": nuclei_hv_map_pred}
            gt = {"instance_map": nuclei_inst_map,
                "nuclei_binary_map":  nuclei_binary_map,
                "instance_types_nuclei": nuclei_type_map}
            
            predictions["instance_map"], predictions["instance_types"] = model.calculate_instance_map(args.ins_cls, predictions, 40)
            predictions["instance_types_nuclei"] = model.generate_instance_nuclei_map(args.ins_cls, predictions["instance_map"], predictions["instance_types"]).cuda()
            predictions["batch_size"] = predictions["nuclei_binary_map"].shape[0]
            predictions["regression_map"] = None
            predictions["num_nuclei_classes"] = args.ins_cls
            predictions["tissue_types"] = None
            batch_metrics = model.calculate_step_metric_validation(args.ins_cls, predictions, gt)
            nuclei_binary_map_pred[nuclei_binary_map_pred >= 0.5] = 1
            nuclei_binary_map_pred[nuclei_binary_map_pred < 0.5] = 0

            color_sem = np.array([[255, 255, 255], [195, 135, 123], [127, 113, 229]])
            mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
            for i in range(0, args.sem_cls):
                sem_show = torch.zeros_like(pred_sem)
                sem_show[pred_sem == i] = 1
                sem_show = sem_show.cpu().detach().numpy()[0][0].astype(np.uint8)
                contours, _ = cv2.findContours(sem_show, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask[sem_show == 1] = color_sem[i]
                contour_color = (color_sem[i] * 0.7).astype(np.uint8).tolist()
                cv2.drawContours(mask, contours, -1, contour_color, thickness=1)
            cv2.imwrite(f"{save_png}/{img_id[0]}_sem.png", mask)

            batch_metrics_tissue, post_tissue = model.calculate_tissue_metric_validation(args.sem_cls, pred_sem, tissue_map)
            mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
            for i in range(0, args.sem_cls):
                sem_show = post_tissue[i].copy()
                sem_show[sem_show > 0] = 1
                sem_show = sem_show.astype(np.uint8)
                contours, _ = cv2.findContours(sem_show, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask[sem_show == 1] = color_sem[i]
                contour_color = (color_sem[i] * 0.7).astype(np.uint8).tolist()
                cv2.drawContours(mask, contours, -1, contour_color, thickness=1)
            cv2.imwrite(f"{save_png}/{img_id[0]}_post_sem.png", mask)

            # Generate an RGB mask for instances with contours
            instance_types_np = predictions["instance_types_nuclei"][0].astype(np.uint8)

            # Create an RGB mask for cell instances
            cell_instance_mask = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
            cell_instance_mask_tissue = mask.copy()

            # Define fixed colors for each cell type
            cell_type_colors = {
                1: [3, 174, 251],    # Red for type 1
                2: [177, 159, 243],    # Green for type 2
                3: [165, 217, 176],    # Blue for type 3
                # Add more types and colors if needed
            }

            for cell_type in range(1, args.ins_cls):  # Iterate over each cell type

                # Find unique instances for the current cell type
                unique_instances = np.unique(instance_types_np[cell_type])

                for instance_id in unique_instances:
                    if instance_id == 0:  # Skip background
                        continue

                    # Create a binary mask for the current instance
                    instance_binary_mask = (instance_types_np[cell_type] == instance_id).astype(np.uint8)

                    # Find contours for the current instance
                    contours, _ = cv2.findContours(instance_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Assign the color based on the cell type
                    instance_color = cell_type_colors.get(cell_type, [128, 128, 128])  # Default to gray if type is unknown
                    cell_instance_mask[instance_binary_mask == 1] = instance_color
                    cell_instance_mask_tissue[instance_binary_mask == 1] = instance_color
                    contour_color = (np.array(instance_color) * 0.7).astype(np.uint8).tolist()

                    # Draw contours on the mask
                    cv2.drawContours(cell_instance_mask, contours, -1, contour_color, thickness=1)
                    cv2.drawContours(cell_instance_mask_tissue, contours, -1, contour_color, thickness=1)

            # Save the RGB cell mask
            cv2.imwrite(f"{save_png}/{img_id[0]}_ins.png", cell_instance_mask)
            cv2.imwrite(f"{save_png}/{img_id[0]}_ins_sem.png", cell_instance_mask_tissue)

            mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
            for i in range(0, args.sem_cls):
                mask_boundary = np.zeros((1024, 1024), dtype=np.uint8)
                sem_show = torch.zeros_like(tissue_map)
                sem_show[tissue_map == i] = 1
                sem_show = sem_show.cpu().detach().numpy()[0][0].astype(np.uint8)
                bin_show = np.zeros_like(sem_show)
                bin_show[sem_show > 0] = 255
                bin_show_copy = bin_show.copy()
                bin_show_reverse = 255 - bin_show
                bin_show_reverse[bin_show_reverse < 0] = 255
                cv2.imwrite(f"{save_png}/{img_id[0]}_sem_binary_{i}.png", bin_show)
                cv2.imwrite(f"{save_png}/{img_id[0]}_sem_binary_reverse_{i}.png", bin_show_reverse)
                boundary = cv2.morphologyEx(sem_show, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
                contours, _ = cv2.findContours(sem_show, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_boundary, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask[sem_show == 1] = color_sem[i]
                contour_color = (color_sem[i] * 0.7).astype(np.uint8).tolist()
                h, w = bin_show_copy.shape
                edge_contours = []
                cv2.drawContours(mask, contours, -1, contour_color, thickness=1)
                for cnt in contours:
                    if np.any(cnt[:, 0, 0] == 0) or np.any(cnt[:, 0, 0] == w-1) or \
                        np.any(cnt[:, 0, 1] == 0) or np.any(cnt[:, 0, 1] == h-1):
                            edge_contours.append(cnt)
                cv2.drawContours(mask_boundary, contours_boundary, -1, 255, thickness=5)
                cv2.drawContours(mask_boundary, edge_contours, -1, 255, thickness=7)
                # mask_boundary = cv2.morphologyEx(mask_boundary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                cv2.imwrite(f"{save_png}/{img_id[0]}_gt_boundary_{i}.png", mask_boundary)
            cv2.imwrite(f"{save_png}/{img_id[0]}_gt_sem.png", mask)

            cell_instance_mask = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
            cell_instance_mask_tissue = mask.copy()
            nuclei_inst_map_numpy = nuclei_inst_map.cpu().detach().numpy()[0][0].astype(np.uint8)
            nuclei_type_map_numpy = nuclei_type_map.cpu().detach().numpy()[0][0].astype(np.uint8)
            for cell_type in range(1, args.ins_cls):  # Iterate over each cell type
                temp_map = nuclei_inst_map_numpy.copy()
                # Find unique instances for the current cell type
                temp_map[nuclei_type_map_numpy!=cell_type] = 0
                unique_instances = np.unique(temp_map)

                for instance_id in unique_instances:
                    if instance_id == 0:  # Skip background
                        continue

                    # Create a binary mask for the current instance
                    instance_binary_mask = (temp_map == instance_id).astype(np.uint8)

                    # Find contours for the current instance
                    contours, _ = cv2.findContours(instance_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Assign the color based on the cell type
                    instance_color = cell_type_colors.get(cell_type, [128, 128, 128])  # Default to gray if type is unknown
                    cell_instance_mask[instance_binary_mask == 1] = instance_color
                    cell_instance_mask_tissue[instance_binary_mask == 1] = instance_color
                    contour_color = (np.array(instance_color) * 0.7).astype(np.uint8).tolist()

                    # Draw contours on the mask
                    cv2.drawContours(cell_instance_mask, contours, -1, contour_color, thickness=1)
                    cv2.drawContours(cell_instance_mask_tissue, contours, -1, contour_color, thickness=1)

            # Save the RGB cell mask
            cv2.imwrite(f"{save_png}/{img_id[0]}_gt_ins.png", cell_instance_mask)     
            cv2.imwrite(f"{save_png}/{img_id[0]}_gt_ins_sem.png", cell_instance_mask_tissue)  
 
            nuclei_binary_map_gt = nuclei_type_map.cpu().detach().numpy()[0][0].astype(np.uint8)
            nuclei_binary_map_gt[nuclei_binary_map_gt == 3] = 1
            for cell_type in range(1, args.ins_cls-1):
                nuclei_binary_map_gt_draw = np.zeros_like(nuclei_binary_map_gt)
                nuclei_binary_map_gt_draw[nuclei_binary_map_gt == cell_type] = 255
                cv2.imwrite(f"{save_png}/{img_id[0]}_gt_ins_binary_{cell_type}.png", nuclei_binary_map_gt_draw)     


            torch.cuda.synchronize()
            end = time.time()
            FPS_score.append(end-start)
            predictions["instance_types_nuclei"] = torch.argmax(torch.tensor(predictions["instance_types_nuclei"]).cuda(), dim=1, keepdim=True)
            gt["instance_types_nuclei"] = torch.tensor(gt["instance_types_nuclei"]).cuda()
            batch_dice_nuclei_multi = TestDiceM_nuclei(predictions["instance_types_nuclei"], gt["instance_types_nuclei"])
            # batch_dice_nuclei_binary = TestDiceB_nuclei(nuclei_binary_map_pred, nuclei_binary_map)
            batch_dice_tissue_multi = TestDiceM_tissue(pred_sem, tissue_map)
            batch_iou_tissue = TestIoU(one_hot(pred_sem, num_classes=args.sem_cls), one_hot(tissue_map, num_classes=args.sem_cls))
            batch_iou_nuclei = TestIoU(one_hot(predictions["instance_types_nuclei"], num_classes=args.ins_cls), one_hot(gt["instance_types_nuclei"], num_classes=args.ins_cls))
            hdscore = hd_score(one_hot(pred_sem, num_classes=args.sem_cls), one_hot(tissue_map, num_classes=args.sem_cls))

            Dice_tissue1.append(batch_dice_tissue_multi[0][1].item())
            Dice_tissue2.append(batch_dice_tissue_multi[0][2].item())
            Dice_tissue_avg.append(batch_dice_tissue_multi[0][1:].mean().item())

            Dice_nuclei1.append(batch_dice_nuclei_multi[0][1].item())
            Dice_nuclei2.append(batch_dice_nuclei_multi[0][2].item())
            Dice_nuclei3.append(batch_dice_nuclei_multi[0][3].item())
            Dice_nuclei_avg.append(batch_dice_nuclei_multi[0][1:].mean().item())

            miou_tissue1.append(batch_iou_tissue[0][1].item())
            miou_tissue2.append(batch_iou_tissue[0][2].item())
            miou_tissue_avg.append(batch_iou_tissue[0][1:].mean().item())
            
            miou_nuclei1.append(batch_iou_nuclei[0][1].item())
            miou_nuclei2.append(batch_iou_nuclei[0][2].item())
            miou_nuclei3.append(batch_iou_nuclei[0][3].item())
            miou_nuclei_avg.append(batch_iou_nuclei[0][1:].mean().item())

            hd_tissue1.append(hdscore[0][1].item())
            hd_tissue2.append(hdscore[0][2].item())
            hd_tissue_avg.append(hdscore[0][1:].mean().item())

            f1_nuclei1.append(batch_metrics['nuclei_type_f1'][1])
            f1_nuclei2.append(batch_metrics['nuclei_type_f1'][2])
            f1_nuclei3.append(batch_metrics['nuclei_type_f1'][3])
            f1_nuclei_avg.append(np.array(batch_metrics['nuclei_type_f1'][1:]).mean())

            prec_nuclei1.append(batch_metrics['nuclei_type_prec'][1])
            prec_nuclei2.append(batch_metrics['nuclei_type_prec'][2])
            prec_nuclei3.append(batch_metrics['nuclei_type_prec'][3])
            prec_nuclei_avg.append(np.array(batch_metrics['nuclei_type_prec'][1:]).mean())

            recall_nuclei1.append(batch_metrics['nuclei_type_recall'][1])
            recall_nuclei2.append(batch_metrics['nuclei_type_recall'][2])
            recall_nuclei3.append(batch_metrics['nuclei_type_recall'][3])
            recall_nuclei_avg.append(np.array(batch_metrics['nuclei_type_recall'][1:]).mean())

            aji_nuclei1.append(batch_metrics['nuclei_type_aji'][1])
            aji_nuclei2.append(batch_metrics['nuclei_type_aji'][2])
            aji_nuclei3.append(batch_metrics['nuclei_type_aji'][3])
            aji_nuclei_avg.append(np.array(batch_metrics['nuclei_type_aji'][1:]).mean())

            pq_nuclei_avg.append(np.array(batch_metrics['nuclei_type_pq'][1:]).mean())
            dq_nuclei_avg.append(np.array(batch_metrics['nuclei_type_dq'][1:]).mean())
            sq_nuclei_avg.append(np.array(batch_metrics['nuclei_type_sq'][1:]).mean())

            pq_tissue_avg.append(np.array(batch_metrics_tissue['pq_scores'][1:]).mean())
            dq_tissue_avg.append(np.array(batch_metrics_tissue['dq_scores'][1:]).mean())
            sq_tissue_avg.append(np.array(batch_metrics_tissue['sq_scores'][1:]).mean())

            pq_nuclei1.append(batch_metrics['nuclei_type_pq'][1])
            pq_nuclei2.append(batch_metrics['nuclei_type_pq'][2])
            pq_nuclei3.append(batch_metrics['nuclei_type_pq'][3])
            dq_nuclei1.append(batch_metrics['nuclei_type_dq'][1])
            dq_nuclei2.append(batch_metrics['nuclei_type_dq'][2])
            dq_nuclei3.append(batch_metrics['nuclei_type_dq'][3])
            sq_nuclei1.append(batch_metrics['nuclei_type_sq'][1])
            sq_nuclei2.append(batch_metrics['nuclei_type_sq'][2])
            sq_nuclei3.append(batch_metrics['nuclei_type_sq'][3])
            pq_tissue1.append(batch_metrics_tissue['pq_scores'][1])
            pq_tissue2.append(batch_metrics_tissue['pq_scores'][2])
            dq_tissue1.append(batch_metrics_tissue['dq_scores'][1])
            dq_tissue2.append(batch_metrics_tissue['dq_scores'][2])
            sq_tissue1.append(batch_metrics_tissue['sq_scores'][1])
            sq_tissue2.append(batch_metrics_tissue['sq_scores'][2])

            image_ids.append(img_id[0])
            torch.cuda.empty_cache()

         
    time_elapsed = time.time() - since
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    

    Dice_nuclei_avg = np.array([x for x in Dice_nuclei_avg if not np.isnan(x)])
    Dice_tissue_avg = np.array([x for x in Dice_tissue_avg if not np.isnan(x)])
    miou_nuclei_avg = np.array([x for x in miou_nuclei_avg if not np.isnan(x)])
    miou_tissue_avg = np.array([x for x in miou_tissue_avg if not np.isnan(x)])
    hd_tissue_avg = np.array([x for x in hd_tissue_avg if not (np.isnan(x) or np.isinf(x))])

    f1_nuclei_avg = np.array([x for x in f1_nuclei_avg if not np.isnan(x)])
    prec_nuclei_avg = np.array([x for x in prec_nuclei_avg if not np.isnan(x)])
    recall_nuclei_avg = np.array([x for x in recall_nuclei_avg if not np.isnan(x)])
    aji_nuclei_avg = np.array([x for x in aji_nuclei_avg if not np.isnan(x)])
    pq_nuclei_avg = np.array([x for x in pq_nuclei_avg if not np.isnan(x)])
    dq_nuclei_avg = np.array([x for x in dq_nuclei_avg if not np.isnan(x)])
    sq_nuclei_avg = np.array([x for x in sq_nuclei_avg if not np.isnan(x)])
    pq_tissue_avg = np.array([x for x in pq_tissue_avg if not np.isnan(x)])
    dq_tissue_avg = np.array([x for x in dq_tissue_avg if not np.isnan(x)])
    sq_tissue_avg = np.array([x for x in sq_tissue_avg if not np.isnan(x)])

    Dice_nuclei1 = np.array([x for x in Dice_nuclei1 if not np.isnan(x)])
    Dice_nuclei2 = np.array([x for x in Dice_nuclei2 if not np.isnan(x)])
    Dice_nuclei3 = np.array([x for x in Dice_nuclei3 if not np.isnan(x)])
    Dice_tissue1 = np.array([x for x in Dice_tissue1 if not np.isnan(x)])
    Dice_tissue2 = np.array([x for x in Dice_tissue2 if not np.isnan(x)])
    miou_nuclei1 = np.array([x for x in miou_nuclei1 if not np.isnan(x)])
    miou_nuclei2 = np.array([x for x in miou_nuclei2 if not np.isnan(x)])
    miou_nuclei3 = np.array([x for x in miou_nuclei3 if not np.isnan(x)])
    miou_tissue1 = np.array([x for x in miou_tissue1 if not np.isnan(x)])
    miou_tissue2 = np.array([x for x in miou_tissue2 if not np.isnan(x)])
    f1_nuclei1 = np.array([x for x in f1_nuclei1 if not np.isnan(x)])
    f1_nuclei2 = np.array([x for x in f1_nuclei2 if not np.isnan(x)])
    f1_nuclei3 = np.array([x for x in f1_nuclei3 if not np.isnan(x)])
    prec_nuclei1 = np.array([x for x in prec_nuclei1 if not np.isnan(x)])
    prec_nuclei2 = np.array([x for x in prec_nuclei2 if not np.isnan(x)])
    prec_nuclei3 = np.array([x for x in prec_nuclei3 if not np.isnan(x)])
    recall_nuclei1 = np.array([x for x in recall_nuclei1 if not np.isnan(x)])
    recall_nuclei2 = np.array([x for x in recall_nuclei2 if not np.isnan(x)])
    recall_nuclei3 = np.array([x for x in recall_nuclei3 if not np.isnan(x)])
    aji_nuclei1 = np.array([x for x in aji_nuclei1 if not np.isnan(x)])
    aji_nuclei2 = np.array([x for x in aji_nuclei2 if not np.isnan(x)])
    aji_nuclei3 = np.array([x for x in aji_nuclei3 if not np.isnan(x)])
    hd_tissue1 = np.array([x for x in hd_tissue1 if not (np.isnan(x) or np.isinf(x))])
    hd_tissue2 = np.array([x for x in hd_tissue2 if not (np.isnan(x) or np.isinf(x))])
    pq_nuclei1 = np.array([x for x in pq_nuclei1 if not np.isnan(x)])
    pq_nuclei2 = np.array([x for x in pq_nuclei2 if not np.isnan(x)])
    pq_nuclei3 = np.array([x for x in pq_nuclei3 if not np.isnan(x)])
    dq_nuclei1 = np.array([x for x in dq_nuclei1 if not np.isnan(x)])
    dq_nuclei2 = np.array([x for x in dq_nuclei2 if not np.isnan(x)])
    dq_nuclei3 = np.array([x for x in dq_nuclei3 if not np.isnan(x)])
    sq_nuclei1 = np.array([x for x in sq_nuclei1 if not np.isnan(x)])
    sq_nuclei2 = np.array([x for x in sq_nuclei2 if not np.isnan(x)])
    sq_nuclei3 = np.array([x for x in sq_nuclei3 if not np.isnan(x)])
    pq_tissue1 = np.array([x for x in pq_tissue1 if not np.isnan(x)])
    pq_tissue2 = np.array([x for x in pq_tissue2 if not np.isnan(x)])
    dq_tissue1 = np.array([x for x in dq_tissue1 if not np.isnan(x)])
    dq_tissue2 = np.array([x for x in dq_tissue2 if not np.isnan(x)])
    sq_tissue1 = np.array([x for x in sq_tissue1 if not np.isnan(x)])
    sq_tissue2 = np.array([x for x in sq_tissue2 if not np.isnan(x)])

    FPS_score.pop(0)
    print('FPS: {:.2f}'.format(1.0/(sum(FPS_score)/len(FPS_score))))
    print(f'AVG Dice Nuclei: {np.mean(Dice_nuclei_avg):.4f} +/- {np.std(Dice_nuclei_avg):.4f} Tissue: {np.mean(Dice_tissue_avg):.4f} +/- {np.std(Dice_tissue_avg):.4f}')
    print(f'AVG mIoU Nuclei: {np.mean(miou_nuclei_avg):.4f} +/- {np.std(miou_nuclei_avg):.4f} Tissue: {np.mean(miou_tissue_avg):.4f} +/- {np.std(miou_tissue_avg):.4f}')
    print(f'AVG HD Tissue: {np.mean(hd_tissue_avg):.4f} +/- {np.std(hd_tissue_avg):.4f}')
    print(f'AVG F1 Nuclei: {np.mean(f1_nuclei_avg):.4f} +/- {np.std(f1_nuclei_avg):.4f}')
    print(f'AVG Precision Nuclei: {np.mean(prec_nuclei_avg):.4f} +/- {np.std(prec_nuclei_avg):.4f}')
    print(f'AVG Recall Nuclei: {np.mean(recall_nuclei_avg):.4f} +/- {np.std(recall_nuclei_avg):.4f}')
    print(f'AVG AJI Nuclei: {np.mean(aji_nuclei_avg):.4f} +/- {np.std(aji_nuclei_avg):.4f}')
    print(f'AVG PQ Nuclei: {np.mean(pq_nuclei_avg):.4f} +/- {np.std(pq_nuclei_avg):.4f}')
    print(f'AVG DQ Nuclei: {np.mean(dq_nuclei_avg):.4f} +/- {np.std(dq_nuclei_avg):.4f}')
    print(f'AVG SQ Nuclei: {np.mean(sq_nuclei_avg):.4f} +/- {np.std(sq_nuclei_avg):.4f}')
    print(f'AVG PQ Tissue: {np.mean(pq_tissue_avg):.4f} +/- {np.std(pq_tissue_avg):.4f}')
    print(f'AVG DQ Tissue: {np.mean(dq_tissue_avg):.4f} +/- {np.std(dq_tissue_avg):.4f}')
    print(f'AVG SQ Tissue: {np.mean(sq_tissue_avg):.4f} +/- {np.std(sq_tissue_avg):.4f}')
    print(f'Dice Nuclei1: {np.mean(Dice_nuclei1):.4f} +/- {np.std(Dice_nuclei1):.4f}')
    print(f'Dice Nuclei2: {np.mean(Dice_nuclei2):.4f} +/- {np.std(Dice_nuclei2):.4f}')
    print(f'Dice Nuclei3: {np.mean(Dice_nuclei3):.4f} +/- {np.std(Dice_nuclei3):.4f}')
    print(f'Dice Tissue1: {np.mean(Dice_tissue1):.4f} +/- {np.std(Dice_tissue1):.4f}')
    print(f'Dice Tissue2: {np.mean(Dice_tissue2):.4f} +/- {np.std(Dice_tissue2):.4f}')
    print(f'mIoU Nuclei1: {np.mean(miou_nuclei1):.4f} +/- {np.std(miou_nuclei1):.4f}')
    print(f'mIoU Nuclei2: {np.mean(miou_nuclei2):.4f} +/- {np.std(miou_nuclei2):.4f}')
    print(f'mIoU Nuclei3: {np.mean(miou_nuclei3):.4f} +/- {np.std(miou_nuclei3):.4f}')
    print(f'mIoU Tissue1: {np.mean(miou_tissue1):.4f} +/- {np.std(miou_tissue1):.4f}')
    print(f'mIoU Tissue2: {np.mean(miou_tissue2):.4f} +/- {np.std(miou_tissue2):.4f}')
    print(f'F1 Nuclei1: {np.mean(f1_nuclei1):.4f} +/- {np.std(f1_nuclei1):.4f}')
    print(f'F1 Nuclei2: {np.mean(f1_nuclei2):.4f} +/- {np.std(f1_nuclei2):.4f}')
    print(f'F1 Nuclei3: {np.mean(f1_nuclei3):.4f} +/- {np.std(f1_nuclei3):.4f}')
    print(f'Precision Nuclei1: {np.mean(prec_nuclei1):.4f} +/- {np.std(prec_nuclei1):.4f}')
    print(f'Precision Nuclei2: {np.mean(prec_nuclei2):.4f} +/- {np.std(prec_nuclei2):.4f}')
    print(f'Precision Nuclei3: {np.mean(prec_nuclei3):.4f} +/- {np.std(prec_nuclei3):.4f}')
    print(f'Recall Nuclei1: {np.mean(recall_nuclei1):.4f} +/- {np.std(recall_nuclei1):.4f}')
    print(f'Recall Nuclei2: {np.mean(recall_nuclei2):.4f} +/- {np.std(recall_nuclei2):.4f}')
    print(f'Recall Nuclei3: {np.mean(recall_nuclei3):.4f} +/- {np.std(recall_nuclei3):.4f}')
    print(f'AJI Nuclei1: {np.mean(aji_nuclei1):.4f} +/- {np.std(aji_nuclei1):.4f}')
    print(f'AJI Nuclei2: {np.mean(aji_nuclei2):.4f} +/- {np.std(aji_nuclei2):.4f}')
    print(f'AJI Nuclei3: {np.mean(aji_nuclei3):.4f} +/- {np.std(aji_nuclei3):.4f}')
    print(f'HD Tissue1: {np.mean(hd_tissue1):.4f} +/- {np.std(hd_tissue1):.4f}')
    print(f'HD Tissue2: {np.mean(hd_tissue2):.4f} +/- {np.std(hd_tissue2):.4f}')
    print(f'PQ Nuclei1: {np.mean(pq_nuclei1):.4f} +/- {np.std(pq_nuclei1):.4f}')
    print(f'PQ Nuclei2: {np.mean(pq_nuclei2):.4f} +/- {np.std(pq_nuclei2):.4f}')
    print(f'PQ Nuclei3: {np.mean(pq_nuclei3):.4f} +/- {np.std(pq_nuclei3):.4f}')
    print(f'DQ Nuclei1: {np.mean(dq_nuclei1):.4f} +/- {np.std(dq_nuclei1):.4f}')
    print(f'DQ Nuclei2: {np.mean(dq_nuclei2):.4f} +/- {np.std(dq_nuclei2):.4f}')
    print(f'DQ Nuclei3: {np.mean(dq_nuclei3):.4f} +/- {np.std(dq_nuclei3):.4f}')
    print(f'SQ Nuclei1: {np.mean(sq_nuclei1):.4f} +/- {np.std(sq_nuclei1):.4f}')
    print(f'SQ Nuclei2: {np.mean(sq_nuclei2):.4f} +/- {np.std(sq_nuclei2):.4f}')
    print(f'SQ Nuclei3: {np.mean(sq_nuclei3):.4f} +/- {np.std(sq_nuclei3):.4f}')
    print(f'PQ Tissue1: {np.mean(pq_tissue1):.4f} +/- {np.std(pq_tissue1):.4f}')
    print(f'PQ Tissue2: {np.mean(pq_tissue2):.4f} +/- {np.std(pq_tissue2):.4f}')
    print(f'DQ Tissue1: {np.mean(dq_tissue1):.4f} +/- {np.std(dq_tissue1):.4f}')
    print(f'DQ Tissue2: {np.mean(dq_tissue2):.4f} +/- {np.std(dq_tissue2):.4f}')
    print(f'SQ Tissue1: {np.mean(sq_tissue1):.4f} +/- {np.std(sq_tissue1):.4f}')
    print(f'SQ Tissue2: {np.mean(sq_tissue2):.4f} +/- {np.std(sq_tissue2):.4f}')

    # Organize the metrics into a dictionary for saving as CSV
    results = {
        "Image ID": image_ids,
        "FPS": FPS_score,
        "Dice Nuclei1": Dice_nuclei1,
        "Dice Nuclei2": Dice_nuclei2,
        "Dice Nuclei3": Dice_nuclei3,
        "Dice Nuclei Avg": Dice_nuclei_avg,
        "Dice Tissue1": Dice_tissue1,
        "Dice Tissue2": Dice_tissue2,
        "Dice Tissue Avg": Dice_tissue_avg,
        "mIoU Nuclei1": miou_nuclei1,
        "mIoU Nuclei2": miou_nuclei2,
        "mIoU Nuclei3": miou_nuclei3,
        "mIoU Nuclei Avg": miou_nuclei_avg,
        "mIoU Tissue1": miou_tissue1,
        "mIoU Tissue2": miou_tissue2,
        "mIoU Tissue Avg": miou_tissue_avg,
        "HD Tissue1": hd_tissue1,
        "HD Tissue2": hd_tissue2,
        "HD Tissue Avg": hd_tissue_avg,
        "F1 Nuclei1": f1_nuclei1,
        "F1 Nuclei2": f1_nuclei2,
        "F1 Nuclei3": f1_nuclei3,
        "F1 Nuclei Avg": f1_nuclei_avg,
        "Precision Nuclei1": prec_nuclei1,
        "Precision Nuclei2": prec_nuclei2,
        "Precision Nuclei3": prec_nuclei3,
        "Precision Nuclei Avg": prec_nuclei_avg,
        "Recall Nuclei1": recall_nuclei1,
        "Recall Nuclei2": recall_nuclei2,
        "Recall Nuclei3": recall_nuclei3,
        "Recall Nuclei Avg": recall_nuclei_avg,
        "AJI Nuclei1": aji_nuclei1,
        "AJI Nuclei2": aji_nuclei2,
        "AJI Nuclei3": aji_nuclei3,
        "AJI Nuclei Avg": aji_nuclei_avg,
        "PQ Nuclei1": pq_nuclei1,
        "PQ Nuclei2": pq_nuclei2,
        "PQ Nuclei3": pq_nuclei3,
        "PQ Nuclei Avg": pq_nuclei_avg,
        "DQ Nuclei1": dq_nuclei1,
        "DQ Nuclei2": dq_nuclei2,
        "DQ Nuclei3": dq_nuclei3,
        "DQ Nuclei Avg": dq_nuclei_avg,
        "SQ Nuclei1": sq_nuclei1,
        "SQ Nuclei2": sq_nuclei2,
        "SQ Nuclei3": sq_nuclei3,
        "SQ Nuclei Avg": sq_nuclei_avg,
        "PQ Tissue1": pq_tissue1,
        "PQ Tissue2": pq_tissue2,
        "PQ Tissue Avg": pq_tissue_avg,
        "DQ Tissue1": dq_tissue1,
        "DQ Tissue2": dq_tissue2,
        "DQ Tissue Avg": dq_tissue_avg,
        "SQ Tissue1": sq_tissue1,
        "SQ Tissue2": sq_tissue2,
        "SQ Tissue Avg": sq_tissue_avg,
    }

    # Convert the dictionary to a DataFrame
    df_results = pd.DataFrame.from_dict(results, orient='index').transpose()

    # Save the DataFrame to a CSV file
    output_csv_path = os.path.join(args.output_dir, f"{args.dataset}_evaluation_results.csv")
    df_results.to_csv(output_csv_path, index=False)

    print(f"Results saved to {output_csv_path}")
    
    # Create a summary dictionary with mean and std of all metrics
    summary = {
        "Metric": [],
        "Mean": [],
        "Std": []
    }
    
    # Add FPS
    summary["Metric"].append("FPS")
    summary["Mean"].append(1.0/(sum(FPS_score)/len(FPS_score)))
    summary["Std"].append(0.0)
    
    # Add all metrics with their means and standard deviations
    for metric_name, values in results.items():
        if metric_name != "Image ID" and metric_name != "FPS":  # Skip non-numeric columns
            values_array = np.array([x for x in values if not (np.isnan(x) or np.isinf(x))])
            if len(values_array) > 0:
                summary["Metric"].append(metric_name)
                summary["Mean"].append(np.mean(values_array))
                summary["Std"].append(np.std(values_array))
    
    # Convert summary to DataFrame and save as CSV
    df_summary = pd.DataFrame(summary)
    summary_csv_path = os.path.join(args.output_dir, f"{args.dataset}_evaluation_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    
    print(f"Summary results saved to {summary_csv_path}")
