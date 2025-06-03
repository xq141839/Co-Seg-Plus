import os
from skimage import io, transform, color, img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as pytorch_transforms
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensor 
import albumentations as A
from scipy.ndimage import center_of_mass, distance_transform_edt
import json

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
        """Obtain the horizontal and vertical distance maps for each
        nuclear instance.

        Args:
            inst_map (np.ndarray): Instance map with each instance labelled as a unique integer
                Shape: (H, W)
        Returns:
            np.ndarray: Horizontal and vertical instance map.
                Shape: (2, H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
                last is vertical (vertical gradient (-1 to 1))
        """
        orig_inst_map = inst_map.copy()  # instance ID map

        x_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(orig_inst_map))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(orig_inst_map == inst_id, np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            if inst_box[0] >= 2:
                inst_box[0] -= 2
            if inst_box[2] >= 2:
                inst_box[2] -= 2
            if inst_box[1] <= orig_inst_map.shape[0] - 2:
                inst_box[1] += 2
            if inst_box[3] <= orig_inst_map.shape[0] - 2:
                inst_box[3] += 2

            # improvement
            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        hv_map = np.stack([x_map, y_map])
        return hv_map

class PUMALoader(Dataset):
        def __init__(self, data_name, jsfiles, transforms, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = f'/home/Qing_Xu/miccai2025/nuclei/datasets/{data_name}'
            self.data_name = data_name
            self.jsfiles = jsfiles
            self.img_tesnor = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ])
            self.transforms = transforms
            self.sam_img_size = 1024
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_mean).view(-1, 1, 1)
            
        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
            image_id = list(self.jsfiles[idx].split('.'))[0]

            image_path = os.path.join(self.path,'image_1024/',image_id)
            tissue_path = os.path.join(self.path,f'mask_tissue_1024_ori/',image_id)
            nuclei_path = os.path.join(self.path,f'mask_nuclei3_1024/',image_id)
 
    
            img = io.imread(image_path+'.png')[:,:,:3].astype('float32') 
            nuclei_tissue_map = np.load(tissue_path+'_tissue.npy').astype(np.uint8)
            nuclei_tissue_map = nuclei_tissue_map[0]
            nuclei_tissue_binary = nuclei_tissue_map.copy().astype(np.uint8)
            nuclei_tissue_binary[nuclei_tissue_binary > 0] = 1
            nuclei_map = np.load(nuclei_path+'_nuclei.npy')
  
            nuclei_binary_map = nuclei_map[0, :, :, 1].copy().astype(np.uint8)
            nuclei_binary_map[nuclei_binary_map > 0] = 1
            nuclei_type_map = nuclei_map[0, :, :, 1].copy().astype(np.uint8)
            nuclei_inst_map = nuclei_map[0, :, :, 0].copy()
            nuclei_hv_map = gen_instance_hv_map(nuclei_inst_map)

            
            data_group = self.transforms(image=img, mask=nuclei_tissue_binary, mask2=nuclei_type_map, mask3=nuclei_binary_map)
            img_re = data_group['image']
            nuclei_tissue_map_res = data_group['mask']
            nuclei_type_map_res = data_group['mask2']
            nuclei_binary_map_res = data_group['mask3']
 
            img = self.img_tesnor(img)
            img = self.preprocess(img)

            data_dict = {"image_id": image_id,
                        "image": img,
                        "image_res": img_re,
                        "tissue_map": nuclei_tissue_map,
                        "nuclei_binary_map": nuclei_binary_map,
                        "nuclei_type_map": nuclei_type_map,
                        "nuclei_inst_map": nuclei_inst_map,
                        "nuclei_hv_map": nuclei_hv_map,
                        "nuclei_tissue_map_res": nuclei_tissue_map_res,
                        "nuclei_type_map_res": nuclei_type_map_res,
                        "nuclei_binary_map_res": nuclei_binary_map_res
            }

            return data_dict
        
        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.sam_img_size - h
            padw = self.sam_img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x
        
        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.sam_img_size - h
            padw = self.sam_img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x
        

class CerberusLoader(Dataset):
        def __init__(self, data_name, jsfiles, transforms, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = f'/home/Qing_Xu/miccai2025/nuclei/datasets/{data_name}'
            self.data_name = data_name
            self.jsfiles = jsfiles
            self.img_tesnor = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ])
            self.transforms = transforms
            self.sam_img_size = 1024
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_mean).view(-1, 1, 1)
            
        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
            image_id = list(self.jsfiles[idx].split('.'))[0]
                
            image_path = os.path.join(self.path, 'nuclei','processed_images',image_id)
            tissue_path = os.path.join(self.path, 'gland','processed_labels',image_id)
            nuclei_path = os.path.join(self.path,'nuclei','processed_labels',image_id)
   
            img = io.imread(image_path+'.png')[:,:,:3].astype('float32')

            nuclei_tissue_map = np.load(tissue_path+'.npy').astype(np.uint8)
            nuclei_tissue_map = nuclei_tissue_map[:, :, 1]
            nuclei_tissue_map[nuclei_tissue_map > 0] = 255
            nuclei_map = np.load(nuclei_path+'.npy')
  
            nuclei_binary_map = nuclei_map[:, :, 1].copy().astype(np.uint8)
            nuclei_binary_map[nuclei_binary_map > 0] = 255
            nuclei_type_map = nuclei_map[:, :, 1].copy().astype(np.uint8)
            nuclei_inst_map = nuclei_map[:, :, 0].copy()
            nuclei_type_map[nuclei_type_map == 3] = 1
            nuclei_type_map[nuclei_type_map == 4] = 1
            nuclei_type_map[nuclei_type_map == 5] = 1
            nuclei_type_map[nuclei_type_map == 6] = 3
            nuclei_hv_map = gen_instance_hv_map(nuclei_inst_map)
            
            data_group = self.transforms(image=img, mask=nuclei_tissue_map, mask2=nuclei_type_map, mask3=nuclei_binary_map)
            img_re = data_group['image']
            nuclei_tissue_map_res = data_group['mask']
            nuclei_type_map_res = data_group['mask2']
            nuclei_binary_map_res = data_group['mask3']

 
            img = self.img_tesnor(img)
            img = self.preprocess(img)

            data_dict = {"image_id": image_id,
                        "image": img,
                        "image_res": img_re,
                        "tissue_map": nuclei_tissue_map,
                        "nuclei_binary_map": nuclei_binary_map,
                        "nuclei_type_map": nuclei_type_map,
                        "nuclei_inst_map": nuclei_inst_map,
                        "nuclei_hv_map": nuclei_hv_map,
                        "nuclei_tissue_map_res": nuclei_tissue_map_res,
                        "nuclei_type_map_res": nuclei_type_map_res,
                        "nuclei_binary_map_res": nuclei_binary_map_res
            }

            return data_dict
        
        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.sam_img_size - h
            padw = self.sam_img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x
        
class GlasLoader(Dataset):
        def __init__(self, data_name, jsfiles, transforms, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = f'/home/Qing_Xu/miccai2025/nuclei/datasets/{data_name}'
            self.data_name = data_name
            self.jsfiles = jsfiles
            self.img_tesnor = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ])
            self.transforms = transforms
            self.sam_img_size = 1024
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_mean).view(-1, 1, 1)
            map_js = f'/home/Qing_Xu/miccai2025/nuclei/datasets/glas/image_mapping.json'               
            with open(map_js, 'r') as f:
                self.map_id = json.load(f)
        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
            image_id = list(self.jsfiles[idx].split('.'))[0]
            gland_id = self.map_id[self.jsfiles[idx]]
            gland_id = list(gland_id.split('.'))[0]

            image_path = os.path.join(self.path, 'nuclei','images',image_id)
            tissue_path = os.path.join(self.path, 'gland','label',gland_id)
            nuclei_path = os.path.join(self.path,'nuclei','npy',image_id)
 
    
            img = io.imread(image_path+'.png')[:,:,:3].astype('float32') 
            nuclei_tissue_map = cv2.imread(tissue_path+'_anno.png', 0)
            nuclei_tissue_map[nuclei_tissue_map > 0] = 255
            nuclei_map = np.load(nuclei_path+'.npy')
  
            nuclei_binary_map = nuclei_map[:, :, 1].copy().astype(np.uint8)
            nuclei_binary_map[nuclei_binary_map > 0] = 255
            nuclei_type_map = nuclei_map[:, :, 1].copy().astype(np.uint8)
            nuclei_type_map[nuclei_type_map == 3] = 1
            nuclei_type_map[nuclei_type_map == 4] = 1
            nuclei_type_map[nuclei_type_map == 5] = 1
            nuclei_type_map[nuclei_type_map == 6] = 3
            nuclei_inst_map = nuclei_map[:, :, 0].copy()
            # print(np.unique(nuclei_inst_map))
            nuclei_hv_map = gen_instance_hv_map(nuclei_inst_map)

            
            data_group = self.transforms(image=img, mask=nuclei_tissue_map, mask2=nuclei_type_map, mask3=nuclei_binary_map)
            img_re = data_group['image']
            nuclei_tissue_map_res = data_group['mask']
            nuclei_type_map_res = data_group['mask2']
            nuclei_binary_map_res = data_group['mask3']
 
            img = self.img_tesnor(img)
            img = self.preprocess(img)

            data_dict = {"image_id": image_id,
                        "image": img,
                        "image_res": img_re,
                        "tissue_map": nuclei_tissue_map,
                        "nuclei_binary_map": nuclei_binary_map,
                        "nuclei_type_map": nuclei_type_map,
                        "nuclei_inst_map": nuclei_inst_map,
                        "nuclei_hv_map": nuclei_hv_map,
                        "nuclei_tissue_map_res": nuclei_tissue_map_res,
                        "nuclei_type_map_res": nuclei_type_map_res,
                        "nuclei_binary_map_res": nuclei_binary_map_res
            }

            return data_dict
        
        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.sam_img_size - h
            padw = self.sam_img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x
        

class tf2Loader(Dataset):
        def __init__(self, data_name, jsfiles, transforms, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = f'/home/Qing_Xu/miccai2025/nuclei/datasets/{data_name}'
            self.data_name = data_name
            self.jsfiles = jsfiles
            self.img_tesnor = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ])
            self.transforms = transforms
            self.sam_img_size = 1024
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_mean).view(-1, 1, 1)
        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
            image_id = list(self.jsfiles[idx].split('.'))[0]

            image_path = os.path.join(self.path,'images',image_id)
            tissue_path = os.path.join(self.path, 'masks','sem',image_id)
            nuclei_path = os.path.join(self.path,'masks','npy',image_id)
 
    
            img = io.imread(image_path+'.png')[:,:,:3].astype('float32') 
            nuclei_tissue_map = np.load(tissue_path+'.npy').astype(np.uint8)
            nuclei_map = np.load(nuclei_path+'.npy')
  
            nuclei_binary_map = nuclei_map[:, :, 1].copy().astype(np.uint8)
            nuclei_binary_map[nuclei_binary_map > 0] = 1
            nuclei_type_map = nuclei_map[:, :, 1].copy().astype(np.uint8)
            nuclei_inst_map = nuclei_map[:, :, 0].copy()
            nuclei_hv_map = gen_instance_hv_map(nuclei_inst_map)

            
            data_group = self.transforms(image=img, mask=nuclei_tissue_map, mask2=nuclei_type_map, mask3=nuclei_binary_map)
            img_re = data_group['image']
            nuclei_tissue_map_res = data_group['mask']
            nuclei_type_map_res = data_group['mask2']
            nuclei_binary_map_res = data_group['mask3']
 
            img = self.img_tesnor(img)
            img = self.preprocess(img)

            data_dict = {"image_id": image_id,
                        "image": img,
                        "image_res": img_re,
                        "tissue_map": nuclei_tissue_map,
                        "nuclei_binary_map": nuclei_binary_map,
                        "nuclei_type_map": nuclei_type_map,
                        "nuclei_inst_map": nuclei_inst_map,
                        "nuclei_hv_map": nuclei_hv_map,
                        "nuclei_tissue_map_res": nuclei_tissue_map_res,
                        "nuclei_type_map_res": nuclei_type_map_res,
                        "nuclei_binary_map_res": nuclei_binary_map_res
            }

            return data_dict
        
        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.sam_img_size - h
            padw = self.sam_img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x



