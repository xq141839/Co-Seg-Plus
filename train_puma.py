import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sympy import ff
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from dataloader import PUMALoader
from loss import *
from tqdm import tqdm
import json
from model import CoSeg
import hydra
from functools import partial
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from sam2.build_sam import build_sam2
from torchmetrics.classification import BinaryAccuracy
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks import one_hot

torch.set_num_threads(8)
# matplotlib.use('TkAgg')

def train_model(model, optimizer, scheduler, num_epochs=5):
    
    since = time.time()
    
    best_model_wts = model.state_dict()

    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
    
            else:
                model.train(False)  

            running_loss_binary = []
            running_loss_mse = []
            running_loss_msge = []
            running_loss_type = []
            running_loss_all = []
            running_loss_sem = []

            running_dice_ins_lymphocyte = []
            running_dice_ins_tumor = []
            running_dice_ins_other = []
            running_dice_ins_binary = []
            running_dice_sem_stroma = []
            running_dice_sem_tumor = []
            running_dice_ins_avg = []
            running_dice_sem_avg = []
        
            # Iterate over data
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            num_samples = 0

            for data_dict in tqdm(dataloaders[phase]):      
                # wrap them in Variable

                img = Variable(data_dict["image"].cuda())
                tissue_map = Variable(data_dict["tissue_map"].cuda()).unsqueeze(1)
                nuclei_tissue_binary = Variable(data_dict["nuclei_tissue_map_res"].cuda()).unsqueeze(1)
                nuclei_binary_map = Variable(data_dict["nuclei_binary_map"].cuda()).unsqueeze(1)
                nuclei_binary_res = Variable(data_dict["nuclei_binary_map_res"].cuda()).unsqueeze(1)
                nuclei_type_map = Variable(data_dict["nuclei_type_map"].cuda()).unsqueeze(1)
                nuclei_inst_map = Variable(data_dict["nuclei_inst_map"].cuda()).unsqueeze(1)
                nuclei_hv_map = Variable(data_dict["nuclei_hv_map"].cuda())


                if phase == 'train':
                # zero the parameter gradients
                    optimizer.zero_grad()

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred_mask_ins, pred_mask_sem, pred_prob_ins, pred_prob_sem = model(x=img)

                    ff_nuclei_loss = dice_ce_loss_b(pred_mask_ins, nuclei_binary_res)
                    ff_tissue_loss = dice_ce_loss_b(pred_mask_sem, nuclei_tissue_binary)
                    log_prob1 = F.log_softmax(pred_prob_sem, dim=1)
                    prob2 = F.softmax(pred_prob_ins, dim=1)

                    log_prob2 = F.log_softmax(pred_prob_ins, dim=1)
                    prob1 = F.softmax(pred_prob_sem, dim=1)

                    kl_loss_symmetric = (F.kl_div(log_prob1, prob2, reduction='mean') + 
                    F.kl_div(log_prob2, prob1, reduction='mean')) / 2

                    loss_forward_1 = ff_nuclei_loss + ff_tissue_loss + kl_loss_symmetric * 10
                    # print(ff_nuclei_loss.item(), ff_tissue_loss.item(), kl_loss_symmetric.item() * 10)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred_mask_ins, pred_mask_sem = model(x=img, 
                                                            prob_ins=pred_mask_ins, 
                                                            prob_sem=pred_mask_sem)
                        
                    nuclei_binary_map_pred = pred_mask_ins[:,0:1,:,:]
                    nuclei_type_map_pred = pred_mask_ins[:,1:1+args.ins_cls,:,:]
                    nuclei_hv_map_pred = pred_mask_ins[:,1+args.ins_cls:3+args.ins_cls,:,:]

                    hv_loss1 = mse_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map)
                    hv_loss2 = msge_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map, focus=nuclei_binary_map, device='cuda')
                    binary_loss = dice_ce_loss_b(nuclei_binary_map_pred, nuclei_binary_map)
                    multi_loss = dice_ce_loss_m(nuclei_type_map_pred, nuclei_type_map)
                    score_mask_ins_b = accuracy_metric(nuclei_binary_map_pred, nuclei_binary_map)
                    score_mask_ins_m = accuracy_metric_m(nuclei_type_map_pred, nuclei_type_map)
                    loss_sem = dice_ce_loss_m(pred_mask_sem, tissue_map)
                    score_mask_sem = accuracy_metric_m(pred_mask_sem, tissue_map)
                    score_mask_sem = torch.nan_to_num(score_mask_sem)
                    score_mask_ins_m = torch.nan_to_num(score_mask_ins_m)
                    loss = 0.5*loss_forward_1 + multi_loss + binary_loss + 8 * hv_loss1 + 2.5 * hv_loss2 + loss_sem * 0.5

                    loss.backward()
                    optimizer.step()

                else:
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            pred_mask_ins, pred_mask_sem, pred_prob_ins, pred_prob_sem = model(x=img)

                        ff_nuclei_loss = dice_ce_loss_b(pred_mask_ins, nuclei_binary_res)
                        ff_tissue_loss = dice_ce_loss_b(pred_mask_sem, nuclei_tissue_binary)
                        log_prob1 = F.log_softmax(pred_prob_sem, dim=1)
                        prob2 = F.softmax(pred_prob_ins, dim=1)

                        log_prob2 = F.log_softmax(pred_prob_ins, dim=1)
                        prob1 = F.softmax(pred_prob_sem, dim=1)

                        kl_loss_symmetric = (F.kl_div(log_prob1, prob2, reduction='mean') + 
                        F.kl_div(log_prob2, prob1, reduction='mean')) / 2

                        loss_forward_1 = ff_nuclei_loss + ff_tissue_loss + kl_loss_symmetric * 10
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            pred_mask_ins, pred_mask_sem = model(x=img, 
                                                            prob_ins=pred_mask_ins, 
                                                            prob_sem=pred_mask_sem)

                        nuclei_binary_map_pred = pred_mask_ins[:,0:1,:,:]
                        nuclei_type_map_pred = pred_mask_ins[:,1:1+args.ins_cls,:,:]
                        nuclei_hv_map_pred = pred_mask_ins[:,1+args.ins_cls:3+args.ins_cls,:,:]
                        
                        hv_loss1 = mse_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map)
                        hv_loss2 = msge_loss(input=nuclei_hv_map_pred, target=nuclei_hv_map, focus=nuclei_binary_map, device='cuda')
                        binary_loss = dice_ce_loss_b(nuclei_binary_map_pred, nuclei_binary_map)
                        multi_loss = dice_ce_loss_m(nuclei_type_map_pred, nuclei_type_map)
                        score_mask_ins_b = accuracy_metric(nuclei_binary_map_pred, nuclei_binary_map)
                        score_mask_ins_m = accuracy_metric_m(nuclei_type_map_pred, nuclei_type_map)
                        loss_sem = dice_ce_loss_m(pred_mask_sem, tissue_map)
                        score_mask_sem = accuracy_metric_m(pred_mask_sem, tissue_map)
                        score_mask_sem = torch.nan_to_num(score_mask_sem)
                        score_mask_ins_m = torch.nan_to_num(score_mask_ins_m)

                    loss = 0.5*loss_forward_1+multi_loss + binary_loss + 8 * hv_loss1 + 2.5 * hv_loss2 + loss_sem * 0.5
                
                # calculate loss and IoU
                running_loss_all.append(loss.item())
                running_loss_type.append(multi_loss.item())
                running_loss_binary.append(binary_loss.item())
                running_loss_sem.append(loss_sem.item() * 0.5)
                running_loss_mse.append(hv_loss1.item() * 8)
                running_loss_msge.append(hv_loss2.item() * 2.5)

                num_iter = args.batch
                if phase == 'valid':
                    num_iter = 1
                for i in range(num_iter):
                    running_dice_ins_lymphocyte.append(score_mask_ins_m[i,1].item())
                    running_dice_ins_tumor.append(score_mask_ins_m[i,2].item())
                    running_dice_ins_other.append(score_mask_ins_m[i,3].item())
                    running_dice_ins_avg.append(torch.mean(score_mask_ins_m[i,1:]).item())
                    running_dice_ins_binary.append(score_mask_ins_b[i].item())
                    running_dice_sem_stroma.append(score_mask_sem[i,1].item())
                    running_dice_sem_tumor.append(score_mask_sem[i,2].item())
                    running_dice_sem_avg.append(torch.mean(score_mask_sem[i,1:]).item())


                
            epoch_loss = np.mean(running_loss_all)

            print('{} Loss ALL: {:.4f} Type: {:.4f} Binary: {:.4f} MSE: {:.4f} MSGE: {:.4f} Tissue: {:.4f}'.format(
                phase, epoch_loss, np.mean(running_loss_type), np.mean(running_loss_binary), np.mean(running_loss_mse), 
                np.mean(running_loss_msge), np.mean(running_loss_sem)))
            
            print('{} Tissue Stroma: {:.4f} Tumor: {:.4f} Avg: {:.4f}'.format(
                phase, np.mean(running_dice_sem_stroma), np.mean(running_dice_sem_tumor), np.mean(running_dice_sem_avg)))
            
            print('{} Instance Lymphocyte: {:.4f} Tumor: {:.4f} Other: {:.4f} Binary: {:.4f} Avg: {:.4f}'.format(
                phase, np.mean(running_dice_ins_lymphocyte), np.mean(running_dice_ins_tumor), np.mean(running_dice_ins_other),
                np.mean(running_dice_ins_binary), np.mean(running_dice_ins_avg)))

            # save parameters
            if phase == 'valid':
                save_point = epoch % 2
                if (epoch_loss <= best_loss) or (save_point == 0):
                    if epoch_loss <= best_loss and epoch > 50:
                        best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    torch.save(best_model_wts, f'outputs/coseg_{args.dataset}_{epoch}.pth')

                scheduler.step()
                print(f"lr: {scheduler.get_last_lr()[0]}")
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='puma', help='data directory')  
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--batch', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=80, help='epoches')
    parser.add_argument('--ins_cls', type=int, default=4, help='number of instance classes (including background)')
    parser.add_argument('--sem_cls', type=int, default=3, help='number of semantic classes (including background)')
    parser.add_argument('--sam_pretrain', type=str,default='pretrain/sam2_hiera_large.pt', help='pretrain weight for SAM2')
    args = parser.parse_args()

    os.makedirs('outputs/', exist_ok=True)

    jsonfile1 = f'datasets/{args.dataset}/data_split.json'
    
    with open(jsonfile1, 'r') as f:
        df1 = json.load(f)
    
    val_files = df1['valid']
    train_files = df1['train']

    train_dataset = PUMALoader(args.dataset, train_files, A.Compose([
        A.Resize(256, 256),
        ], 
        additional_targets={'mask2': 'mask','mask3': 'mask','mask4': 'mask','mask5': 'mask'}))
    val_dataset = PUMALoader(args.dataset, val_files, A.Compose([
        A.Resize(256, 256),
        ],
        additional_targets={'mask2': 'mask','mask3': 'mask','mask4': 'mask','mask5': 'mask'}))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    
    dataloaders = {'train':train_loader,'valid':val_loader}
    
    model_cfg = "sam2_hiera_l.yaml"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    model = CoSeg(build_sam2(model_cfg, args.sam_pretrain, mode='train'))


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    model = model.cuda()

    for n, value in model.model.image_encoder.named_parameters():
        if (f"edge" in n) or (f"neck" in n):
            value.requires_grad = True
        else:
            value.requires_grad = False

    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )
    print('Trainable Params = ' + str(trainable_params/1000**2) + 'M')
    total_params = sum(
	param.numel() for param in model.parameters()
    )
    print('Total Params = ' + str(total_params/1000**2) + 'M')
    print('Ratio = ' + str(trainable_params/total_params*100) + '%')

        
    # Loss, IoU and Optimizer
    dice_ce_loss_b = DiceCELoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True)
    dice_ce_loss_m = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, sigmoid=False) 
    mse_loss = MSELossMaps()
    msge_loss = MSGELossMaps()
    accuracy_metric = DiceEval(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True) #BinaryIoU()
    accuracy_metric_m = DiceEval(include_background=True, to_onehot_y=True, softmax=True, sigmoid=False)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = args.lr)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    train_model(model, optimizer, exp_lr_scheduler, num_epochs=args.epoch)