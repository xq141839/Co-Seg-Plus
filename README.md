# *Co-Seg++*: Mutual Prompt-Guided Collaborative Learning for Versatile Medical Segmentation
<p align="center">
  <img src="figs/logo.png" alt="" width="150" height="150">
</p>

<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="semanticGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#4CAF50;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#8BC34A;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="instanceGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#2196F3;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#03DAC6;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="promptGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#FF9800;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FFC107;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="coreGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#E91E63;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#9C27B0;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#673AB7;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="titleGrad" x1="0%" y1="0%" x2="200%" y2="0%">
      <stop offset="0%" style="stop-color:#2196F3;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#4CAF50;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FF9800;stop-opacity:1" />
      <animateTransform attributeName="gradientTransform" type="translate" values="-200 0;200 0;-200 0" dur="4s" repeatCount="indefinite"/>
    </linearGradient>
  </defs>
  
  <!-- Background -->
  <rect width="400" height="200" fill="#ffffff"/>
  
  <!-- Intertwined Learning Paths -->
  <ellipse cx="80" cy="70" rx="30" ry="20" fill="none" stroke="url(#semanticGrad)" stroke-width="3" opacity="0.8">
    <animateTransform attributeName="transform" type="rotate" values="0 80 70;120 80 70;240 80 70;360 80 70" dur="6s" repeatCount="indefinite"/>
  </ellipse>
  
  <ellipse cx="120" cy="70" rx="20" ry="30" fill="none" stroke="url(#instanceGrad)" stroke-width="3" opacity="0.8">
    <animateTransform attributeName="transform" type="rotate" values="120 120 70;240 120 70;360 120 70;480 120 70" dur="6s" repeatCount="indefinite"/>
  </ellipse>
  
  <ellipse cx="100" cy="80" rx="25" ry="25" fill="none" stroke="url(#promptGrad)" stroke-width="3" opacity="0.8" transform="rotate(45 100 80)">
    <animateTransform attributeName="transform" type="rotate" values="45 100 80;165 100 80;285 100 80;405 100 80" dur="6s" repeatCount="indefinite"/>
  </ellipse>
  
  <!-- Weaving Lines -->
  <line x1="60" y1="50" x2="100" y2="65" stroke="url(#coreGrad)" stroke-width="2" opacity="0.6">
    <animate attributeName="opacity" values="0.3;1;0.3" dur="4s" repeatCount="indefinite"/>
  </line>
  <line x1="120" y1="50" x2="80" y2="90" stroke="url(#coreGrad)" stroke-width="2" opacity="0.6">
    <animate attributeName="opacity" values="0.3;1;0.3" dur="4s" begin="1s" repeatCount="indefinite"/>
  </line>
  <line x1="70" y1="100" x2="110" y2="60" stroke="url(#coreGrad)" stroke-width="2" opacity="0.6">
    <animate attributeName="opacity" values="0.3;1;0.3" dur="4s" begin="2s" repeatCount="indefinite"/>
  </line>
  
  <!-- Intersection Nodes -->
  <circle cx="85" cy="60" r="4" fill="url(#coreGrad)" opacity="0.8">
    <animate attributeName="r" values="4;6;4" dur="2s" repeatCount="indefinite"/>
  </circle>
  <circle cx="110" cy="85" r="4" fill="url(#coreGrad)" opacity="0.8">
    <animate attributeName="r" values="4;6;4" dur="2s" begin="0.5s" repeatCount="indefinite"/>
  </circle>
  <circle cx="90" cy="90" r="4" fill="url(#coreGrad)" opacity="0.8">
    <animate attributeName="r" values="4;6;4" dur="2s" begin="1s" repeatCount="indefinite"/>
  </circle>
  
  <!-- Central Core -->
  <circle cx="100" cy="75" r="18" fill="url(#coreGrad)" opacity="0.9">
    <animate attributeName="r" values="18;22;18" dur="3s" repeatCount="indefinite"/>
  </circle>
  <text x="100" y="82" text-anchor="middle" fill="white" font-family="Arial, sans-serif" font-size="14" font-weight="bold">++</text>
  
  <!-- Task Labels -->
  <rect x="40" y="25" width="50" height="16" rx="8" fill="url(#semanticGrad)" opacity="0.9"/>
  <text x="65" y="36" text-anchor="middle" fill="white" font-family="Arial, sans-serif" font-size="10" font-weight="bold">Semantic</text>
  
  <rect x="110" y="25" width="45" height="16" rx="8" fill="url(#instanceGrad)" opacity="0.9"/>
  <text x="132" y="36" text-anchor="middle" fill="white" font-family="Arial, sans-serif" font-size="10" font-weight="bold">Instance</text>
  
  <rect x="70" y="120" width="40" height="16" rx="8" fill="url(#promptGrad)" opacity="0.9"/>
  <text x="90" y="131" text-anchor="middle" fill="white" font-family="Arial, sans-serif" font-size="10" font-weight="bold">Prompt</text>
  
  <!-- Title -->
  <text x="220" y="80" font-family="Arial, sans-serif" font-size="42" font-weight="800" fill="url(#titleGrad)">Co-Seg++</text>
  <text x="220" y="100" font-family="Arial, sans-serif" font-size="12" fill="#666" font-weight="500" letter-spacing="1px">MEDICAL VERSATILE SEGMENTATION</text>
</svg>

<!-- <i>The icon is generated by recraft.ai.</i> -->


### [ArXiv Paper]() 

[Qing Xu](https://scholar.google.com/citations?user=IzA-Ij8AAAAJ&hl=en&authuser=1)<sup>1</sup> [Yuxiang Luo]()<sup>2</sup> [Wenting Duan](https://scholar.google.com/citations?user=H9C0tX0AAAAJ&hl=en&authuser=1)<sup>3</sup> [Zhen Chen](https://franciszchen.github.io/)<sup>4✉</sup> 

<sup>1</sup>UNNC &emsp; <sup>2</sup>Sichuan University &emsp; <sup>3</sup>Univeristy of Lincoln &emsp; <sup>4</sup>HKISI, CAS &emsp;

<sup>✉</sup> Corresponding Author. 

-------------------------------------------
![introduction](figs/method.png)

## 📰News

- **[2025.06.03]** We have released the code for Co-Seg++ !
## 🛠Setup

```bash
git clone https://github.com/xq141839/Co-Seg-Plus.git
cd Co-Seg-Plus
conda create -f Co-Seg-Plus.yaml
```

**Key requirements**: Cuda 12.2+, PyTorch 2.4+, mamba-ssm 2.1.0+

## 📚Data Preparation
- **PUMA**: [https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018)

The data structure is as follows.
```
HRMedSeg
├── datasets
│   ├── image_1024
│     ├── training_set_metastatic_roi_001.png
|     ├── ...
|   ├── mask_sem_1024
│     ├── training_set_metastatic_roi_001_nuclei.npy
|     ├── ...
|   ├── mask_ins_1024
│     ├── training_set_metastatic_roi_001_tissue.npy
|     ├── ...
|   ├── data_split.json
```
The json structure is as follows.

    { 
     "train": ['training_set_metastatic_roi_061.png'],
     "valid": ['training_set_metastatic_roi_002.png'],
     "test":  ['training_set_metastatic_roi_009.png'] 
     }

## 🎪Quickstart
* Train the Co-Seg++ with the default settings:
```python
python train.py --dataset data/$YOUR DATASET NAME$ --sam_pretrain pretrain/$SAM2 CHECKPOINT$
```

## Acknowledgements

* [SAM2](https://github.com/facebookresearch/sam2)
* [Medical-SAM-Adapter](https://github.com/SuperMedIntel/Medical-SAM-Adapter)
* [MambaVision](https://github.com/NVlabs/MambaVision)


