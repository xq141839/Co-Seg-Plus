# *Co-Seg++*: Mutual Prompt-Guided Collaborative Learning for Versatile Medical Segmentation
<p align="center">
  <img src="figs/logo.svg" alt="" width="600" height="200">
</p>


[[`arXiv`](https://arxiv.org/abs/2506.17159)] 

[Qing Xu](https://scholar.google.com/citations?user=IzA-Ij8AAAAJ&hl=en&authuser=1)<sup>1</sup> [Yuxiang Luo]()<sup>2</sup> [Wenting Duan](https://scholar.google.com/citations?user=H9C0tX0AAAAJ&hl=en&authuser=1)<sup>3</sup> [Zhen Chen](https://franciszchen.github.io/)<sup>4✉</sup> 

<sup>1</sup>University of Nottingham &emsp; <sup>2</sup>Sichuan University &emsp; <sup>3</sup>Univeristy of Lincoln &emsp; <sup>4</sup>HKISI, CAS &emsp;

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
- **PUMA**: [Challenge Link](https://puma.grand-challenge.org/)

The data structure is as follows.
```
Co-Seg-Plus
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

## 📜Citation
If you find this work helpful for your project, please consider citing the following paper:
```
@article{xu2025co,
  title={Co-Seg++: Mutual Prompt-Guided Collaborative Learning for Versatile Medical Segmentation},
  author={Xu, Qing and Luo, Yuxiang and Duan, Wenting and Chen, Zhen},
  journal={arXiv preprint arXiv:2506.17159},
  year={2025}
}
```

## Acknowledgements

* [SAM2](https://github.com/facebookresearch/sam2)
* [Medical-SAM-Adapter](https://github.com/SuperMedIntel/Medical-SAM-Adapter)
* [MambaVision](https://github.com/NVlabs/MambaVision)


