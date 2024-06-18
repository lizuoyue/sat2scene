# [CVPR 2024 Highlight] Sat2Scene: 3D Urban Scene Generation from Satellite Images with Diffusion

[![arXiv](https://img.shields.io/badge/arXiv-2401.10786-b31b1b.svg)](https://arxiv.org/abs/2401.10786)
[![Paper](https://img.shields.io/badge/Paper-CVPR_2024_Highlight-243f7b.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Sat2Scene_3D_Urban_Scene_Generation_from_Satellite_Images_with_Diffusion_CVPR_2024_paper.html)
[![YouTube](https://img.shields.io/badge/YouTube-NqFy20zjFHU-ea3323.svg)](https://www.youtube.com/watch?v=NqFy20zjFHU)

This is XXX/
[Project page] (https://shinkyo0513.github.io/Sat2Scene/)

[![Pipeline](https://github.com/shinkyo0513/Sat2Scene/blob/master/static/images/pipeline.jpg)]

## Environment requirement
* `PyTorch 2.0.0+cu117`
* `MinkowskiEngine 0.5.4` [Minkowski Engine](https://nvidia.github.io/MinkowskiEngine/overview.html)
* `ema_pytorch`


## 3D sparse diffusion model

The trained model checkpoint can be downloaded [here]().
Exemplary point cloud files

```
CUDA_VISIBLE_DEVICES=0 python3 denoising_diffusion_pytorch/denoising_diffusion_minkowski.py \
  --dataset_folder "/path/to/your/dataset/folder/containing/txt/point/cloud/files" \
  --dataset_mode test \
  --work_folder folder_containing_ckpt_file \
  --sampling_steps 1000 \
  --use_ema \
  --num_sample 1 \
  --point_scale 15 \
  --ckpt 218 \
  --save_folder your_save_folder
```

Training code coming soon.

## Rendering
Training and inference code coming soon.

## BibTeX

```
@InProceedings{li2024sat2scene,
    author    = {Li, Zuoyue and Li, Zhenqiang and Cui, Zhaopeng and Pollefeys, Marc and Oswald, Martin R.},
    title     = {Sat2Scene: 3D Urban Scene Generation from Satellite Images with Diffusion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {7141-7150}
}
```
