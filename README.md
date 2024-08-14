# [CVPR 2024 Highlight] Sat2Scene: 3D Urban Scene Generation from Satellite Images with Diffusion

[![arXiv](https://img.shields.io/badge/arXiv-2401.10786-b31b1b.svg)](https://arxiv.org/abs/2401.10786)
[![Paper](https://img.shields.io/badge/Paper-CVPR_2024_Highlight-243f7b.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Sat2Scene_3D_Urban_Scene_Generation_from_Satellite_Images_with_Diffusion_CVPR_2024_paper.html)
[![YouTube](https://img.shields.io/badge/YouTube-NqFy20zjFHU-ea3323.svg)](https://www.youtube.com/watch?v=NqFy20zjFHU)

This is the official code of the CVPR 2024 paper Sat2Scene: 3D Urban Scene Generation from Satellite Images with Diffusion. For more details, please refer to our paper and [project page](https://shinkyo0513.github.io/Sat2Scene/).

![Pipeline](https://github.com/shinkyo0513/Sat2Scene/blob/master/static/images/pipeline.jpg)

## Environment requirement

For 3D generation part
* `PyTorch 2.0.0+cu117`
* `MinkowskiEngine 0.5.4` [Minkowski Engine](https://nvidia.github.io/MinkowskiEngine/overview.html)
* `ema_pytorch`

For rendering part
* Coming soon

## Dataset preparation

Due to huge storage requirements and potential license issues, it is not easy to provide the processed dataset for downloading.
Here we provide instructions on how to process the dataset step-by-step.
Please first refer to [HoliCity](https://github.com/zhou13/holicity) dataset page to download the dataset and then follow the bullet points below.
* Resize all the panorama images to a resolution of `4096x2048` using `holicity_dataset/holicity_resize.py`.
* (__GPU required__) Run the segmentation model [ViT-Adapter](https://github.com/czczup/ViT-Adapter/tree/main/segmentation) by replacing the file `ViT-Adapter/segmentation/image_demo.py` with 
  `holicity_dataset/image_demo.py` and run the following script __in ViT-Adapter's repository__. The results will be saved in the folder `holicity_4096x2048_seg` with
  each element a `.npz` file containing the segmentation and a blended `.jpg` image for visualization.
  ```
  CUDA_VISIBLE_DEVICES=0 python image_demo.py \
    configs/cityscapes/mask2former_beit_adapter_large_896_80k_cityscapes_ss.py \
    checkpoints/mask2former_beit_adapter_large_896_80k_cityscapes.pth.tar \
    "holicity_4096x2048/*.jpg" \
    --out holicity_4096x2048_seg
  ```
* Create __scene__ point cloud dataset using `holicity_dataset/make_scene_dataset.py` for the 3D sparse diffusion model part.
  The resulting `.npz` files contain the following attributes.
  * `"coord"`: Nx3, float64
  * `"color"`: Nx3, uint8
  * `"dist"`: N, float64, for loss weight
  * `"sem"`: N, int64, for loss weight
  * `"geo_is_not_ground"`: N, bool, only for debug, __not used__ finally, can be omitted
* Create __single view__ point cloud dataset using `holicity_dataset/TODO` for the rendering part.
  The resulting `.XXX` files contain the following attributes.
  * ``: ???

## 3D sparse diffusion model

### Inference

The inference process can be run on NVIDIA TITAN RTX with 24GB memory. The trained model checkpoint can be downloaded [here](https://drive.google.com/file/d/1Ii4abHbRUtO6hrjc0JUWCuDARwBiaZ54/view?usp=drivesdk). Place the checkpoint file in the folder `checkpoint_folder`.

Exemplary point cloud files can be downloaded [here](https://drive.google.com/drive/folders/1ZBeuMITxBHB0-rbUUbz5dNoYRCSvJUdo?usp=sharing). Place the point cloud `.txt` file in the folder `point_cloud_files`. (__IMPORTANT__!) For customized point clouds, it is required that the unit of the coordinates is the __meter__ and the point density is __~400__ points per square meter sampled with [Poisson Disk](https://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.sample_points_poisson_disk.html).

After running the below script, the result files will be saved in the folder `checkpoint_folder/result_files`.

```
CUDA_VISIBLE_DEVICES=0 python3 denoising_diffusion_pytorch/denoising_diffusion_minkowski.py \
  --dataset_folder point_cloud_files \
  --dataset_mode test \
  --work_folder checkpoint_folder \
  --sampling_steps 1000 \
  --use_ema \
  --num_sample 1 \
  --point_scale 15 \
  --ckpt 218 \
  --save_folder result_files
```

[MeshLab](https://www.meshlab.net/) can visualize the exemplary point clouds (left) and the textures produced by the 3D sparse diffusion model (right).

![ex0](img/0.png)
![ex1](img/1.png)
![ex2](img/2.png)
![ex3](img/3.png)
![ex4](img/4.png)

### Training

Instruction coming soon.

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
