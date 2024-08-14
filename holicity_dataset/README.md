# Dataset preparation

Due to huge storage requirements and potential license issues, it is not easy to provide the processed dataset for downloading.
Here we provide instructions on how to process the dataset step-by-step.
Please first refer to [HoliCity](https://github.com/zhou13/holicity) dataset page to download the dataset and then follow the bullet points below.

* Place the dataset in the folder `holicity_dataset/HoliCity` and the dataset structure should be similar to below.
  ```
  ./sat2scene/holicity_dataset/HoliCity
      - 2008-07
          - *_camr.npz
          - *_dpth.npz
          - *_imag.jpg
          - *_nrml.npz
          - *_plan.npz
          - *_plan.png
          - *_sgmt.png
          - *_vpts.npz
          ...
      - 2008-09
      - 2008-10
      ...
      - 2019-05
      - split-all-v1-bugfix
          - filelist.txt
          - test-middlesplit.txt
          - test-randomsplit.txt
          - train-middlesplit.txt
          - train-randomsplit.txt
          - valid-middlesplit.txt
  ```




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
