# Human Animaker  
This project is based on [TADA](https://github.com/TingtingLiao/TADA), generating textured animatable parametric model for a single human scan. 

<!-- https://github.com/TingtingLiao/hanimaker/assets/45743512/ef821ae4-1515-4303-9266-5c2a1a1fe5b9 -->

<!-- https://github.com/TingtingLiao/hanimaker/assets/45743512/61f6f681-c9b3-4795-ae02-e0f5db6ee877 -->

**Input vs Prediction:**

https://github.com/TingtingLiao/hanimaker/assets/45743512/d97aa23f-8208-4e61-b8cc-3fef67c97e7b

 
[gui_demo.webm](https://github.com/TingtingLiao/hanimaker/assets/45743512/a19ac688-bb85-4eb9-8ced-3b5d99699b03)

## Install 
```bash
conda create -n hanimaker python=3.9 
conda activate hanimaker 

# cuda12.1+torch2.3.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# pytorch3d 
pip install fvcore iopath 
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
 
pip install -r requirements.txt
```

## Data 
Download extra data [here](). 
 

## Usage   
```bash
# animatable avatar generation 
python main.py --obj_file data/example/0525/0525.obj --smplx_file data/example/0525/0525_smplx.pkl --save_dir ./out/0525
  
# animation with gui 
python animation_gui.py --smplx_params  out/0525/smplx_param.npy --motion_file data/aist/motions/gBR_sBM_cAll_d06_mBR3_ch06.pkl
```

### Citation 
```bibtex
@inproceedings{liao2024tada,
  title={{TADA! Text to Animatable Digital Avatars}},
  author={Liao, Tingting and Yi, Hongwei and Xiu, Yuliang and Tang, Jiaxiang and Huang, Yangyi and Thies, Justus and Black, Michael J.},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2024}
}
```