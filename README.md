# Human Animaker  
This project is based on [TADA](https://github.com/TingtingLiao/TADA), generating textured animatable parametric model from a single human scan. 
 
**Input vs Predict**

https://github.com/TingtingLiao/hanimaker/assets/45743512/240a9479-b1c5-4a47-addd-37f3a775bd91

[gui_demo.webm](https://github.com/TingtingLiao/hanimaker/assets/45743512/426beec9-3cbe-4fc9-bcd7-f2f6c3f56950)


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
- Download extra data [here](https://huggingface.co/Luffuly/hanimaker/tree/main). 
- (Optional) For input scans and smplx you can use [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset). 

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