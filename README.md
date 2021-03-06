# Deep Recognition of Vanishing-Point-Constrained Building Planes in Urban Street Views
By Zhiliang ZENG, Mengyang WU, Wei ZENG, and Chi-Wing Fu

[2021/03/06: updated demo code & pretrained model]

## Introduction

This repository contains the inference code for our TIP paper: ['Deep Floor Plan Recognition Using a Multi-Task Network with Room-Boundary-Guided Attention'](https://ieeexplore.ieee.org/document/9068429). 

## Requirements

- Please install OpenCV
- Please install Python
- Please install tensorflow-gpu

Our origin code was written by using tensorflow-gpu==1.10.1 & Python2.7, now we tried to upgrade to tensorflow-gpu==1.14.0 & Python3.6. The code is not well tested, it may be a little different from the results shown in our paper.  

## Python packages

- [numpy]
- [scipy]
- [Pillow]
- [pylsd]
- [matplotlib]

## Vanishing point tool

We are using the method proposed by 'A-Contrario Horizon-First Vanishing Point Detection Using Second-Order Grouping Laws', you can find & download the Matlab tool [here](https://members.loria.fr/GSimon/software/v/).

## Data

Please send request email to: (zlzeng6@gmail.com)

## Usage

To use our demo code, please first download the our pretrained model [here](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155052510_link_cuhk_edu_hk/Ep4xSafmM5RAjqJkWJ-MyX4BfopFHy3Nuc9Udd2tusgJMA?e=Uc7dhW) and GeoNet [here](https://github.com/xjqi/GeoNet), unzip and put it into "pretrained" folder, then run

```bash
cd segmentation/
python infer.py
```

## Citation

If you find our work useful in your research, please consider citing:

---
	@ARTICLE{zlzeng2020deepbuildingplane,
	  author={Z. {Zeng} and M. {Wu} and W. {Zeng} and C. -W. {Fu}},
	  journal={IEEE Transactions on Image Processing}, 
	  title={Deep Recognition of Vanishing-Point-Constrained Building Planes in Urban Street Views}, 
	  year={2020},
	  volume={29},
	  number={},
	  pages={5912-5923},
	  doi={10.1109/TIP.2020.2986894}}

---