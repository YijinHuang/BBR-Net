## BBR-net
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4041331.svg)](https://doi.org/10.5281/zenodo.4041331)

This repository is the official implementation of the paper published in ISBI2020: ***Automated Hemorrhage Detection from Coarsely Annotated Fundus Images in Diabetic Retinopathy***.




### Overview
![Picture1](./img/pipeline.png)

Our proposed BBR-net and pipeline aims to refine coarsely-annotated dataset for object detection. In the paper, we present its application in hemorrhage detection in fundus images. Also, this approach works well in most of object detection tasks, especially tiny lesion detection task. 




### Usage
1. Besides the dataset you want to refine, a small dataset with well-annotated bounding boxes is required.

1. Use `utils/clahe_gaussian.py` to apply clach augmentation to your images.

2. Organize your well-annotated dataset in a csv file with formation `path_to_img.JPG,x1,y1,x2,y2,class` for each lesion (bounding box). For example:

```
img1.JPG,529,1796,596,1849
img1.JPG,692,1490,736,1530
img1.JPG,1922,170,1965,214
img2.JPG,1948,2013,2049,2107
img2.JPG,2354,1987,2442,2062
```

3. Run `utils/generate_coarse_patches.py` to generate a coarsely-annotated dataset with groundtruth.
4. Update the file path in `config.py` and train the BBR-net by running `main.py`.
6. Use `utils/refine.py` to refine your coarsely-annotated dataset with trained BBR-net.




### Acknowledgements

The BBR-net is built on the torchvision implementation of the ResNet.



### Citation

```
@inproceedings{huang2020automated,
  title={Automated Hemorrhage Detection from Coarsely Annotated Fundus Images in Diabetic Retinopathy},
  author={Huang, Yijin and Lin, Li and Li, Meng and Wu, Jiewei and Cheng, Pujin and Wang, Kai and Yuan, Jin and Tang, Xiaoying},
  booktitle={2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)},
  pages={1369--1372},
  year={2020},
  organization={IEEE}
}
```

