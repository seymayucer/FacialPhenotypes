# Measuring Hidden Bias within Face Recognition via Racial Phenotype
Face Verification and Identification Test Code for the study: Understanding Racial Bias using Facial Phenotypes.

### Abstract

Recent work reports disparate performance for intersectional racial groups across face recognition tasks: face verification and identification. However, the definition of those racial groups has a significant impact on the underlying findings of such racial bias analysis. Previous studies define these groups based on either demographic information (e.g. African, Asian etc.) or skin tone (e.g. lighter or darker skins). The use of such sensitive or broad group definitions has disadvantages for bias investigation and subsequent counter-bias solutions design. By contrast, this study introduces an alternative racial bias analysis methodology via facial phenotype attributes for face recognition. We use the set of observable characteristics of an individual face where a race-related facial phenotype is hence specific to the human face and correlated to the racial profile of the subject. We propose categorical test cases to investigate the individual influence of those attributes on bias within face recognition tasks. We compare our phenotype-based grouping methodology with previous grouping strategies and show that phenotype-based groupings uncover hidden bias without reliance upon any potentially protected attributes or ill-defined grouping strategies. Furthermore, we contribute corresponding phenotype attribute category labels for two face recognition tasks: RFW for face verification and VGGFace2 (test set) for face identification.

## Installation


To start working, you will need to install the following packages:

- PyTorch
- thundersvm
- logging
- cv2

Or you can install all using pip or conda: ` pip install -r setup/requirements.txt `.


## Reproducing experiment results

Take the following steps to reproduce the performance reported in the paper:


1- For face verification, please install [RFW dataset](http://www.whdeng.cn/RFW/testing.html) and for face identification [VGGFace 2.](https://drive.google.com/file/d/1jdZw6ZmB7JRK6RS6QP3YEr2sufJ5ibtO/view) 

2- For both tasks, images should be aligned to 112x112. You can align images using preprocess/face_align.py

~~~
python preprocess/face_align.py --data_dir /path/to/dataset --output_dir /path/to/destination/vggface2_train_224_aligned/ --landmark_file '/path/to/VGGFace2/bb_landmark/loose_landmark_train.csv'
~~~ 

3- To download pre-trained models and labels please visit [here](http://doi.org/10.15128/r2hm50tr746). After installation, please put model.ckpt to model/ folder and put FDA files to test_assets/ folder.

4- Face verification test:

~~~
python face_verification.py --data_dir path/to/rfw/aligned/ --model_dir ./model/model.ckpt --pair_file ./test_assets/pairs/AttributePairs/eye_monolid_pairs_6000_selected.csv 
~~~

5- Face identification test:
~~~
python face_identification.py --data_dir  path/to/vgg/aligned/ --model_dir ./model/model.ckpt --img_list_file AttributeLabels/test_list.txt 
~~~

## Attribute Labels

Attribute categorization is presented below.

| **Attribute**  | **Categories**               |
|---------------------|-------------------------------------|
| **Skin Type**  | Type 1 / 2 / 3 / 4 / 5 / 6          |
| **Eye Shape**  | Monolid / Other                     |
| **Nose Shape** | Wide / Narrow                       |
| **Lip Shape**  | Full / Small                         |
| **Hair Type**  | Straight / Wavy / Curly / Bald      |
| **Hair Color** | Red / Blonde / Brown / Black / Gray |

The distribution of race-relavent phenotype attributes of [RFW](https://github.com/seymayucer/RacialPhenotypesFREvaluation/blob/main/figures/rfw-phenotype-dist.pdf) and [VGGFace2 test](https://github.com/seymayucer/RacialPhenotypesFREvaluation/blob/main/figures/vggtest-phenotype-dist.pdf) datasets.


## References
If you are making use of this work in any way (including our pre-trained models or datasets), you must please reference the following articles in any report, publication, presentation, software release or any other associated materials:

```
@InProceedings{yucermeasuring,
  author = {Yucer, S. and Tektas, F. and Al Moubayed, N. and Breckon, T.P.},
  title = {Measuring Hidden Bias within Face Recognition via Racial Phenotypes},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2022},
  publisher = {IEEE},
  url = {https://breckon.org/toby/publications/papers/yucer22phenotypes.pdf},
  arxiv = {http://arxiv.org/abs/2110.09839},
}
```


[Measuring Hidden Bias within Face Recognition via Racial Phenotypes](https://breckon.org/toby/publications/papers/yucer22phenotypes.pdf)
(Yucer, Tektas, Al Moubayed, Breckon), IEEE/CVF Winter Conference on Applications of Computer Vision,  2022.


