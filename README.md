## Measuring Hidden Bias within Face Recognition via Racial Phenotype

<a name="Paper" class="btn btn-outline-warning mr-2" href="https://arxiv.org/pdf/2110.09839.pdf">Paper</a> -
<a name="Poster"  class="btn btn-outline-warning mr-2" href="https://seymayucer.github.io/assets/pdfs/391_poster.pdf">Poster</a> -
<a name="Video"  class="btn btn-outline-warning mr-2" href="https://youtu.be/78OsQ_stkL4">Video</a> -
<a name="Dataset"  class="btn btn-outline-warning mr-2" href="https://collections.durham.ac.uk/files/r2hm50tr746">Dataset</a>


### Abstract

Recent work reports disparate performance for intersectional racial groups across face recognition tasks: face verification and identification. However, the definition of those racial groups has a significant impact on the underlying findings of such racial bias analysis. Previous studies define these groups based on either demographic information (e.g. African, Asian etc.) or skin tone (e.g. lighter or darker skins). The use of such sensitive or broad group definitions has disadvantages for bias investigation and subsequent counter-bias solutions design. By contrast, this study introduces an alternative racial bias analysis methodology via facial phenotype attributes for face recognition. We use the set of observable characteristics of an individual face where a race-related facial phenotype is hence specific to the human face and correlated to the racial profile of the subject. We propose categorical test cases to investigate the individual influence of those attributes on bias within face recognition tasks. We compare our phenotype-based grouping methodology with previous grouping strategies and show that phenotype-based groupings uncover hidden bias without reliance upon any potentially protected attributes or ill-defined grouping strategies. Furthermore, we contribute corresponding phenotype attribute category labels for two face recognition tasks: RFW for face verification and VGGFace2 (test set) for face identification.


### Racial Phenotypes for Face Recognition

We propose using race-related facial (phenotype) characteristics within face recognition to investigate racial bias by categorising representative racial characteristics on the face and exploring the impact of each characteristic phenotype attribute: **skin types, eyelid type, nose shape, lips shape, hair colour and hair type.** 



![alt text](https://github.com/seymayucer/FacialPhenotypes/blob/main/figures/phenotype_photos.png)


Facial phenotype attributes and their categorisation.



## Getting Started

To start working with this project you will need to take the following steps:

- Install Python packages using `conda env create --file environment.yaml`

- For face verification, please install [RFW dataset](http://www.whdeng.cn/RFW/testing.html) and for face identification [VGGFace 2.](https://drive.google.com/file/d/1jdZw6ZmB7JRK6RS6QP3YEr2sufJ5ibtO/view) 

- Download pre-trained models and annotations from [here](http://doi.org/10.15128/r2hm50tr746). After installation, please place model.ckpt under *models/* folder and place FDA files under *test_assets/* folder.

## Reproducing experiment results

To reproduce the performance reported in the paper: First, align images to 112x112.

#### RFW Face Alignment

~~~
python face_alignment.py --dataset_name RFW --data_dir datasets/test/data/African/ --output_dir datasets/test_aligned/African --landmark_file datasets/test/txts/African/African_lmk.txt 
~~~ 


#### VGGFace Face Alignment

~~~
python face_alignment.py --dataset_name VGGFace2 --data_dir datasets/VGGFace2/ --output_dir datasets/test_aligned/VGGFace2_aligned --landmark_file datasets/VGGFace2/bb_landmark/loose_bb_test.csv
~~~ 

#### Attribute-based Face Verification:

~~~
python face_atribute_verification.py --data_dir datasets/test_aligned/ --model_dir models/setup1_model/model --pair_file test_assets/AttributePairs/setup1/skintype_type1_6000.csv --batch_size 32
~~~

#### Cross Attribute-based Face Verification:

~~~
python face_cross_atribute_verification.py --input_predictions test_assets/AttributeCrossPairs/skintype_type2.csv --dist_name 'vgg_dist' --output_path test_assets/AttributeCrossPairs
~~~


<!-- python face_verification.py --data_dir path/to/rfw/aligned/ --model_dir ./model/model.ckpt --pair_file ./test_assets/pairs/AttributePairs/eye_monolid_pairs_6000_selected.csv  -->
<!-- #### Subgroup-based Face Verification:


python face_verification_mxnet.py --data_dir datasets/test_aligned/ --model_dir models/setup1_model/model --pair_file test_assets/SubgroupPairs/meta_skin-0_lips-big_eye-other_nose-narrow_hairtype-straight_00102_20k_selected.csv --batch_size 32 --> 
<!-- 
#### Face identification test:
~~~
python face_identification.py --data_dir  path/to/vgg/aligned/ --model_dir ./model/model.ckpt --img_list_file AttributeLabels/test_list.txt 
~~~ -->

The distribution of race-relavent phenotype attributes of [RFW](https://github.com/seymayucer/RacialPhenotypesFREvaluation/blob/main/figures/rfw-phenotype-dist.pdf) and [VGGFace2 test](https://github.com/seymayucer/RacialPhenotypesFREvaluation/blob/main/figures/vggtest-phenotype-dist.pdf) datasets.


## BibTeX
If you are making use of this work in any way (including our pre-trained models or datasets), you must please reference the following articles in any report, publication, presentation, software release or any other associated materials:

```
@InProceedings{yucermeasuring,
  author = {Yucer, S. and Tektas, F. and Al Moubayed, N. and Breckon, T.P.},
  title = {Measuring Hidden Bias within Face Recognition via Racial Phenotypes},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2022},
  publisher = {IEEE},
  arxiv = {http://arxiv.org/abs/2110.09839},
}
```



