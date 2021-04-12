# RacialPhenotypesFR
Face Verification and Identification Test Code for the study: Understanding Racial Bias using Facial Phenotypes.


## Getting started


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

3- To download pre-trained models and labels please visit [here](https://collections.durham.ac.uk/collections/r2x633f102r). After installation, please put model.ckpt to model/ folder and put FDA files to test_assets/ folder.

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
| **Lip Shape**  | Big / Small                         |
| **Hair Type**  | Straight / Wavy / Curly / Bald      |
| **Hair Color** | Red / Blonde / Brown / Black / Gray |

The distribution of race-relavent phenotype attributes of [RFW](https://github.com/seymayucer/RacialPhenotypesFREvaluation/blob/main/figures/rfw-phenotype-dist.pdf) and [VGGFace2 test](https://github.com/seymayucer/RacialPhenotypesFREvaluation/blob/main/figures/vggtest-phenotype-dist.pdf) datasets.


## References

https://github.com/wujiyang/Face_Pytorch 
