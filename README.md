# RacialPhenotypesFR
Face Verification and Identification Test Code for Racial Phenotype Evaluation on RFW and VGGFace2


## Getting started


To start working, you will need to install the following packages:

- PyTorch
- thundersvm
- logging
- cv2
- numpy
- pandas
- sklearn
- tqdm
- pathlib

or you can install all using pip or conda: ` pip install -r setup/requirements.txt `

## Data Preparation
To test face verification and identification you need to align images to 112. To test atribute classifier you need to aligne images to 224.

~~~
python preprocess/face_align.py --data_dir /path/to/dataset --output_dir /path/to/destination/vggface2_train_224_aligned/ --landmark_file '/path/to/VGGFace2/bb_landmark/loose_landmark_train.csv'
~~~ 
## Reproducing experiment results

Take the following steps to reproduce the performance reported in the paper:

1- For both task, images should be aligned to 112x112.

2- For face verification please install RFW from offical website.
~~~
python face_verification.py --data_dir path/to/rfw/aligned/ --model_dir ./model/Iter_021000_net.ckpt --pair_file ./test_assets/pairs/AttributePairs/eye_narrow_pairs_6000_selected.csv 
~~~


~~~
python face_identification.py --data_dir  path/to/vgg/aligned/ --model_dir model/Iter_021000_net.ckpt --img_list_file AttributeLabels/test_list.txt 
~~~

## Attribute Labels

Attribute categorization is presented in the below.

| **Attribute**  | **Categories**               |
|---------------------|-------------------------------------|
| **Skin Type**  | Type 1 / 2 / 3 / 4 / 5 / 6          |
| **Eye Shape**  | Monolid / Other                     |
| **Nose Shape** | Wide / Narrow                       |
| **Lip Shape**  | Big / Small                         |
| **Hair Type**  | Straight / Wavy / Curly / Bald      |
| **Hair Color** | Red / Blonde / Brown / Black / Gray |

The distribution of race-relavent phenotype attributes of [RFW](https://github.com/seymayucer/RacialPhenotypesFREvaluation/blob/main/figures/rfw-phenotype-dist.pdf) and [VGGFace2 test](https://github.com/seymayucer/RacialPhenotypesFREvaluation/blob/main/figures/vggtest-phenotype-dist.pdf) datasets.


## Results



## References

