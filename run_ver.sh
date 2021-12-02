# bin/bash

# yourfilenames=`ls /mnt/SSD/FacialPhenotypes/validation/selected_pairs/*.csv`
yourfilenames='ls test_assets/AttributeCrossPairs/*.csv'
# #yourfilenames=`ls /mnt/SSD/FacialPhenotypes/validation/unique_pairs/bupt/easiest-type3.csv`

for eachfile in $yourfilenames
do
  echo "$eachfile"
  python  cross_atribute_verification.py --input_predictions "$eachfile" --dist_name 'vgg_dist' --output_path "test_assets/AttributeCrossPairs"
done


# yourfilenames=`ls /mnt/SSD/FacialPhenotypes/validation/selected_pairs/*.csv`

# for eachfile in $yourfilenames
# do
#   echo "$eachfile"
#   python  metrics_adapted_rfw.py --input_predictions "$eachfile" --output_path "/mnt/SSD/FacialPhenotypes/validation/selected_pairs"
# done


# yourfilenames=`ls pair_selection/RFW_Attribute_Based/bupt_cos_dist_6k/*100K_6000.csv`

# for eachfile in $yourfilenames
# do
#   echo "$eachfile"
#   python verification_single.py --pair_file "$eachfile" --output_path "pair_selection/RFW_Attribute_Based/bupt_cos_dist_6k/" --model "../validation/mxnet_models/Balanced_Softmax/model-balanced-soft,0"

# done


# yourfilenames=`ls pair_selection/RFW_Attribute_Based/vgg_cos_dist_6k/*100K_6000.csv`

# for eachfile in $yourfilenames
# do
#   echo "$eachfile"
#   python verification_single.py --pair_file "$eachfile" --output_path "pair_selection/RFW_Attribute_Based/vgg_cos_dist_6k/" --model "../validation/mxnet_models/r100-arcface-emore/model,1"

# done