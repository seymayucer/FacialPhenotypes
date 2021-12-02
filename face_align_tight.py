import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from skimage import transform as trans
import argparse

# those points cause better results but for continuity and comparibility with arcface, arcface points have used.
# src1 = np.array(
#     [
#         [51.642, 50.115],
#         [57.617, 49.990],
#         [35.740, 69.007],
#         [51.157, 89.050],
#         [57.025, 89.702],
#     ],
#     dtype=np.float32,
# )
# # <--left
# src2 = np.array(
#     [
#         [45.031, 50.118],
#         [65.568, 50.872],
#         [39.677, 68.111],
#         [45.177, 86.190],
#         [64.246, 86.758],
#     ],
#     dtype=np.float32,
# )

# # ---frontal
# src3 = np.array(
#     [
#         [39.730, 51.138],
#         [72.270, 51.138],
#         [56.000, 68.493],
#         [42.463, 87.010],
#         [69.537, 87.010],
#     ],
#     dtype=np.float32,
# )

# # -->right
# src4 = np.array(
#     [
#         [46.845, 50.872],
#         [67.382, 50.118],
#         [72.737, 68.111],
#         [48.167, 86.758],
#         [67.236, 86.190],
#     ],
#     dtype=np.float32,
# )

# # -->right profile
# src5 = np.array(
#     [
#         [54.796, 49.990],
#         [60.771, 50.115],
#         [76.673, 69.007],
#         [55.388, 89.702],
#         [61.257, 89.050],
#     ],
#     dtype=np.float32,
# )

# src = np.array([src1, src2, src3, src4, src5])


arcface_src = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

arcface_src = np.expand_dims(arcface_src, axis=0)
src_map = {112: arcface_src, 224: arcface_src * 2}
# lmk is prediction; src is template


def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float("inf")
    src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]

        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

    return min_M, min_index


def norm_crop(img, landmark, image_size):
    M, pose_index = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def face_align(input_dir, output_dir, metadata, aligned_size):

    for index, row in metadata.iterrows():

        img_dir = Path(input_dir) / Path("/".join(row[0].split("/")[-2:])).with_suffix(
            ".jpg"
        )

        img = cv2.imread(str(img_dir))

        landmark = row.values[1:]
        landmark = np.array(landmark.reshape(5, 2), dtype="float")
        # bbox = row.values[11:15].reshape(1, 4)
        aligned_img = norm_crop(img, landmark, aligned_size)

        dst_dir = Path(output_dir) / Path("/".join(row[0].split("/")[-2:])).with_suffix(
            ".jpg"
        )

        # row.NAME_ID).with_suffix(".jpg")
        dst_dir.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(dst_dir), aligned_img)

        if index % 1000 == 0:
            print("aligned", index)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image alignment to 112 or 224 only")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="vggface_test/test/aligned_data",
        help="dataset root",
    )
    parser.add_argument(
        "--dataset_name", type=str, help="RFW,VGGFace2,LFW",
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets/test_aligned", help="dataset root",
    )
    parser.add_argument(
        "--landmark_file",
        type=str,
        default="datasets/VGGFace2/bb_landmark/loose_landmark_test.csv",
        help="pair file to test",
    )

    parser.add_argument("--image_size", type=int, default=112, help="112 or 224")

    args = parser.parse_args()

    if args.dataset_name == "RFW":
        landmark = pd.read_csv(args.landmark_file, header=None, delimiter="\t")
        landmark = landmark.drop([1], axis=1)

    elif args.dataset_name == "VGGFace2":
        landmark = pd.read_csv(args.landmark_file, header=[1, 2])

    elif args.dataset_name == "LFW":
        landmark = pd.read_csv(args.landmark_file, header=None, delimiter="\t")
        landmark = landmark.drop([1], axis=1)

    print(landmark.head())

    face_align(args.data_dir, args.output_dir, landmark, aligned_size=args.image_size)

