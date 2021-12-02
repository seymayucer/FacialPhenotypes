"""
December 22 2020
syucer

"""

import cv2
import numpy as np
import pandas as pd

import tqdm
import argparse
from pathlib import Path

import model
from torch.nn import DataParallel
import torch.utils.data
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import logging

logging.basicConfig(level=logging.INFO)


class FaceVerification:
    def __init__(self, model=None, batch_size=32, data_dir=None):
        super().__init__()
        logging.info("Face Verification for RFW.")
        self.data_dir = data_dir
        self.image_size = 112
        self.batch_size = batch_size
        self.model = model

    def load_model(self, feature_dim, model_dir=None):
        logging.info("Model Loading")
        net = model.CBAMResNet(100, feature_dim=feature_dim, mode="ir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.load_state_dict(torch.load(model_dir)["net_state_dict"])
        net = DataParallel(net).to(self.device)
        self.model = net.eval()
        return self.model

    def load_images(self, inp_csv_file):
        logging.info("Image Data Loading")
        issame_list = []
        pairs = pd.read_csv(inp_csv_file)
        data_list = np.empty(
            (2, pairs.shape[0] * 2, 3, self.image_size, self.image_size)
        )

        j = 0
        for i, row in pairs.iterrows():

            if i % 1000 == 0:
                logging.info("processing {}".format(i))
            issame = row.issame
            path1 = "{}/{}/{}_{:04d}.jpg".format(
                self.data_dir,
                row.Class_ID_s1,
                row.Class_ID_s1.split("/")[1],
                int(row.img_id_s1),
            )
            path2 = "{}/{}/{}_{:04d}.jpg".format(
                self.data_dir,
                row.Class_ID_s2,
                row.Class_ID_s2.split("/")[1],
                int(row.img_id_s2),
            )
            im1 = cv2.imread(path1)
            im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
            im1 = im1 - 127.5
            im1 = im1 * 0.0078125
            im1 = np.transpose(im1, axes=(2, 0, 1))

            im2 = cv2.imread(path2)
            im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
            im2 = im2 - 127.5
            im2 = im2 * 0.0078125
            im2 = np.transpose(im2, axes=(2, 0, 1))

            for flip in [0, 1]:

                if flip == 1:
                    im1 = np.flip(im1, 2)
                data_list[flip][j, ...] = im1

            for flip in [0, 1]:
                if flip == 1:
                    im2 = np.flip(im2, 2)
                data_list[flip][j + 1, ...] = im2

            issame_list.append(issame)
            j = j + 2
        # bins shape should be 12000,3,112,112

        self.data = np.asarray(data_list)
        self.issame = np.asarray(issame_list)
        logging.info("Pairs are loaded, shape: {}.".format(self.data.shape))
        return self.data, self.issame, pairs.shape

    def clean_data(self):
        self.data = None
        self.issame = None

    def verify(self, model=None):
        if model is not None:
            self.model = model
        embeddings_list = []

        for i in range(len(self.data)):
            data_side = self.data[i]
            embeddings = np.zeros((data_side.shape[0], 512))
            print(data_side.shape, len(self.data), embeddings.shape)
            for idx in tqdm.tqdm(range(0, len(data_side), self.batch_size)):

                batch_data = data_side[idx : idx + self.batch_size]
                batch_data = torch.FloatTensor(batch_data).to(self.device)

                embed = self.model(batch_data)
                embeddings[idx : idx + self.batch_size] = embed.detach().cpu().numpy()
                embed = None

            embeddings_list.append(embeddings)

        embeddings = embeddings_list[0] + embeddings_list[1]
        embeddings = preprocessing.normalize(embeddings)

        tpr, fpr, accuracy, best_thresholds = self.evaluate(
            embeddings, self.issame, nrof_folds=10
        )
        acc2, std2 = np.mean(accuracy), np.std(accuracy)
        logging.info("Accuracy {}".format(acc2))
        return tpr, fpr, acc2, std2

    def evaluate(self, embeddings, actual_issame, nrof_folds=10):
        # Calculate evaluation metrics
        thresholds = np.arange(0, 4, 0.01)
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        tpr, fpr, accuracy, best_thresholds = self.calculate_roc(
            thresholds,
            embeddings1,
            embeddings2,
            np.asarray(actual_issame),
            nrof_folds=nrof_folds,
        )

        return tpr, fpr, accuracy, best_thresholds

    def calculate_roc(
        self, thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10
    ):
        assert embeddings1.shape[0] == embeddings2.shape[0]
        assert embeddings1.shape[1] == embeddings2.shape[1]
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = StratifiedKFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))
        accuracy = np.zeros((nrof_folds))
        best_thresholds = np.zeros((nrof_folds))
        indices = np.arange(nrof_pairs)

        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

        for fold_idx, (train_set, test_set) in enumerate(
            k_fold.split(indices, actual_issame)
        ):
            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                _, _, acc_train[threshold_idx] = self.calculate_accuracy(
                    threshold, dist[train_set], actual_issame[train_set]
                )
            best_threshold_index = np.argmax(acc_train)

            best_thresholds[fold_idx] = thresholds[best_threshold_index]
            for threshold_idx, threshold in enumerate(thresholds):
                (
                    tprs[fold_idx, threshold_idx],
                    fprs[fold_idx, threshold_idx],
                    _,
                ) = self.calculate_accuracy(
                    threshold, dist[test_set], actual_issame[test_set]
                )
            _, _, accuracy[fold_idx] = self.calculate_accuracy(
                thresholds[best_threshold_index],
                dist[test_set],
                actual_issame[test_set],
            )

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        return tpr, fpr, accuracy, best_thresholds

    def calculate_accuracy(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(
            np.logical_and(
                np.logical_not(predict_issame), np.logical_not(actual_issame)
            )
        )
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / dist.size
        return tpr, fpr, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Verification for RFW")
    parser.add_argument(
        "--data_dir", type=str, default="RFW/test/aligned_data", help="dataset root"
    )
    parser.add_argument(
        "--pair_file",
        type=str,
        default="./AttributePairs/eye_narrow_pairs_6000_selected.csv",
        help="pair file to test",
    )
    parser.add_argument(
        "--model_dir", type=str, default="/model/", help="pre-trained model directory"
    )

    parser.add_argument("--batch_size", type=int, default="32", help="batch_size")
    args = parser.parse_args()

    validation = FaceVerification(
        batch_size=args.batch_size, model=None, data_dir=args.data_dir
    )

    validation.load_model(feature_dim=512, model_dir=args.model_dir)
    _, _, _shape = validation.load_images(args.pair_file)
    tpr, fpr, acc, std = validation.verify()
    logging.info(
        "Testing Accuracy {} for {} in shape {}".format(acc, args.pair_file, _shape[0])
    )

