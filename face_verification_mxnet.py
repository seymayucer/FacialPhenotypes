"""
December 22 2020
syucer

"""


import os
import argparse
import sys
import numpy as np
from scipy import misc
from sklearn.model_selection import StratifiedKFold
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
import pickle
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd
import pandas as pd
from numpy import linalg as line
from scipy import spatial
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)


class FaceVerification:
    def __init__(self, model=None, batch_size=32, data_dir=None):
        super().__init__()
        logging.info("Face Verification for RFW.")
        self.data_dir = data_dir
        self.image_size = 112
        self.batch_size = batch_size
        self.model = model

    def load_model(self, model_dir=None):
        logging.info("Model Loading")
        ctx = mx.gpu(0)
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_dir, 1)
        all_layers = sym.get_internals()
        sym = all_layers["fc1_output"]
        self.model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.model.bind(
            data_shapes=[
                ("data", (self.batch_size, 3, self.image_size, self.image_size))
            ]
        )
        self.model.set_params(arg_params, aux_params)
        return self.model

    def load_images(self, inp_csv_file):
        logging.info("Image Data Loading")
        issame_list, data_list = [], []
        pairs = pd.read_csv(inp_csv_file)
        # data_list = list(
        #     np.empty((2, pairs.shape[0] * 2, 3, self.image_size, self.image_size))
        # )
        for flip in [0, 1]:
            data = nd.empty((pairs.shape[0] * 2, 3, self.image_size, self.image_size))
            data_list.append(data)

        j = 0
        for i, row in pairs.iterrows():

            if i % 1000 == 0:
                logging.info("processing {}".format(i))

            issame_list.append(row.issame)
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
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            im1 = np.transpose(im1, (2, 0, 1))  # 3*112*112, RGB
            im1 = mx.nd.array(im1)

            im2 = cv2.imread(path2)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            im2 = np.transpose(im2, (2, 0, 1))  # 3*112*112, RGB
            im2 = mx.nd.array(im2)

            for flip in [0, 1]:

                if flip == 1:
                    im1 = mx.ndarray.flip(im1, 2)
                data_list[flip][j][:] = im1

            for flip in [0, 1]:
                if flip == 1:
                    im2 = mx.ndarray.flip(im2, 2)
                data_list[flip][j + 1][:] = im2
                # data_list[flip][i][:] = img

            j = j + 2
        # bins shape should be 2,12000,3,112,112

        # data = np.asarray(data_list)
        self.issame = np.asarray(issame_list)

        self.data = data_list
        logging.info("Pairs are loaded, shape: 2x{}.".format(self.data[0].shape))
        return self.data, self.issame, pairs.shape

    def clean_data(self):
        self.data = None
        self.issame = None

    def verify(self, model=None):
        data_list = self.data

        embeddings_list = []
        time_consumed = 0
        _label = nd.ones((self.batch_size,))
        for i in range(len(data_list)):
            data = data_list[i]
            embeddings = None
            ba = 0
            while ba < data.shape[0]:
                bb = min(ba + self.batch_size, data.shape[0])
                count = bb - ba
                _data = nd.slice_axis(data, axis=0, begin=bb - self.batch_size, end=bb)
                time0 = datetime.datetime.now()

                db = mx.io.DataBatch(data=(_data,), label=(_label,))

                self.model.forward(db, is_train=False)
                net_out = self.model.get_outputs()
                _embeddings = net_out[0].asnumpy()
                time_now = datetime.datetime.now()
                diff = time_now - time0
                time_consumed += diff.total_seconds()
                if embeddings is None:
                    embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
                embeddings[ba:bb, :] = _embeddings[(self.batch_size - count) :, :]
                ba = bb
            embeddings_list.append(embeddings)

        _xnorm = 0.0
        _xnorm_cnt = 0
        for embed in embeddings_list:
            for i in range(embed.shape[0]):
                _em = embed[i]
                _norm = np.linalg.norm(_em)
                _xnorm += _norm
                _xnorm_cnt += 1
        _xnorm /= _xnorm_cnt

        acc1 = 0.0
        std1 = 0.0
        embeddings = embeddings_list[0] + embeddings_list[1]
        embeddings = sklearn.preprocessing.normalize(embeddings)
        print(embeddings.shape)
        print("infer time", time_consumed)

        tpr, fpr, accuracy, best_thresholds = self.evaluate(
            embeddings, self.issame, nrof_folds=10
        )

        acc2, std2 = np.mean(accuracy), np.std(accuracy)
        logging.info("Accuracy {}".format(acc2))
        return tpr, fpr, acc2, std2

    def evaluate(self, embeddings, actual_issame, nrof_folds=10):
        # Calculate evaluation metrics
        thresholds = np.arange(-1, 1, 0.001)
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

        assert embeddings1.shape[1] == embeddings2.shape[1]
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        # k_fold = LFold(n_splits=nrof_folds, shuffle=False)
        k_fold = StratifiedKFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))

        tnrs = np.zeros((nrof_folds, nrof_thresholds))
        fnrs = np.zeros((nrof_folds, nrof_thresholds))

        f1s = np.zeros((nrof_folds))

        accuracy = np.zeros((nrof_folds))
        indices = np.arange(nrof_pairs)

        veclist = np.concatenate((embeddings1, embeddings2), axis=0)
        meana = np.mean(veclist, axis=0)
        embeddings1 -= meana
        embeddings2 -= meana
        dist = np.sum(embeddings1 * embeddings2, axis=1)
        dist = dist / line.norm(embeddings1, axis=1) / line.norm(embeddings2, axis=1)

        for fold_idx, (train_set, test_set) in enumerate(
            k_fold.split(indices, actual_issame)
        ):
            # print(train_set.shape, actual_issame[train_set].sum())
            # print(test_set.shape, actual_issame[test_set].sum())

            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                _, _, _, _, acc_train[threshold_idx], f1 = self.calculate_accuracy(
                    threshold, dist[train_set], actual_issame[train_set]
                )
            best_threshold_index = np.argmax(acc_train)
            # print('threshold', thresholds[best_threshold_index])
            for threshold_idx, threshold in enumerate(thresholds):
                (
                    tprs[fold_idx, threshold_idx],
                    fprs[fold_idx, threshold_idx],
                    tnrs[fold_idx, threshold_idx],
                    fnrs[fold_idx, threshold_idx],
                    _,
                    _,
                ) = self.calculate_accuracy(
                    threshold, dist[test_set], actual_issame[test_set]
                )

            _, _, _, _, accuracy[fold_idx], f1s[fold_idx] = self.calculate_accuracy(
                thresholds[best_threshold_index],
                dist[test_set],
                actual_issame[test_set],
            )
        tpr = np.mean(tprs, 0)[best_threshold_index]
        fpr = np.mean(fprs, 0)[best_threshold_index]

        # tnr = np.mean(tnrs, 0)[best_threshold_index]
        # fnr = np.mean(fnrs, 0)[best_threshold_index]

        return tpr, fpr, accuracy, thresholds[best_threshold_index]

    def calculate_accuracy(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        actual_issame = np.less(actual_issame, 0.5)

        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
            actual_issame, predict_issame
        ).ravel()

        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)

        tnr = 0 if (fp + tn == 0) else float(tn) / float(fp + tn)
        fnr = 0 if (fn + tp == 0) else float(fn) / float(fn + tp)

        acc = float(tp + tn) / dist.size
        f1 = sklearn.metrics.f1_score(predict_issame, actual_issame)

        return tpr, fpr, tnr, fnr, acc, f1


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
    validation.load_model(model_dir=args.model_dir)
    _, _, _shape = validation.load_images(args.pair_file)
    tpr, fpr, acc, std = validation.verify()
    logging.info(
        "Testing Accuracy {} for {} in shape {}".format(acc, args.pair_file, _shape[0])
    )

