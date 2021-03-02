'''
Jan 11 2020
syucer
'''

import cv2
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import argparse
import cbam
from pathlib import Path
from torch.nn import DataParallel
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from thundersvm import SVC
from itertools import chain
import logging
logging.basicConfig(level=logging.INFO)


class FaceIdentification:
    def __init__(self,
                 model=None,
                 batch_size=32,
                 dataset_name='vggface2_test',
                 data_dir=None):
        super().__init__()
        logging.info('Face Identification for VGGFace2.')

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.image_size = 112
        self.batch_size = batch_size
        self.model = model

    def load_model(self, feature_dim, model_dir=None):
        logging.info('Model Loading')
        net = cbam.CBAMResNet(100, feature_dim=feature_dim, mode='ir')
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(torch.load(model_dir)['net_state_dict'])
        net = DataParallel(net).to(self.device)
        self.model = net.eval()
        return self.model

    def get_batch_img(self, batch_list):
        batch = []
        for img_dir in batch_list:
            img_path = Path(self.data_dir) / img_dir
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img - 127.5
            img = img * 0.0078125
            img = np.transpose(img, axes=(2, 0, 1))
            batch.append(img)

        return batch

    def get_features(self, image_list, model=None):
        self.image_list = image_list.files

        if Path('features/{}_features.npy'.format(self.dataset_name)).exists():
            logging.info('Feature files are found.')
            self.load_features()
        else:
            logging.info('Feature Extraction, It may take a while...')
            if model is not None:
                self.model = model

            embeddings = np.zeros((len(self.image_list), 512))

            for idx in tqdm.tqdm(
                    range(0, len(self.image_list), self.batch_size)):

                batch_list = self.image_list[idx:idx + self.batch_size]
                batch_data = self.get_batch_img(batch_list)
                batch_data = torch.FloatTensor(batch_data).to(self.device)

                embed = self.model(batch_data)
                embeddings[idx:idx +
                           self.batch_size] = embed.detach().cpu().numpy()

                embed = None
            self.features = embeddings
            self.save_feature()

        labels = image_list.Class_ID
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(labels)

    def train_svm(self, train_ixs):
        logging.info('SVM Training...')
        X_train = self.features[train_ixs]
        y_train = self.labels[train_ixs]
        self.svm_model = SVC(kernel="linear", C=1).fit(X_train, y_train)

    def test_svm(self, test_ixs):
        TEST_BATCH_SIZE = 1000
        preds = []
        X_test = self.features[test_ixs]
        y_test = self.labels[test_ixs]
        with tqdm(total=len(test_ixs), file=sys.stdout) as pbar:
            for i in range(0, len(test_ixs), TEST_BATCH_SIZE):
                X_test_batch = X_test[i:i + TEST_BATCH_SIZE]
                pred = self.svm_model.predict(X_test_batch)
                preds.append(pred)
                # update tqdm
                pbar.set_description("Processed: %d" % (1 + i))
                pbar.update(TEST_BATCH_SIZE)
        y_pred = np.array(list(chain(*preds)))
        logging.info("Overall Accuracy: {}".format(
            accuracy_score(y_test, y_pred)))
        return y_test, y_pred

    def load_features(self):
        self.features = np.load('features/{}_features.npy'.format(
            self.dataset_name),
                                allow_pickle=True)
        self.image_list = np.load('features/{}_paths.npy'.format(
            self.dataset_name),
                                  allow_pickle=True)

    def save_feature(self):
        np.save('features/{}_features.npy'.format(self.dataset_name),
                self.features)
        np.save('features/{}_paths.npy'.format(self.dataset_name),
                self.image_list)


def generate_fold(data, NUM_FOLDS=3, TEST_SAMPLE_SIZE=50):
    all_folds = []
    for fold in range(0, NUM_FOLDS):
        class_folds = {"train": [], "test": []}
        for i, group in data.groupby("Class_ID"):
            num_samples = group.shape[0]
            test_mask = np.zeros(num_samples, dtype=np.bool)
            if TEST_SAMPLE_SIZE * NUM_FOLDS > num_samples:
                start = fold * TEST_SAMPLE_SIZE
                end = start + TEST_SAMPLE_SIZE
                ix = [i % num_samples for i in range(start, end)]
            else:
                class_fold_size = num_samples // NUM_FOLDS
                start = fold * class_fold_size
                end = start + class_fold_size
                ix = range(start, end)

            test_mask[ix] = True
            try:
                class_folds["test"].append(group[test_mask].sample(
                    n=TEST_SAMPLE_SIZE, random_state=0))

            except:
                logging.warning('fold error')
            class_folds["train"].append(group[~test_mask])

        class_folds["test"] = pd.concat(class_folds["test"])
        class_folds["train"] = pd.concat(class_folds["train"])
        all_folds.append(class_folds)
    return all_folds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Face Identification for VGGface2')
    parser.add_argument('--data_dir',
                        type=str,
                        default='vggface_test/test/aligned_data',
                        help='dataset root')
    parser.add_argument(
        '--img_list_file',
        type=str,
        default='./AttributePairs/eye_narrow_pairs_6000_selected.csv',
        help='pair file to test')
    parser.add_argument('--model_dir',
                        type=str,
                        default='/model/',
                        help='lfw image root')
    args = parser.parse_args()

    validation = FaceIdentification(batch_size=16,
                                    model=None,
                                    data_dir=args.data_dir)

    validation.load_model(feature_dim=512,
                          model_dir='model/Iter_021000_net.ckpt')

    image_list = pd.read_csv(args.img_list_file, names=['files'])
    image_list['Class_ID'] = image_list.files.apply(lambda x: x.split('/')[-2])
    all_folds = generate_fold(image_list)
    validation.get_features(image_list)
    for afold in all_folds[0:1]:
        test_ix = afold['test'].index.values
        train_ix = afold['train'].index.values

        # import pdb
        # pdb.set_trace()

        validation.train_svm(train_ix)
        validation.test_svm(test_ix)
