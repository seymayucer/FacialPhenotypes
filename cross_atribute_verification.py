"""
    Demo code for evaluation of bias in face verification
    Author: Tomas Sixta (tomas.sixta@gmail.com)
"""
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
import numpy as np
import pandas as pd
import errno
import argparse


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    actual_issame = np.greater(actual_issame, 0.5)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))
    )
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    fnr = 0 if (tp + fn == 0) else float(fn) / float(tp + fn)
    tnr = 0 if (fp + tn == 0) else float(tn) / float(fp + tn)

    acc = float(tp + tn) / dist.size
    return tpr, fpr, fnr, tnr, acc


def split_positive_negative(data, dist_name):
    """Splits matches and scores lists into positive (genuine) and negative (impostor).
    Args:
        anno_by_templateid: Annotation dictionary
        cids: Dictionary with column ids
        matches: List with matches
        scores: List with scores
        errata_positive: List of template_ids, positive pairs containing these will be removed
    Returns:
        positive_matches: Genuine matches
        positive_scores: Scores corresponding to genuine matches
        negative_matches: Impostor matches
        negative_scores: Scores corresponding to impostor matches
    """
    pos = data[data["Class_ID_s1"] == data["Class_ID_s2"]]
    positive_scores = list(pos[dist_name].values)
    positive_matches = pos[
        ["Class_ID_s1", "img_id_s1", "Class_ID_s2", "img_id_s2"]
    ].values

    neg = data[data["Class_ID_s1"] != data["Class_ID_s2"]]
    negative_scores = list(neg[dist_name].values)
    negative_matches = neg[
        ["Class_ID_s1", "img_id_s1", "Class_ID_s2", "img_id_s2"]
    ].values
    return (positive_matches, positive_scores, negative_matches, negative_scores)


def calculate_aucs(roc_dict, positive_scores, negative_scores):
    """Calculate AUC for every element (list of scores) of roc_dict dictionary.
    The results is, that for every combination of values of legitimate and positive attributes we will have AUC
    """
    ret = {}
    all_fpr, all_tpr, all_thresholds = roc_curve(
        [1] * len(positive_scores) + [0] * len(negative_scores),
        positive_scores + negative_scores,
    )  # , drop_intermediate=False)
    all_fpr, all_tpr, all_thresholds = (
        np.flip(all_fpr),
        np.flip(all_tpr),
        np.flip(all_thresholds),
    )

    for xL_key in tqdm(roc_dict):
        ret[xL_key] = {}
        for aF_key in roc_dict[xL_key]:
            sorted_data = sorted(roc_dict[xL_key][aF_key])
            threshold_indices = np.searchsorted(all_thresholds, sorted_data)
            previous_ti = 0
            tpr = []
            for i, ti in enumerate(threshold_indices):
                tpr += [(len(sorted_data) - i) / len(sorted_data)] * (ti - previous_ti)
                previous_ti = ti
            tpr += [0.0] * (len(all_fpr) - len(tpr))
            tpr = np.array(tpr)
            ret[xL_key][aF_key] = auc(all_fpr, tpr)

    return ret


def evaluation_metrics(data, output_path, dist_name):
    """
    Calculate evaluation metrics used in the competition. By default it calculates overall accuracy in terms of AUC-ROC,
    bias for gender & skin colour both for positive and negative samples as well as biases for gender and skin colour separately.
    Returns:
        Human readable dictionary with calculated metrics.
    """

    (
        positive_matches,
        positive_scores,
        negative_matches,
        negative_scores,
    ) = split_positive_negative(data, dist_name)

    results = {
        "bias_in_positive_samples": {},
        "bias_in_negative_samples": {},
        "auc": None,
    }  # Human readable dictionary with measures of bias and accuracy
    ### Accuracy measured by AUC threshold selection i think
    results["auc"] = roc_auc_score(
        [1] * len(positive_scores) + [0] * len(negative_scores),
        positive_scores + negative_scores,
    )

    all_fpr, all_tpr, all_thresholds = roc_curve(
        [1] * len(positive_scores) + [0] * len(negative_scores),
        positive_scores + negative_scores,
    )  # , drop_intermediate=False)
    all_fpr, all_tpr, all_thresholds = (
        np.flip(all_fpr),
        np.flip(all_tpr),
        np.flip(all_thresholds),
    )

    optimal_idx = np.argmax(all_tpr - all_fpr)
    optimal_threshold = all_thresholds[optimal_idx]
    print(
        "fpr :",
        all_fpr[optimal_idx],
        "tpr:",
        all_tpr[optimal_idx],
        "best threshold",
        optimal_threshold,
    )
    from pathlib import Path

    filep = open(Path(output_path) / "vgg_covariance_matrix_results.log", "a+")
    data["unique_cat"] = data["category1"] + "_" + data["category2"]
    for i, category_pairs in data.groupby(["unique_cat"]):

        # For negative samples, the ROC curve shows TNR @ FNR. To calculate this with code designed for TPR @ FPR it's necessary to
        # invert the signs in positive_scores and negative_scores and also treat positive_scores as if they were the negative class
        # and negative_scores as if they were the positive class

        tpr, fpr, fnr, tnr, acc = calculate_accuracy(
            optimal_threshold,
            category_pairs[dist_name].values,
            category_pairs.issame.values,
        )

        print("tpr, fpr, fnr,acc\n", tpr, fpr, fnr, acc, category_pairs.issame.sum())
        filep.write(
            f"{category_pairs.category1.unique()[0]} {category_pairs.category2.unique()[0]} {acc} {tpr} {fpr} {fnr} {tnr}\n"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce number of matches")
    parser.add_argument(
        "--input_predictions", type=str,
    )
    parser.add_argument("--output_path", type=str, default="./output/", help="")
    parser.add_argument("--dist_name", type=str, default="None", help="")
    args = parser.parse_args()
    args.input_predictions = os.path.expanduser(args.input_predictions)
    args.output_path = os.path.expanduser(args.output_path)

    try:
        os.makedirs(args.output_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    print("Loading submitted scores...")
    data = pd.read_csv(args.input_predictions)  # we have that
    data["issame"] = (data["Class_ID_s1"] == data["Class_ID_s2"]).values
    # import pdb

    # pdb.set_trace()
    print("Calculating bias and accuracy...")
    ###RUN EVALUATION

    evaluation = evaluation_metrics(data, args.output_path, args.dist_name)

    print(evaluation)

    print("Test submission was successful")

