import time
import numpy as np

import torch
import torch.nn.functional as F

from .hoc import get_consensus_patterns

def count_knn_distribution(cfg, dataset, sample, k=10, norm='l2'):
    """ Count the distribution of KNN
    Args:
        cfg: configuration
        dataset: the data for estimation
        sample: the index of samples
        k : the number of classes
    """

    time1 = time.time()
    num_classes = cfg.num_classes
    knn_labels, values = get_consensus_patterns(dataset, sample, k=k)
    # make the self-value less dominant (intuitive)
    values[:, 0] = 2.0 * values[:, 1] - values[:, 2]

    knn_labels_cnt = torch.zeros(len(sample), num_classes)

    for i in range(num_classes):
        # knn_labels_cnt[:,i] += torch.sum(1.0 * (knn_labels == i), 1) # not adjusted
        # adjusted based on the above intuition
        knn_labels_cnt[:,
                       i] += torch.sum((1.0 - values) * (knn_labels == i), 1)

    time2 = time.time()
    if cfg.details:
        print(f'Running time for k = {k} is {time2 - time1} s')

    if norm == 'l2':
        # normalized by l2-norm -- cosine distance
        knn_labels_prob = F.normalize(knn_labels_cnt, p=2.0, dim=1)
    elif norm == 'l1':
        # normalized by mean
        knn_labels_prob = knn_labels_cnt / \
            torch.sum(knn_labels_cnt, 1).reshape(-1, 1)
    else:
        raise NameError('Undefined norm')
    return knn_labels_prob


def get_score(knn_labels_cnt, label):
    """ Get the corruption score. Lower score indicates the sample is more likely to be corrupted.
    Args:
        knn_labels_cnt: KNN labels
        label: corrupted labels
    """
    score = F.nll_loss(torch.log(knn_labels_cnt + 1e-8),
                      label, reduction='none')
    return score


def simi_feat_batch(cfg, dataset):
    """ Construct the set of data that are likely to be corrupted.
    """

    # Build Feature Clusters --------------------------------------
    num_classes = cfg.num_classes

    sample_size = int(len(dataset) * 0.9)
    if cfg.hoc_cfg is not None and cfg.hoc_cfg.sample_size:
        sample_size = np.min((cfg.hoc_cfg.sample_size, int(len(dataset)*0.9)))

    idx = np.random.choice(range(len(dataset)), sample_size, replace=False)

    knn_labels_cnt = count_knn_distribution(
        cfg, dataset=dataset, sample=idx, k=cfg.detect_cfg.k, norm='l2')

    score = get_score(knn_labels_cnt, torch.tensor(dataset.label[idx]))
    score_np = score.cpu().numpy()
    sel_idx = dataset.index[idx]  # raw index

    label_pred = np.argmax(knn_labels_cnt.cpu().numpy(), axis=1).reshape(-1)
    if cfg.detect_cfg.method == 'mv':
        # test majority voting
        # print(f'Use MV')
        sel_true_false = label_pred != dataset.label[idx]
        sel_noisy = (sel_idx[sel_true_false]).tolist()
        suggest_label = label_pred[sel_true_false].tolist()
    elif cfg.detect_cfg.method == 'rank':
        # print(f'Use ranking')

        sel_noisy = []
        suggest_label = []
        for sel_class in range(num_classes):
            thre_noise_rate_per_class = 1 - \
                min(1.0 * cfg.T_given_noisy[sel_class][sel_class], 1.0)
            # clip the outliers
            if thre_noise_rate_per_class >= 1.0:
                thre_noise_rate_per_class = 0.95
            elif thre_noise_rate_per_class <= 0.0:
                thre_noise_rate_per_class = 0.05
            sel_labels = dataset.label[idx] == sel_class
            thre = np.percentile(
                score_np[sel_labels], 100 * (1 - thre_noise_rate_per_class))

            indicator_all_tail = (score_np >= thre) * (sel_labels)
            sel_noisy += sel_idx[indicator_all_tail].tolist()
            suggest_label += label_pred[indicator_all_tail].tolist()
    else:
        raise NameError('Undefined method')

    # raw index, raw index, suggested true label
    return sel_noisy, sel_idx, suggest_label
