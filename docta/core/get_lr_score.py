"""
 This file will generate a long-tail score for each submitted sample 
"""
# Import required packages
import math
import torch
import numpy as np
from collections import Counter
from .hoc import get_consensus_patterns
from docta.datasets.customize import CustomizedDataset


def lt_score(data, feature_type, k=10):
    """
    Input: data
           the extracted embedding / features from the given dataset; 
           {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
                k --> the number of nearest neighbors considered;
           {array-like, sparse matrix} of shape (n_samples, n_labels)
    Output: the long-tail score of each sample.
    """
    return score_from_embedding(data, k)


def score_from_embedding(data, k):
    """
    Input: embed --> the extracted embedding / features from the given dataset; 
           {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
           k --> the number of nearest neighbors considered;
    Long-tail score calculation:
           step 1: use the scikit-learn to find k-nearest neighbors of each embedding;
           step 2: for each sample, calculate the averaged embedding distance x w.r.t. its closed k neighbors (non-negative);
           step 3: map the distance to 0-1 range to form a long-tail score, i.e., f(x) = 2 / (1 + exp(-x)) - 1;
    Output: the long-tail score of each sample.
    """
    # To clarify, the label here is just a placeholder and is not used
    label = [0 for i in range(len(data))] 
    print("Customizing the extracted embeddings as a dataset...")
    dataset = CustomizedDataset(feature=data, label=label)
    sample = np.arange(len(dataset))
    print("Getting consensus patterns...")
    _, values = get_consensus_patterns(dataset, sample, k=k)
    np_values = values.numpy()
    mean_dist = np.mean(np_values, 1)
    lt_scores = []
    for i in range(mean_dist.shape[0]):
      tmp = np.round((2.0 / (1 + math.exp(-mean_dist[i]))) - 1.0, 4)
      lt_scores.append(tmp)
    return lt_scores

