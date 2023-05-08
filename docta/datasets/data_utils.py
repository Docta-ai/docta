import os
import os.path
import copy
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal
import torch
import torch.nn.functional as F 
import pandas as pd
import collections
from typing import Callable


def load_label(label_path, clean_label = None, key = None, clean_key = None):
    noise_label = torch.load(label_path)
    if key is None: # default key is 'noisy_label'
        key = 'noisy_label'
    if clean_key is None: # default clean key is 'clean_label'
        clean_key = 'clean_label'

    if isinstance(noise_label, dict):
        if clean_key in noise_label.keys() and clean_label is not None: # sanity check
            clean_label = noise_label['clean_label']
            assert torch.sum(torch.tensor(clean_label) - clean_label) == 0  
        return noise_label[key].reshape(-1)
    else:
        return noise_label.reshape(-1)
    


def noisify_general(clean_label, noise_rate, random_state = 0):
    """ 
        Synthesize class-dependent label noise according to a random T
    """ 
    clean_label = np.asarray(clean_label)
    num_class = len(np.unique(clean_label))
    acc = 1 - noise_rate
    std_acc = 0.05 if num_class > 2 else 0.01
    T_diag = acc + std_acc * 2 * (np.random.rand(num_class) - 0.5)
    T_diag[T_diag > 1.0] = 1.0

    T = generate_T_from_diagonal(T_diag)
    print(f'Add synthetic random noise according to T = \n{np.round(T * 100,1)}')
    noisy_label = multiclass_noisify(clean_label, T = np.array(T), random_state = random_state)
    actual_noise = (noisy_label != clean_label).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise, flush=True)
    
    return noisy_label, actual_noise



def generate_T_from_diagonal(diag): 
    """ 
        Randomly generate the label noise transition matrix (T) with given diagonal elements (diag).
        See ref: https://proceedings.mlr.press/v162/zhu22k/zhu22k.pdf
    """ 
    # 
    K = diag.shape[0]
    T = np.zeros((K, K))
    for i in range(diag.shape[0]):
        T[i, i] = diag[i]
        tmp = np.random.dirichlet(np.ones(K-1)) * (1 - diag[i])
        while np.sum(tmp > 0.9*T[i, i]) > 0:
            tmp = np.random.dirichlet(np.ones(K-1)) * (1 - diag[i])
        T[i, np.arange(K)!=i] = tmp  # use this one
    return T


def multiclass_noisify(clean_label, T, random_state=0):
    """ 
        Flip classes according to transition probability matrix T.
    """
    assert T.shape[0] == T.shape[1]
    assert np.max(clean_label) < T.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(T.sum(axis=1), np.ones(T.shape[1]))
    assert (T >= 0.0).all()

    m = clean_label.shape[0]
    noisy_label = clean_label.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = clean_label[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, T[i, :], 1)[0]
        noisy_label[idx] = np.where(flipped == 1)[0]

    return noisy_label

def get_T_true_from_data(clean_label, noisy_label):
    K = len(np.unique(clean_label))
    T_true = np.zeros((K,K))
    for i in range(len(clean_label)):
        T_true[clean_label[i]][noisy_label[i]] += 1
    T_true /= np.sum(T_true, 1).reshape(-1,1)
    return T_true

def print_samples(cfg, feature, label, indices):
    for i in indices:
        print(f'\nInstance {i+1} is:')
        print(f'Feature: {feature[i]}')
        if len(cfg.label) > 0:
            print(f'Label: {cfg.label[label[i]]}')
        
        else:
            print(f'Label: {label[i]}')
    # print(cfg.label)
    # cnt = collections.Counter(label)
    # label_key = {i: cnt[i] for i in cnt if cnt[i] > 100}
    # sorted_dict = dict(sorted(label_key.items(), key=lambda x: x[1])[::-1])
    # print(sorted_dict)
    # print(f'length of label: {len(label)}')

def load_csv(path):
    if not path.endswith('.csv'):
        path += '.csv'
    data = pd.read_csv(path)
    print(f'Load {path} finished')
    return data

def load_tsv(path):
    if not path.endswith('.tsv'):
        path += '.tsv'
    data = pd.read_csv(path, delimiter='\t')
    print(f'Load {path} finished')
    return data

def load_embedding(idx_list: list, data_path: Callable[[int], str], duplicate = True):
   
    if duplicate:
        div = 2
        inx_range = lambda x: list(range(idx_list[x-1]+1, idx_list[x])) + list(range(idx_list[x-1], idx_list[x]))
    else:
        div = 1
        inx_range = lambda x: list(range(idx_list[x-1]+1, idx_list[x]))

    dataset = torch.load(data_path(idx_list[0])) 
    print(f'idx range for training data {[idx_list[0]] + inx_range(1)}')
    for i in inx_range(1):    
        dataset_new = torch.load(data_path(i))
        dataset.update(dataset_new)
    print(f'#Samples (dataset-train) {len(dataset)//div}.')
    
    if len(idx_list) == 3:
        test_dataset = torch.load(data_path(idx_list[1])) 
        print(f'idx range for test data {[idx_list[1]] + inx_range(2)}')
        for i in inx_range(2):
            dataset_new = torch.load(data_path(i))
            test_dataset.update(dataset_new)
        print(f'#Samples (dataset-test) {len(test_dataset)//div}.')
    else:
        test_dataset = None
    return dataset, test_dataset


def load_dataset(cfg, data_converter, data_loader):
    dataset_path = cfg.dataset_path
    os.makedirs(cfg.save_path, exist_ok=True)
    if os.path.exists(dataset_path): 
        data = torch.load(dataset_path)
        print(f'load preprocessed dataset from {dataset_path}')
        feature = data['feature']
        label = data['label']
        index = len(feature)
    elif os.path.exists(dataset_path[:-3] + '0' + dataset_path[-3:]):
        i = 0
        feature = []
        label = []
        while 1:
            try:
                dataset_path_new = dataset_path[:-3] + f'{i}' + dataset_path[-3:]
                data = torch.load(dataset_path_new)
                print(f'load preprocessed dataset from {dataset_path_new}')
                feature += data['feature']
                label += data['label']
                i += 1
            except:
                break
        index = len(feature)        

    else:
        print(f'preprocessed dataset {dataset_path} not existed. Creat it.')

        data = data_loader(os.path.join(cfg.data_foldername, cfg.file_name))
        feature, label, index = data_converter(data)
        total_len = 10**6
        for i in range(len(label)//total_len + 1):
            torch.save({'feature': feature.tolist()[i*total_len:(i+1)*total_len], 'label': label.tolist()[i*total_len:(i+1)*total_len]}, dataset_path[:-3] + str(i) + dataset_path[-3:])
            print(f'Saved preprocessed dataset to {dataset_path[:-3] + str(i) + dataset_path[-3:]}')
    return np.asarray(feature),np.asarray(label), index

# if __name__ == "__main__":
#     data = load_csv('./data/jigsaw/train.csv')
#     pass

