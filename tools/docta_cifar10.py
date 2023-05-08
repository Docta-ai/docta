import sys
import os
o_path = os.getcwd()
sys.path.append(o_path) # set path so that modules from other foloders can be loaded

import torch
from docta.utils.config import Config
from docta.datasets import Cifar10_noisy
from docta.datasets.data_utils import load_embedding

from docta.core.preprocess import Preprocess
from docta.core.report import Report

from docta.apis import DetectLabel
from docta.apis import Diagnose

cfg = Config.fromfile('./config/cifar10.py')
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Cifar10_noisy(cfg, train=True)
test_dataset = None

# preprocess the dataset, get embeddings
pre_processor = Preprocess(cfg, dataset, test_dataset)
pre_processor.encode_feature()
print(pre_processor.save_ckpt_idx)


data_path = lambda x: cfg.save_path + f'embedded_{cfg.dataset_type}_{x}.pt'
dataset, _ = load_embedding(pre_processor.save_ckpt_idx, data_path, duplicate=True)

# initialize report
report = Report()

# diagnose labels
estimator = Diagnose(cfg, dataset, report = report)
estimator.hoc()

# print diagnose reports
import numpy as np
np.set_printoptions(precision=1, suppress=True)
T = report.diagnose['T']
p = report.diagnose['p_clean']
print(f'T_est is \n{T * 100}')
print(f'p_est is \n{p * 100}')

# label error detection
detector = DetectLabel(cfg, dataset, report = report)
detector.detect()

# print results
dataset_raw = Cifar10_noisy(cfg, train=True)
label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
label_error = np.array(report.detection['label_error'])
label_curation = np.array(report.curation['label_curation'])
idx = label_error[:, 0].astype(int)
sel = label_curation[:, 2] > 0.2
print(f'Found {np.sum(sel)} label errors from {len(dataset_raw)} samples')

# generate cured labels
cured_labels = dataset_raw.label[:, 1]
cured_labels[label_curation[sel, 0].astype(int)] = label_curation[sel, 1].astype(int)
save_path = cfg.save_path + f'cured_labels_{cfg.dataset_type}.pt'
torch.save(cured_labels, save_path)
print(f'Saved cured labels to {save_path}')
