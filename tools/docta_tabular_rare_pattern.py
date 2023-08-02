import sys
import os
# o_path = os.getcwd()
# sys.path.append(o_path) # set path so that modules from other foloders can be loaded
import pandas as pd
import torch


from docta.utils.config import Config
from docta.datasets import Customize_Image_Folder, Cifar10_clean, TabularDataset
from docta.datasets.data_utils import load_embedding
from docta.core.preprocess import Preprocess
from docta.core.get_lr_score import lt_score
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--feature_type', default='embedding')
    parser.add_argument('--suffix', default='tabular', help='cifar, c1m, c1m_subset, tabular')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.fromfile(f'./config/lt_{args.suffix}.py')
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.feature_type = args.feature_type
if 'CIFAR' in cfg.dataset_type:
    dataset = Cifar10_clean(cfg, train=True)
elif 'Tabular' in cfg.dataset_type:
    dataset = TabularDataset(root_path=cfg.data_root)
else:
    dataset = Customize_Image_Folder(root=cfg.data_root, transform=None)
test_dataset = None

# preprocess the dataset, get embeddings
pre_processor = Preprocess(cfg, dataset, test_dataset)

if cfg.feature_type == 'embedding':

    if 'Tabular' not in cfg.dataset_type:
        # For non-tabular data, we will adopt feature encoder to extract features from your data
        pre_processor.encode_feature() 
        print(pre_processor.save_ckpt_idx)
        data_path = lambda x: cfg.save_path + f'embedded_{cfg.dataset_type}_{x}.pt'
        dataset, _ = load_embedding(pre_processor.save_ckpt_idx, data_path, duplicate=False)
    data = dataset.feature
else:
    raise NotImplementedError(f'feature_type {cfg.feature_type} not defined.')


longtail_scores = lt_score(data=data, feature_type=cfg.feature_type, k=cfg.embedding_cfg.n_neighbors)
dict_df = {'idx': [i for i in range(len(longtail_scores))], 'longtail_scores': longtail_scores}
df = pd.DataFrame(dict_df)
os.makedirs(cfg.save_path, exist_ok=True)
lt_path = f'{cfg.save_path + cfg.feature_type}.csv'
df.to_csv(lt_path, index=False)
print(f"Long-tail score saved to {lt_path}")
