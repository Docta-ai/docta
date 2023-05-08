import sys
import os
o_path = os.getcwd()
sys.path.append(o_path) # set path so that modules from other foloders can be loaded

import torch
import argparse

from docta.utils.config import Config
from docta.datasets import HH_RLHF
from docta.core.preprocess import Preprocess
from docta.datasets.data_utils import load_embedding

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--config', help='train config file path', default='./config/hh_rlhf_red-team-attempts_QA.py')
    args = parser.parse_args()
    return args




args = parse_args()
cfg = Config.fromfile(args.config)
print(cfg.dataset_type)
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



dataset = HH_RLHF(cfg, train=True)
if 'red_team_attempts' in dataset.jsonfilename:
    dataset.label = dataset.label[0].astype(float).astype(int)
test_dataset = None
print('HH_RLHF load finished')


pre_processor = Preprocess(cfg, dataset, test_dataset)
pre_processor.encode_feature()
print(pre_processor.save_ckpt_idx)


data_path = lambda x: cfg.save_path + f'embedded_{cfg.dataset_type}_{x}.pt'
dataset, _ = load_embedding(pre_processor.save_ckpt_idx, data_path, duplicate=True)



from docta.apis import DetectLabel
from docta.core.report import Report
report = Report()
detector = DetectLabel(cfg, dataset, report = report)
detector.detect()

report_path = cfg.save_path + f'{cfg.dataset_type}_report.pt'
torch.save(report, report_path)
print(f'Report saved to {report_path}')