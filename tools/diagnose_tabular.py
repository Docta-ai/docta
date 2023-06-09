import sys
import os
o_path = os.getcwd()
sys.path.append(o_path) # set path so that modules from other foloders can be loaded

import torch
import argparse

from docta.utils.config import Config
from docta.datasets import TabularDataset


torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--config', help='train config file path', default='./config/label_error_tabular.py')
    args = parser.parse_args()
    return args



args = parse_args()
cfg = Config.fromfile(args.config)
print(cfg.dataset_type)
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



dataset = TabularDataset(root_path=cfg.data_root)
test_dataset = None
print('Tabular-data load finished')


from docta.apis import DetectLabel
from docta.core.report import Report
report = Report()
detector = DetectLabel(cfg, dataset, report = report)
detector.detect()

os.makedirs(cfg.save_path, exist_ok=True)
report_path = cfg.save_path + f'{cfg.dataset_type}_diagnose_report.pt'
torch.save(report, report_path)
print(f'Report saved to {report_path}')