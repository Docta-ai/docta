import sys
import os
o_path = os.getcwd()
sys.path.append(o_path) # set path so that modules from other foloders can be loaded

import torch
from docta.datasets import HH_RLHF
from docta.utils.config import Config
import numpy as np

import argparse

def print_dialogue(dialogue): # print dialogue between human and assistant
    human = dialogue['Human:']
    assistant = dialogue['Assistant:']
    while human or assistant:
        try:
            print(f'[Human:] {human.pop(0)}')
        except:
            print(f'[Human:] ')
        try:
            print(f'[Assistant:] {assistant.pop(0)}')
        except:
            print(f'[Assistant:] ')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--config_key', help='[harmless-base, helpful-base, helpful-online, helpful-rejection-sampled]', default='helpful-rejection-sampled')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.fromfile(f'./config/hh_rlhf_{args.config_key}.py')
print(cfg.dataset_type)

print("chosen: AI Assistant is harmless or helpful\nrejected: AI Assistant is harmful or not helpful")

cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


report = torch.load(f'./results/HH-RLHF_{args.config_key}/HH-RLHF_{args.config_key}_report.pt')

dataset_raw = HH_RLHF(cfg, train=True)
label_name = ['rejected', 'chosen']

label_error = np.array(report.detection['label_error'])
label_curation = np.array(report.curation['label_curation'])
idx = label_error[:, 0].astype(int)

cnt = 0
total_len = len(dataset_raw)
print(f'total {len(idx)}')

suggest_confidence_all = [i[2] for i in label_curation]
order = np.argsort(suggest_confidence_all)[::-1]

chang_idx = set()
reversed_idx = set()
cnt = 0
for i in order:
    
    suggest_label = label_curation[i][1].astype(int)
    suggest_confidence = label_curation[i][2]
    if suggest_confidence > 0.55:
        dataset_raw.all[idx[i] % len(dataset_raw.all)]['suggest_chosen_rejected'][1-dataset_raw.label[idx[i]]] = label_name[suggest_label]
        
        chang_idx.add(idx[i] % len(dataset_raw.all))
        if len(np.unique(dataset_raw.all[idx[i] % len(dataset_raw.all)]['suggest_chosen_rejected'])) > 1:
            reversed_idx.add(idx[i] % len(dataset_raw.all))
        dataset_raw.all[idx[i] % len(dataset_raw.all)]['suggest_confidence'] = suggest_confidence

    else:
        dataset_raw.all[idx[i] % len(dataset_raw.all)]['suggest_chosen_rejected'] = ['chosen', 'rejected']
        dataset_raw.all[idx[i] % len(dataset_raw.all)]['suggest_confidence'] = 1.0 - suggest_confidence

print(f'{len(chang_idx)}/{len(dataset_raw.all)} = {len(chang_idx)/len(dataset_raw.all)}')
print(f'Reversed pairs {len(reversed_idx)}/{len(chang_idx)} = {len(reversed_idx)/len(chang_idx)}')

import gzip, json
save_path = dataset_raw.jsonfilename[:-9] + '_docta.jsonl.gz'
with gzip.open(save_path, 'wt', encoding='utf-8') as f:
    for item in dataset_raw.all:
        f.write(json.dumps(item) + '\n')

print(f'Cleaned datasets saved to {save_path}')

