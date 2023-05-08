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

def wrapup_results(dataset, report, warpped_result = [], weight = 1.0):
    # process raw
    raw_label_rec = []
    for i in range(len(dataset.feature)): # init
        sample_i = dict(
            raw_idx = int(dataset.label[1][i]),
            conf_corrupted = 0.0,
            curation = int(dataset.label[0][i]),
            conf_cure = 1.0,
            feature = dataset.feature[i],
            label = int(dataset.label[0][i]),
        )
        raw_label_rec.append(sample_i)

    label_error = np.array(report.detection['label_error'])
    label_curation = np.array(report.curation['label_curation'])
    assert (label_curation[:,0] == label_error[:,0]).all()
    idx = label_error[:, 0].astype(int) 
    for i in range(len(idx)): # label error detection and curation
        raw_label_rec[idx[i]]['conf_corrupted'] = label_error[i][1]
        raw_label_rec[idx[i]]['curation'] = label_curation[i][1]
        raw_label_rec[idx[i]]['conf_cure'] = label_curation[i][2]

    # warp-up results
    if warpped_result == []:
        warpped_result = [{'conf_corrupted':[], 'conf_cure':[], 'curation':[], 'feature':[], 'label': 1, 'feature_raw': []} for i in range(int(max(dataset.label[1])+1))]
    for i in range(len(raw_label_rec)):
        warpped_result[raw_label_rec[i]['raw_idx']]['conf_corrupted'].append(raw_label_rec[i]['conf_corrupted'] * weight)
        warpped_result[raw_label_rec[i]['raw_idx']]['curation'].append(raw_label_rec[i]['curation'])
        warpped_result[raw_label_rec[i]['raw_idx']]['conf_cure'].append(raw_label_rec[i]['conf_cure'] * weight)
        warpped_result[raw_label_rec[i]['raw_idx']]['feature'].append(raw_label_rec[i]['feature'])
        warpped_result[raw_label_rec[i]['raw_idx']]['label'] = raw_label_rec[i]['label']
        warpped_result[raw_label_rec[i]['raw_idx']]['feature_raw'] = dataset.feature_raw[raw_label_rec[i]['raw_idx']]
    return warpped_result

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.fromfile(f'./config/hh_rlhf_red-team-attempts_raw.py')

warpped_result = []
# weight = {'raw': 0.5, 'QA': 1.0, 'summarize': 1.0}
weight = {'raw': 0.5, 'QA': 1.0}
for key in ['raw', 'QA']:
    cfg = Config.fromfile(f'./config/hh_rlhf_red-team-attempts_{key}.py')
    print(cfg.dataset_type)

    # print("chosen: AI Assistant is harmless or helpful\nrejected: AI Assistant is harmful or not helpful")

    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    report = torch.load(cfg.save_path + f'{cfg.dataset_type}_report.pt')

    dataset = HH_RLHF(cfg, train=True)

    warpped_result = wrapup_results(dataset, report, warpped_result = warpped_result, weight = weight[key])
    print(key)

# summarize results
summarized_result = []
conf_detection_rec = []
for i in range(len(warpped_result)):
    
    # update conf_cure
    warpped_result[i]['conf_cure'] = np.asarray(warpped_result[i]['conf_cure'])
    warpped_result[i]['conf_cure'][:-1] /= (len(warpped_result[i]['conf_cure']) - 1.0) / 2.0 # all methods have equal weights


    conf_detection = np.mean(warpped_result[i]['conf_corrupted'])
    avg_conf_curation = np.asarray(warpped_result[i]['conf_cure'])
    avg_conf_curation /= np.sum(avg_conf_curation)
    curation = np.asarray(warpped_result[i]['curation']).astype(int)
    conf_curation = np.array([np.sum(avg_conf_curation[curation==i]) for i in range(cfg.num_classes)])


    conf_curation /= np.sum(conf_curation)

    summarized_result.append(dict(
        conf_detection = conf_detection,
        conf_curation = conf_curation,
        label = warpped_result[i]['label'],
        feature_raw = warpped_result[i]['feature_raw'],
    ))
    conf_detection_rec.append(conf_detection)


order = np.argsort(conf_detection_rec)[::-1]

name = ['Not harmful (1/5)', 'Mild harmful (2/5)', 'Moderate harmful (3/5)', 'Harmful (4/5)', 'Severe harmful (5/5)']

cnt = 0
for i in order:
    suggest_confidence = summarized_result[i]['conf_curation']
    if np.isnan(suggest_confidence[0]):
        dataset.all[i]['suggest_rating'] = int(dataset.all[i]['rating'])
        dataset.all[i]['suggest_confidence'] = conf_detection_rec[i]
        continue
    suggest_label = np.argmax(summarized_result[i]['conf_curation'])
    if suggest_label == 0:
        suggest_label = int(np.round(np.sum(suggest_confidence * np.arange(len(suggest_confidence)))))
    if conf_detection_rec[i] > 0.4:
        dataset.all[i]['suggest_rating'] = int(suggest_label)
        dataset.all[i]['suggest_confidence'] = conf_detection_rec[i]
        if abs(suggest_label - summarized_result[i]['label']) >= 1:
            cnt += 1
    else:
        dataset.all[i]['suggest_rating'] = int(dataset.all[i]['rating'])
        dataset.all[i]['suggest_confidence'] = conf_detection_rec[i]


    



print(f'# total corrupted labels: {cnt}. Total data: {len(order)}')

import gzip, json
save_path = dataset.jsonfilename[:-9] + '_docta.jsonl.gz'
with gzip.open(save_path, 'wt', encoding='utf-8') as f:
    for item in dataset.all:
        f.write(json.dumps(item) + '\n')

print(f'cleaned datasets saved to {save_path}')
