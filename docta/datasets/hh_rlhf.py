from .customize import CustomizedDataset
import gzip, json, re
import numpy as np
import os
import torch


class HH_RLHF(CustomizedDataset):
    def __init__(self, cfg, train=True):
        filename = cfg.file_name

        if filename == 'red-team-attempts':
            train = 'red_team_attempts'
        elif filename == 'red_team_attempts_docta':
            dataset_type = 'HH-RLHF'
            dataset_type += '_' + filename
            cfg.dataset_path = cfg.save_path + f'dataset_{dataset_type}.pt'
            train = 'red_team_attempts_docta'
            filename = 'red-team-attempts'

        else:
            train = 'train' if train else 'test'
            if '_docta' in filename:
                dataset_type = 'HH-RLHF'
                dataset_type += '_' + filename
                cfg.dataset_path = cfg.save_path + f'dataset_{dataset_type}.pt'
                train += '_docta'
                filename = filename.strip('_docta')
        self.jsonfilename = cfg.data_root + f"/{filename}/{train}.jsonl.gz"
        print(f'Json filename is {self.jsonfilename}')
        try:
            self.load_data()
        except:
            raise ImportError(f'Data must be downloaded from https://github.com/anthropics/hh-rlhf and saved to {cfg.data_root}. self.jsonfilename is {self.jsonfilename}')
        self.feature_raw = self.chosen + self.rejected
        dataset_path = cfg.dataset_path

        # load & save datasets
        os.makedirs(cfg.save_path, exist_ok=True)
        if os.path.exists(dataset_path) and '_docta' not in self.jsonfilename: 
            data = torch.load(dataset_path)
            print(f'load preprocessed dataset from {dataset_path}')
            feature = data['feature']
            label = data['label']
            if len(label) == 3: 
                label[1] = label[2].copy()
                label = label.astype(float)

        else:
            print(f'preprocessed dataset {dataset_path} not existed. Creat it.')

            if 'red_team_attempts' not in self.jsonfilename:
                self.filter_data(key = 'Assistant:')
                feature = self.result['chosen'] + self.result['rejected']
                for i in range(len(feature)):
                    feature[i] = ' '.join(feature[i])
                label = [1] * len(self.result['chosen']) + [0] * len(self.result['rejected'])
            else:
                feature = []
                label = [[], []]
                
                if cfg.preprocess == 'raw':
                # each response as a feature. Check each A individually
                    for i in range(len(self.chosen)):
                        sample = self.chosen[i]
                        feature += sample['Assistant:']
                        label[0] += [self.rejected[i]] * len(sample['Assistant:'])
                        label[1] += [i] * len(sample['Assistant:'])  # original index
                        
                elif cfg.preprocess == 'QA':
                # each Q&A as a feature. Inputs that have more than 376 tokens will be truncated
                    for i in range(len(self.chosen)):
                        sample = self.chosen[i]
                        feature += [ ' '.join(' '.join(['[Human:]', sample['Human:'][i], '[Assistant:]', sample['Assistant:'][i]]).split(' ')[-374:]) for i in range(min(len(sample['Human:']), len(sample['Assistant:']))) ] 
                        label[0] += [self.rejected[i]] * min(len(sample['Human:']), len(sample['Assistant:']))
                        label[1] += [i] * min(len(sample['Human:']), len(sample['Assistant:']))  # original index
                
                else:
                    raise NameError(f'Undefined preprocess {cfg.preprocess}')
                
            label = np.array(label)
            torch.save({'feature': feature, 'label': label}, dataset_path)
            print(f'Saved preprocessed dataset to {dataset_path}')

        index = range(len(feature))
        super(HH_RLHF, self).__init__(feature, label, index=index, preprocess=None)
                
                


    def split_string_by_keywords(self, input_str, keywords):
        regex = re.compile('({})'.format('|'.join(map(re.escape, keywords))))
        substrings = regex.split(input_str.strip())
        substrings = [s.strip() for s in substrings if len(s.strip()) > 0]
        result = {}
        for keyword in keywords:
            result[keyword] = [substrings[i+1] for i in range(len(substrings) - 1) if substrings[i].startswith(keyword) and (substrings[i+1] not in keywords)]
        return result # divide responses according to human/assistant
        # dict{Human: xxx, Assistant: xxx}

    
    def load_data(self):
        chosen, rejected, all = [], [], []
        with gzip.open(self.jsonfilename, mode="rt", encoding='utf-8') as f:
            data = f.read().strip().splitlines()
            for i in range(len(data)):
                if 'red_team_attempts' in self.jsonfilename and '_docta' not in self.jsonfilename:
                    json_tmp = json.loads(data[i][1:-1]) if i == 0 else json.loads(data[i][:-1])
                else:
                    json_tmp = json.loads(data[i])

                if 'red_team_attempts' in self.jsonfilename:
                    all.append(json_tmp)
                    chosen_split = self.split_string_by_keywords(json_tmp['transcript'].replace('\n', " ").strip(), keywords = ['Human:', 'Assistant:'])
                    chosen.append(chosen_split)
                    rejected.append(json_tmp['rating']) # it is a rating in this case
                else:
                    if 'suggest_chosen_rejected' not in json_tmp:
                        json_tmp['suggest_chosen_rejected'] = ['chosen', 'rejected']
                    if 'suggest_confidence' not in json_tmp:
                        json_tmp['suggest_confidence'] = 0.5
                    all.append(json_tmp)
                    chosen_split = self.split_string_by_keywords(json_tmp['chosen'].replace('\n', " ").strip(), keywords = ['Human:', 'Assistant:'])
                    chosen.append(chosen_split)
                    

                    rejected_split = self.split_string_by_keywords(json_tmp['rejected'].replace('\n', " ").strip(), keywords = ['Human:', 'Assistant:'])
                    rejected.append(rejected_split)
                
        self.chosen, self.rejected, self.all = chosen, rejected, all


    def filter_data(self, key = 'Assistant:'):
        rec, chosen_filtered, rejected_filtered = [], [], []
        for i in range(len(self.chosen)):
            chosen = self.chosen[i][key]
            rejected = self.rejected[i][key]
            if len(chosen) == 0: # 
                chosen = [self.chosen[i]['Human:'][2*j + 1] for j in range(len(self.chosen[i]['Human:'])//2)] 
            
            if len(rejected) == 0: # 
                rejected = [self.rejected[i]['Human:'][2*j + 1] for j in range(len(self.rejected[i]['Human:'])//2)] 

            cnt = 0
            range_i = min(len(chosen), len(rejected))
            for j in range(range_i):
                if chosen[j] != rejected[j]:
                    cnt += 1
                    chosen_filtered.append(chosen[j:])
                    rejected_filtered.append(rejected[j:])
            if cnt == 0:
                chosen_filtered.append(chosen[j:])
                rejected_filtered.append(rejected[j:])
            rec.append(cnt) # rec must be no larger than 1

        assert max(rec) == 1
        self.result = dict(
            chosen = chosen_filtered,
            rejected = rejected_filtered
        )

