# detect label errors, rare patterns, unfair patterns
from .diagnose import Diagnose
from docta.core.knn import simi_feat_batch
import numpy as np
import time
from tqdm import tqdm

class DetectLabel(Diagnose):
    # detect label errors using either a learning-centric or data-centric method
    def __init__(self, cfg, dataset, model = None, report = None) -> None:
        """
            If model is None: use pretrained models.
            Otherwise, use the provided models.            
        """
        super(DetectLabel, self).__init__(cfg=cfg, dataset=dataset, model=model, report = report)
        self.all_methods = ['simifeat']



    def detect(self):
        name = self.cfg.detect_cfg.name
        # default detector is simifeat
        if name is None:
            return self.simifeat()
        elif name in self.all_methods:
            return eval('self.'+ name)()
        else:
            raise NameError(f'Undefined detector name. Input is {name}. Should be in [simifeat, model_conf]')



    def simifeat(self):
        # A data-centric method
        # Adapted from the paper: Z. Zhu, Z. Dong, Y. Liu. Detecting Corrupted Labels Without Training a Model to Predict. ICML'22.
        # Code: https://github.com/UCSC-REAL/SimiFeat
        # Paper: https://proceedings.mlr.press/v162/zhu22a/zhu22a.pdf
        print(f'Detecting label errors with simifeat.')
        num_epoch = self.cfg.detect_cfg.num_epoch
        sel_noisy_rec = np.zeros((num_epoch, len(self.dataset)))
        sel_times_rec = np.zeros(len(self.dataset))
        suggest_label_rec = np.zeros((len(self.dataset), self.cfg.num_classes))

        if self.cfg.detect_cfg.method == 'rank': # estimate T with hoc (by default) 
            if self.report.diagnose['T'] is None:
                self.hoc()
            T_given_noisy = self.report.diagnose['T'] * self.report.diagnose['p_clean'] / self.report.diagnose['p_org']
            if self.cfg.details:
                print("T given noisy:")
                print(np.round(T_given_noisy, 2))
            self.cfg.T_given_noisy = T_given_noisy

        print(f'Use SimiFeat-{self.cfg.detect_cfg.method} to detect label errors.')
        time0 = time.time()
        for epoch in tqdm(range(num_epoch)):
            if self.cfg.details:
                print(f'Epoch {epoch}. Time elapsed: {time.time() - time0} s')
            sel_noisy, sel_idx, suggest_label = simi_feat_batch(self.cfg, self.dataset)
            sel_noisy_rec[epoch][np.asarray(sel_noisy)] = 1
            sel_times_rec[np.asarray(sel_idx)] += 1
            suggest_label_rec[np.asarray(sel_noisy), suggest_label] += 1
        
        noisy_avg = (np.sum(sel_noisy_rec, 0) + 1) / (sel_times_rec + 2)
        # sel_clean_summary = np.round(1.0 - noisy_avg).astype(bool)
        sel_noisy_summary = np.round(noisy_avg).astype(bool)
        num_label_errors = np.sum(sel_noisy_summary)
        print(f'[SimiFeat] We find {num_label_errors} corrupted instances from {sel_noisy_summary.shape[0]} instances')
        idx = np.argsort(noisy_avg)[-num_label_errors:][::-1] # raw index
        suggest_matrix = (suggest_label_rec + 1) / (np.sum(suggest_label_rec, 1).reshape(-1,1) + self.cfg.num_classes) # #samples * #classes

        # update report
        detection = dict(
            label_error = [[i, noisy_avg[i]] for i in idx]
        )

        suggest_matrix[range(len(suggest_matrix)), np.array(self.dataset.label)] = -1
        curation = dict(
            label_curation = [[i, np.argmax(suggest_matrix[i]), suggest_matrix[i][np.argmax(suggest_matrix[i])] * noisy_avg[i]] for i in idx]
        )
        self.report.update(detection=detection, curation=curation)



    

class DetectFeature(Diagnose):
    
    def __init__(self, cfg, data, report = None) -> None:
        super(DetectFeature, self).__init__(cfg=cfg, dataset=data, report=report).__init__()

    
    def rare_score(self):
        from docta.core.get_lr_score import lt_score                
        self.rare_scores = lt_score(data=self.dataset, feature_type=self.cfg.feature_type, k=self.cfg.embedding_cfg.n_neighbors)
        detection = dict(
            rare_example = [[i, self.rare_scores[i]] for i in range(len(self.rare_scores))]
        )
        self.report.update(detection=detection)


class DetectPattern(DetectLabel):
    
    def __init__(self, cfg, dataset, model = None) -> None:
        super(DetectPattern, self).__init__(cfg=cfg, dataset=dataset, model=model).__init__()
        pass
