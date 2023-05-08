# estimate T with training-based methods (given a trained model) or training-free methods (use a pre-trained model)

from docta.core.hoc import estimator_hoc
from docta.core.report import Report

# from docta.core.preprocess import extract_image_embedding

import numpy as np

class Diagnose:
    def __init__(self, cfg, dataset, model = None, report = None) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.model = model
        self.report = Report() if report is None else report
        self.all_methods = ['hoc', 'dnn', 'anchor']



    def hoc(self):
        T_est, p_est, _ = estimator_hoc(cfg=self.cfg, dataset=self.dataset)
        noisy_posterior = np.array([sum(self.dataset.label == i) for i in range(self.cfg.num_classes)]) * 1.0
        noisy_posterior /= np.sum(noisy_posterior)
        diagnose = dict(
            T = T_est,
            p_clean = p_est,
            p_org = noisy_posterior,
        )
        self.report.update(diagnose=diagnose)

