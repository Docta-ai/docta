class Report:
    def __init__(self, **kwargs) -> None:

        self.names = self.__dict__

        self.diagnose = kwargs['diagnose'] if 'diagnose' in kwargs else dict(
            T = None, # square matrix
            p_clean = None, # row vector
            p_org = None, # row vector
            class_distribution = None, # histogram
            group_distribution = None, # histogram           
        )
        self.detection = kwargs['detection'] if 'detection' in kwargs else dict(
            label_error = None, # (index, confidence)
            coexistence = None, # (index, confidence)
            rare_example = None, # (index, confidence)          
        )
        self.curation = kwargs['curation'] if 'curation' in kwargs else dict(
            label_curation = None, # (index, suggested_label, confidence)
            sampling_strategy = None, # (index, suggested_sample)
            feature_curation = None, # (index, suggested_feature)          
        )
        self.audition = kwargs['audition'] if 'audition' in kwargs else dict(
            model_perf = None, # (head_perf, tail_perf, overall_perf). With label, with noisy label, or without label
            fairness = None, # (group_vec, performance_vec), disparity  
            stress_test = None, # model_perf after distribution shift
        )


    
    def update(self, **kwargs):
        for i in kwargs:
            self.names[i] = kwargs[i]



if __name__ == '__main__':
    report = Report()
    dict_test1 = dict(
            label_error = 1, # (index, confidence)
            coexistence = 2, # (index, confidence)
            rare_example = None, # (index, confidence)          
        )
    dict_test2 = dict(
            label_curation = 3, # (index, suggested_label)
            sampling_strategy = 4, # (index, suggested_sample)
            feature_curation = None, # (index, suggested_feature)          
        )
    report.update(detection = dict_test1, curation = dict_test2)