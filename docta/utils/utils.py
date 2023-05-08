import copy

def dataset_cfg(all_datasets):
    new_datasets = {}
    for i in all_datasets:
        cfg_data = all_datasets[i]
        if isinstance(cfg_data.file_name, list):
            file_name_list = cfg_data.file_name
        else:
            file_name_list = [cfg_data.file_name]
        if 'preprocess' in cfg_data:
            if isinstance(cfg_data.preprocess, list):
                preprocess_list = cfg_data.preprocess
            else:
                preprocess_list = [cfg_data.preprocess]
        else:
            preprocess_list = ['raw']
        for file_name_i in file_name_list:
                for preprocess_i in preprocess_list:
                    cfg_tmp = copy.deepcopy(cfg_data)
                    cfg_tmp.file_name = file_name_i
                    cfg_tmp.preprocess = preprocess_i
                    name_new_dataset = i.split('/')[0] + '_' + file_name_i + '_' + preprocess_i
                    new_datasets[name_new_dataset] = _dataset_cfg(**cfg_tmp)
    return new_datasets
        

def _dataset_cfg(dataset_name, file_name, label, 
                  preprocess = 'raw', 
                  label_sel = None,
                  embedding_model = 'sentence-transformers/all-mpnet-base-v2',
                  data_root = './data/',
                  save_root = './results/'):
    
    dataset_type = dataset_name + '_' + file_name if preprocess is None else dataset_name + '_' + file_name + '_' + preprocess
    dataset_type = dataset_type.replace('/', '_')
    save_path = save_root + dataset_type + '/'

    data_cfg = dict(
        file_name = file_name,
        dataset_name = dataset_name,
        label = label,
        label_sel = label_sel,
        num_classes = len(label),
        preprocess = preprocess,
        embedding_model = embedding_model,
        dataset_type = dataset_type,
        data_root = data_root + dataset_name + '/' if data_root == './data/' else data_root,
        save_path = save_path,
        dataset_path = save_path + f'/dataset_{dataset_type}.pt'
    )
    return data_cfg