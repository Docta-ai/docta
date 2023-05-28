# dataset settings
seed = 0
dataset_type = 'HH-RLHF'
modality = 'text' # image, text, tabular
num_classes = 5
data_root = './data/hh-rlhf/'
file_name = 'red-team-attempts'
preprocess = 'raw' # QA, raw, summarize
dataset_type += '_' + file_name + '_' + preprocess
save_path = f'./results/{dataset_type}/'
dataset_path = save_path + f'dataset_{dataset_type}.pt'

details = False

embedding_model = 'sentence-transformers/all-mpnet-base-v2'
embedding_cfg = dict(
    shuffle = False,
    batch_size = 256,  
    save_num = 800,
    num_workers = 2,
)


hoc_cfg = dict(
    max_step = 1501, 
    T0 = None, 
    p0 = None, 
    lr = 0.1, 
    num_rounds = 50, 
    sample_size = 50000,
    already_2nn = False,
    device = 'cpu'
)


detect_cfg = dict(
    num_epoch = 51,
    sample_size = 50000,
    k = 10,
    name = 'simifeat',
    method = 'rank'
)