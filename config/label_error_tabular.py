# dataset settings
seed = 0
dataset_type = 'Tabular'
modality = 'tabular' # image, text, tabular
data_root = './demo_imgs/noisy_tabular/noisy_Iris.csv'
file_name = 'train'
save_path = f'./results/{dataset_type}/'
dataset_path = save_path + f'dataset_{dataset_type}.pt'
dataset_type += '_' + file_name
save_path = f'./results/{dataset_type}/label_error_'

feature_type = 'embedding' 

details = False


hoc_cfg = dict(
    max_step = 1501, 
    T0 = None, 
    p0 = None, 
    lr = 0.1, 
    num_rounds = 50, 
    sample_size = 35000,
    already_2nn = False,
    device = 'cpu'
)


detect_cfg = dict(
    num_epoch = 51,
    sample_size = 35000,
    k = 10,
    name = 'simifeat',
    method = 'rank'
)