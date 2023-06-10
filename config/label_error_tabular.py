# dataset settings
seed = 0
dataset_type = 'Tabular'
modality = 'tabular' # image, text, tabular
num_classes = 2
data_root = './demo_imgs/noisy_tabular/noisy_twonorm.csv'
file_name = 'train'
save_path = f'./results/{dataset_type}/'
dataset_path = save_path + f'dataset_{dataset_type}.pt'
dataset_type += '_' + file_name
save_path = f'./results/{dataset_type}/label_error_'

feature_type = 'embedding' 

embedding_model = 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
embedding_cfg = dict(
    save_num = 50,
    shuffle = False,
    batch_size = 1024,
    num_workers = 1,
)


accuracy = dict(topk = 1, threth = 0.5)
n_epoch = 10
print_freq = 390
details = False

train_cfg = dict(
    shuffle = True,
    batch_size = 128,
    num_workers = 1,
)

test_cfg = dict(
    shuffle = False,
    batch_size = 128,
    num_workers = 1,
)


optimizer = dict(
    name = 'SGD',
    config = dict(
        lr = 0.1    
    )
)


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