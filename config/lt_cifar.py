# dataset settings
seed = 0
dataset_type = 'CIFAR'
modality = 'image' # image, text, tabular
num_classes = 10
data_root = './data/cifar/'

file_name = 'c10'
dataset_type += '_' + file_name
save_path = f'./results/{dataset_type}/'
feature_type = 'embedding'

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

infl_cfg = dict(
    shuffle = False,
    batch_size = 32,
    num_workers = 1,
)


embedding_model = 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
embedding_cfg = dict(
    shuffle = False,
    batch_size = 1024,
    save_num = 50,
    num_workers = 1,
    use_pca = False,
    use_mi = False,
    n_neighbors = 10
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
    sample_size = 15000,
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