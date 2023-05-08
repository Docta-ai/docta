# dataset settings
seed = 0
dataset_type = 'Clothing1M_subset'
modality = 'image' # image, text, tabular
num_classes = 14
data_root = './demo_imgs/c1m_subset/'
file_name = None
save_path = f'./results/{dataset_type}/'
dataset_path = save_path + f'dataset_{dataset_type}.pt'


details = False

feature_type = 'embedding' 


embedding_model = 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
embedding_cfg = dict(
    shuffle = False,
    batch_size = 100,
    save_num = 50,
    num_workers = 1,
    use_pca = False,
    use_mi = False,
    n_neighbors = 10
)
