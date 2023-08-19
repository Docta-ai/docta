# dataset settings
seed = 0
dataset_type = 'CIFAR'
modality = 'image' # image, text, tabular
num_classes = 100
data_root = './data/cifar/'
label_sel = 1 # which label/attribute we want to diagnose
train_label_sel = label_sel # 1 for noisy
test_label_sel = train_label_sel

file_name = 'c100'
dataset_type += '_' + file_name
save_path = f'./results/{dataset_type}/'