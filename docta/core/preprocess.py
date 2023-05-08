import torch
import numpy as np
from docta.datasets import CustomizedDataset, Customize_Image_Folder, Cifar10_clean
from docta.datasets.data_utils import load_embedding
from .core_utils import mean_pooling
import open_clip
import torchvision
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os
from tqdm import tqdm


def build_dataloader(cfg_loader, dataset):
    data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                   batch_size = cfg_loader.batch_size,
                                   num_workers = cfg_loader.num_workers,
                                   shuffle = cfg_loader.shuffle)
    return data_loader

def save_extracted_dataset(cfg, dataset_embedding, dataset_label, dataset_idx, save_cnt):
    dataset_label = np.concatenate(dataset_label)
    dataset_label = dataset_label[:, cfg.train_label_sel] if len(dataset_label.shape) > 1 else dataset_label

    dataset = CustomizedDataset(feature=np.concatenate(dataset_embedding), label=dataset_label, index=np.concatenate(dataset_idx))
    os.makedirs(cfg.save_path, exist_ok=True)
    save_path = cfg.save_path + f'embedded_{cfg.dataset_type}_{save_cnt}.pt'
    torch.save(dataset, save_path)
    print(f'Save {len(dataset_idx)} instances to {save_path}')


    
def extract_embedding(cfg, encoder, dataset_list):
    save_cnt = 0
    save_num = cfg.embedding_cfg.save_num
    ckpt_idx = [0]
    for dataset in dataset_list:
        train_loader = build_dataloader(cfg.embedding_cfg, dataset)
        dataset_embedding, dataset_idx, dataset_label = [], [], []
        for i, batch in tqdm(enumerate(train_loader)):
            """ 
                batch should always be tuple. Must follow the structure:
                (feature, label/attribute_tuple, index)
            """
            batch_feature = batch[0] 
            embedding = extract_embedding_batch(cfg, encoder, batch_feature)
            dataset_embedding.append(embedding.cpu().numpy())
            if isinstance(batch[1], list):
                dataset_label.append(torch.stack(batch[1]).cpu().numpy().transpose())
            else:
                dataset_label.append(batch[1].cpu().numpy())
            dataset_idx.append(batch[2])
            # print(i)
            if (i+1) % save_num == 0:
                save_extracted_dataset(cfg, dataset_embedding, dataset_label, dataset_idx, save_cnt)
                dataset_embedding, dataset_idx, dataset_label = [], [], []
                save_cnt += 1

        if len(dataset_label) > 0:   
            save_extracted_dataset(cfg, dataset_embedding, dataset_label, dataset_idx, save_cnt)
            dataset_embedding, dataset_idx, dataset_label = [], [], []
            save_cnt += 1
        ckpt_idx.append(save_cnt)
    return ckpt_idx



def extract_embedding_batch(cfg, encoder, batch_feature):
    try:
        encoder, tokenizer = encoder
        encoded_input = tokenizer(batch_feature).to(cfg.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            model_output = encoder(**encoded_input)
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        embedding = F.normalize(sentence_embeddings, p=2, dim=1)
    except: 
        with torch.no_grad(), torch.cuda.amp.autocast():
            embedding = encoder(batch_feature.to(cfg.device))
    return embedding






class Preprocess:

    def __init__(self, cfg, dataset, test_dataset=None) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.test_dataset = test_dataset
    
    def get_encoder(self):
        if 'CLIP' in self.cfg.embedding_model:
            model_embedding, _, preprocess = open_clip.create_model_and_transforms(self.cfg.embedding_model)
            model_embedding.to(self.cfg.device)

            if self.cfg.modality == 'image':
                preprocess = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(mode=None),
                preprocess
                ]) # from array or tensor to PIL Image, for resize
                return model_embedding.encode_image, preprocess
            elif self.cfg.modality == 'text':
                tokenizer = open_clip.get_tokenizer(self.cfg.embedding_model)
                return model_embedding.encode_text, tokenizer
            else:
                raise NameError(f'Modality {self.cfg.modality} has not been supported yet')
        elif 'sentence-transformers' in self.cfg.embedding_model:
            # model = SentenceTransformer(self.cfg.embedding_model)
            
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.embedding_model)
            model = AutoModel.from_pretrained(self.cfg.embedding_model).to(self.cfg.device)
            def token_st(sentences):
                return tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

            return (model, token_st), None


        else:
            raise NameError('Undefined pre-trained models {self.cfg.embedding_model}')


    def encode_feature(self):
        encoder, preprocess = self.get_encoder()
        dataset_list = []
        dataset_list += [CustomizedDataset(feature=self.dataset.feature, label=self.dataset.label, preprocess=preprocess)]
        if self.test_dataset is not None:
            dataset_list += [CustomizedDataset(feature=self.test_dataset.feature, label=self.test_dataset.label, preprocess=preprocess)]
        self.save_ckpt_idx = extract_embedding(self.cfg, encoder, dataset_list) # idx of different saved files


    def preprocess_rare_pattern(self):
        data = None
        cfg = self.cfg
        if self.test_dataset is not None:
            print('Ignore test_dataset')
        if cfg.feature_type == 'embedding':
            self.encode_feature() 
            print(self.save_ckpt_idx)
            data_path = lambda x: cfg.save_path + f'embedded_{cfg.dataset_type}_{x}.pt'
            dataset, _ = load_embedding(self.save_ckpt_idx, data_path, duplicate=False)
            data = dataset.feature
        else:
            raise NotImplementedError(f'feature_type {cfg.feature_type} not defined.')

        self.data_rare_pattern = data
