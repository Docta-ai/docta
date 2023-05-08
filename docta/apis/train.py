# functions for training a model

from docta.core.preprocess import build_dataloader
from docta.models.loss_funcs import Accuracy

import torch




def set_optimizer(cfg, model):
    if 'name' in cfg:
        opt_clsname = getattr(torch.optim, cfg.name)
        optimizer = opt_clsname(model.parameters(), **cfg.config)
        print(f'Optimizer {cfg}')
    else:
        try:
           lr = cfg.config.lr
           optimizer = torch.optim.SGD(model.parameters(), lr = lr)
           print(f'Optimizer is not specialized. Use SGD with lr = {lr} by default')
        except:
           print(f'Optimizer is not specialized. Use SGD with lr = 0.1 by default')
    return optimizer
        


def test_model(cfg, model, dataset):
    # model: torch model
    # dataset: torch dataset
    # cfg: config. See ./utils/config.py
    accuracy = Accuracy(**cfg.accuracy)
    

    # print('build dataloader')
    test_loader = build_dataloader(cfg.test_cfg, dataset)
    # print(f'test loader is built with config:\n{cfg.test_cfg}')

    print('test model')
    model.eval()
    test_total, test_correct = 0, 0
    for _, batch in enumerate(test_loader):
        features = batch[0].to(cfg.device)
        if len(batch[1].shape) > 1: # single label or multiple label
            labels = batch[1][:, cfg.test_label_sel].to(cfg.device)
        else:
            labels = batch[1].to(cfg.device)
        # labels = batch[1][:, cfg.test_label_sel].to(cfg.device)
        with torch.cuda.amp.autocast():
            _, logits = model(features)
        prec = accuracy(logits, labels) 
        test_total += 1
        test_correct += prec

    return (test_correct / test_total).item()       

def train_model(cfg, model, dataset, loss_func, test_dataset=None):
    # model: torch model
    # dataset: torch dataset
    # cfg: config. See ./utils/config.py
    def train_epoch(epoch):
        model.train()
        train_total, train_correct = 0, 0
        for i, batch in enumerate(train_loader):
            """ 
                batch should always be tuple. Must satisfy the following structure:
                (feature, label/attribute_tuple, index)
            """
            # (features, labels, indexes)
            batch_size = batch[0].shape[0]
            features = batch[0].to(cfg.device)
            if len(batch[1].shape) > 1: # single label or multiple label
                labels = batch[1][:, cfg.train_label_sel].to(cfg.device)
            else:
                labels = batch[1].to(cfg.device)
            with torch.cuda.amp.autocast():
                _, logits = model(features)
            prec = accuracy(logits, labels) 
            train_total += 1
            train_correct += prec
            loss = loss_func(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % cfg.print_freq == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                    %(epoch+1, cfg.n_epoch, i+1, len(dataset)//batch_size, prec, loss.data))
    
    accuracy = Accuracy(**cfg.accuracy)
    optimizer = set_optimizer(cfg.optimizer, model)


    print('build dataloader')
    train_loader = build_dataloader(cfg.train_cfg, dataset)
    print(f'training loader is built with config:\n{cfg.train_cfg}')

    print('train model')
    for epoch in range(cfg.n_epoch):
        train_epoch(epoch)
        if test_dataset is not None:
            test_acc = test_model(cfg, model, test_dataset)
            print(f'[Epoch {epoch+1}/{cfg.n_epoch}] test accuracy: {test_acc}')

    return model