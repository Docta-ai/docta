import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PeerLoss(nn.Module):
    def __init__(self, alpha_plan = None):
        super(PeerLoss,self).__init__()
        self.alpha_plan = alpha_plan

        
    def forward(self, epoch, inputs, targets, inputs_peer = None, targets_peer = None): # inputs are model logits (before softmax)
        alpha = 1.0 if self.alpha_plan is None else self.alpha_plan[epoch]
        if inputs_peer is None or targets_peer is None:
            idx_1, idx_2 = np.arange(inputs.shape[0]), np.arange(inputs.shape[0])
            np.random.shuffle(idx_1)
            np.random.shuffle(idx_2)
            inputs_peer, targets_peer = inputs[idx_1], inputs[idx_2]

        loss = F.cross_entropy(inputs, targets, reduce = False)
        loss_ = -torch.log(F.softmax(inputs_peer) + 1e-8)
        loss_peer = torch.gather(loss_, 1, targets_peer.view(-1,1)).view(-1)
        loss = loss - alpha * loss_peer

        return loss

class Cores(PeerLoss):
    def __init__(self, alpha_plan = None, noisy_prior = None):
        super(Cores,self).__init__(alpha_plan = alpha_plan)
        self.noisy_prior = noisy_prior

    def forward(self, epoch, inputs, targets): # inputs are model logits (before softmax)
        alpha = 1.0 if self.alpha_plan is None else self.alpha_plan[epoch]
        
        loss = F.cross_entropy(inputs, targets, reduce = False)
        loss_ = -torch.log(F.softmax(inputs) + 1e-8)
        if self.noisy_prior is None:
            loss =  loss - alpha*torch.mean(loss_, 1)
        else:
            loss =  loss - alpha*torch.sum(torch.mul(self.noisy_prior, loss_),1)

        return loss

