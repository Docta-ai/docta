import torch
import torch.nn as nn
import torch.nn.functional as F

class ForwardLoss(nn.Module):
    def __init__(self, T):
        super(ForwardLoss, self).__init__()
        self.T = T # T should be in the same device as inputs, e.g., both in CPU or both in CUDA

        
    def forward(self, inputs, targets): # inputs are model logits (before softmax)
        # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)
        outputs = F.softmax(inputs, dim=1)
        outputs = outputs @ self.T
        outputs = torch.log(outputs)
        loss = F.nll_loss(outputs, targets) # targets must be catagorical classes. 
        return loss


class BackwardLoss(ForwardLoss):
    def __init__(self, T):
        super(BackwardLoss, self).__init__(T = T)

        
    def forward(self, inputs, targets): # inputs are model logits (before softmax)
        trans_mat_inv = torch.inverse(self.T)
        outputs = F.softmax(inputs, dim=1)
        outputs = torch.log(outputs)
        loss = -torch.mean(torch.sum( (F.one_hot(targets, self.T.shape[0]).float() @ trans_mat_inv) * outputs, axis=1 ), axis = 0)
        return loss

