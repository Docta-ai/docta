import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, feature_dim, hidsizes=[1024, 512, 32], outputs=1, dropout=0., activation='relu'):
        super(MLP, self).__init__()

        if activation == 'relu':
            self.ac_fn = torch.nn.ReLU
        elif activation == 'tanh':
            self.ac_fn = torch.nn.Tanh
        elif activation == 'sigmoid':
            self.ac_fn = torch.nn.Sigmoid
        elif activation == 'leaky':
            self.ac_fn = torch.nn.LeakyReLU
        elif activation == 'elu':
            self.ac_fn = torch.nn.ELU
        elif activation == 'relu6':
            self.ac_fn = torch.nn.ReLU6

        self.mlp = []
        hidsizes = [feature_dim] + hidsizes
        for i in range(1, len(hidsizes)):
            self.mlp.append(nn.Linear(hidsizes[i-1], hidsizes[i]))
            self.mlp.append(nn.Dropout(dropout))
            self.mlp.append(self.ac_fn())
        self.mlp = nn.Sequential(*self.mlp, nn.Linear(hidsizes[-1], outputs))
        # print(self.mlp)

    def forward(self, x):
        if x.dim() < 2:
            x = x.unsqueeze(0)
        return None, self.mlp(x).squeeze(-1)

    # @staticmethod
    # def layer_init(layer, w_scale=1.0):
    #     nn.init.orthogonal_(layer.weight.data)
    #     layer.weight.data.mul_(w_scale)
    #     nn.init.constant_(layer.bias.data, 0)
    #     return layer
