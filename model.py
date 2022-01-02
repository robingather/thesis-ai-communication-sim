import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import constants as C

class MLP(nn.Module):
    # multi-layer perceptron class
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        if C.SECOND_HIDDEN:
            self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)
        self.to(device=C.DEVICE)

    def forward(self, x):
        # feed-forward
        x = F.relu(self.linear1(x))
        if C.SECOND_HIDDEN:
            x = F.relu(self.linear2(x))
        x = self.head(x)
        return x

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        return self.get_weights(torch.load(file_name))

    def get_weights(self, state_dict=None):
        weights = []
        if state_dict==None:
            state_dict = self.state_dict()
        for genes in state_dict.values():
            weights.extend(genes.data.flatten().cpu().detach().numpy())
        #print(len(weights))
        return weights

    def set_weights(self, weights):
        nI, nH, nO = C.N_INPUTS.pred, C.N_HIDDEN, C.N_ACTIONS

        with torch.no_grad():
            l1_w = torch.tensor(weights[:nI*nH],dtype=torch.float32,device=C.DEVICE).view(nH,-1); cut = nI*nH
            l1_b = torch.tensor(weights[cut:cut+nH],dtype=torch.float32,device=C.DEVICE); cut += nH
            
            if C.SECOND_HIDDEN:
                l2_w = torch.tensor(weights[cut:cut+nH*nH],dtype=torch.float32,device=C.DEVICE).view(nH,-1); cut += nH*nH
                l2_b = torch.tensor(weights[cut:cut+nH],dtype=torch.float32,device=C.DEVICE); cut += nH

            h_w = torch.tensor(weights[cut:cut+nH*nO],dtype=torch.float32,device=C.DEVICE).view(nO,-1); cut += nH*nO
            h_b = torch.tensor(weights[cut:],dtype=torch.float32,device=C.DEVICE)

            self.linear1.weight = nn.Parameter(l1_w)
            self.linear1.bias = nn.Parameter(l1_b)
            if C.SECOND_HIDDEN:
                self.linear2.weight = nn.Parameter(l2_w)
                self.linear2.bias = nn.Parameter(l2_b)
            self.head.weight = nn.Parameter(h_w)
            self.head.bias = nn.Parameter(h_b)