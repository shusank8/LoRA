import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Union
import math


class LoraBase:
    def __init__(self, rank = 8, alpha = 8, dropout = 0):
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.scaling = self.alpha / self.rank **(1/2)
    
    def load_pretrained_weights(self, state_dict):
        self.weight.data = state_dict['weight']
        if "bias" in state_dict.keys():
            self.weight.bias = state_dict['bias']


class LoraLinear(nn.Linear, LoraBase):
    def __init__(self, in_features:int, out_features:int, rank:int, alpha:int, dropout:int = 0, bias:bool=True):
        nn.Linear.__init__(self, in_features, out_features, bias)
        LoraBase.__init__(self, rank, alpha, dropout)
        self.weight.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def merge_weight(self):
        m_w = self.weight.data + self.scaling * (self.lora_A @ self.lora_B).T
        state_dict = {"weight":m_w}
        if self.bias is not None:
            state_dict['bias'] = self.bias
        merged_linear = nn.Linear(
            in_features=self.in_features, out_features = self.out_features
        )
        merged_linear.load_state_dict(state_dict)
        return merged_linear

    def forward(self, x):
        # linear layer as it is:
        linear_output = x @ self.weight.T
        if self.bias is not None:
            linear_output+=self.bias
        
        # (IN, RANK) @ (RANK, OUT)=> (IN, OUT)
        lora_mult = (self.lora_A @ self.lora_B) * self.scaling

        # LORA output:
        lora_out = self.dropout(x) @ lora_mult

        return linear_output + lora_out



test_model = LoraLinear(4, 8, 2, 2)
x = torch.randn(2, 4)
o1 = test_model(x)

ln = test_model.merge_weight()
o2 = ln(x)

print(o1==o2)
