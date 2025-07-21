import math 
import torch 
import torch.nn as nn
import torch.nn.functional as F



class LoRABase:

    def __init__(self, rank = 8, lora_alpha = 8, lora_dropout = 0):
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.scaling = self.lora_alpha / self.rank**(1/2)



