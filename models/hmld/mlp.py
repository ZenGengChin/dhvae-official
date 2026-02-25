import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.margpt.helpers import DropPath, drop_path





class AdaLnMLPBlock(nn.Module):
    def __init__(self, d_model:int=256, d_cond:int=768, mlp_ratio:int=4, dropout:float=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_cond = d_cond
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(d_cond, 2*d_model))
        
        self.layernorm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.linear1 = nn.Linear(d_model, (d_model * mlp_ratio))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear((d_model * mlp_ratio), d_model)
        
    
    def forward(self, x:Tensor, cond_B1D:Tensor):
        # x in shape of [B, L, E], cond_B1D in shape of [B, 1, D]
        x_orig = x
        scale, shift = self.ada_lin(cond_B1D).view(-1, cond_B1D.shape[1], 2, self.d_model).unbind(2)
        x = self.layernorm(x).mul(scale.add(1)).add_(shift)
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.linear2(x)
        return x + x_orig

class AdaLnMLP(nn.Module):
    def __init__(self, d_cond:int=768, d_model:int=256, mlp_ratio:int=4, nlayer:int=4):
        super().__init__()
        
        self.blocks = nn.Sequential(
            *[AdaLnMLPBlock(d_model=d_model, d_cond=d_cond, mlp_ratio=mlp_ratio) for i in range(nlayer)]
        )
            
    def forward(self, x:Tensor, cond_B1D:Tensor):
        '''x in shape of [B, L, E]'''
        for block in self.blocks:
            x = block(x, cond_B1D)
        return x
        
class SkipAdaLnMLP(nn.Module):
    def __init__(self, d_cond:int=768, d_model:int=256, mlp_ratio:int=4, nlayer:int=4):
        super().__init__()
        
        self.nlayer = nlayer if nlayer % 2 == 1 else nlayer + 1
        self.nlayer_in = (self.nlayer - 1) // 2
        
        self.in_layers = nn.ModuleList([
            AdaLnMLPBlock(d_model=d_model, 
                          d_cond=d_cond, 
                          mlp_ratio=mlp_ratio,
                          dropout=0.1)
            for i in range(self.nlayer_in)
        ])
        
        self.mid_layer = AdaLnMLPBlock(
            d_model=d_model,
            d_cond=d_cond,
            mlp_ratio=mlp_ratio,
            dropout=0.1)
        
        
        self.out_layers = nn.ModuleList([
            AdaLnMLPBlock(d_model=d_model, 
                          d_cond=d_cond, 
                          mlp_ratio=mlp_ratio,
                          dropout=0.1)
            for i in range(self.nlayer_in)
        ])
        
        self.linear_layers = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(self.nlayer_in)])
        
        
    def forward(self, x:Tensor, cond_B1D:Tensor):
        xs = []
        for module in self.in_layers:
            module: AdaLnMLPBlock
            x = module.forward(x, cond_B1D)
            xs.append(x)
        x = self.mid_layer(x, cond_B1D)
        for (module, linear) in zip(self.out_layers, self.linear_layers):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module.forward(x, cond_B1D)
        return x
        