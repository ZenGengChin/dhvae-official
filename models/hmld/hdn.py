# Copyright (c) UWA CSSE. and its affiliates. All Rights Reserved
"""
Implementations of Denoiser of HMLD
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from models.hmld.net import ParallelTransformer
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer


class ParallelDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.latent_dim
        self.nhead = cfg.nhead
        self.d_ffn = cfg.dim_feedforward
        self.nlayer_self = cfg.nlayer_self
        self.nlayer_cross = cfg.nlayer_cross
        self.dropout = cfg.dropout
        self.batch_first = True
        
        self.denoiser = ParallelTransformer(
            d_model=self.d_model,
            nhead=self.nhead,
            d_ffn=self.d_ffn,
            nselflayer=self.nlayer_self,
            ncrosslayer=self.nlayer_cross,
            dropout=self.dropout,
            batch_first=self.batch_first,
            concat='trunc'
        )
        
    def forward(self, x, cond_BLD, attn_mask: Tensor = None):
        """
        * Currently only support one-shot inference. AR is not supported.
        Args:
            x (Tensor): [B, latent_size, latent_dim]
            cond_B1D (Tensor): [B, cond_size(2 basicly), cond_dim]
            attn_mask (Tensor, optional): [B, latent_size]
        Returns:
            Tensor: [B, latent_size, latent_dim]
        """
        return self.denoiser.forward(left=x, right=cond_BLD, left_attn_mask=attn_mask)




from models.hmld.adaln import AdaLNSelfAttn, AdaLNBeforeHead, AdaLNTransformer, SkipAdaLNTransformer

class AdaLNDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.d_model = cfg.latent_dim
        self.nhead = cfg.nhead
        self.d_ffn = cfg.dim_feedforward
        self.nlayer = cfg.nlayer
        self.dropout = cfg.dropout
        self.skip = cfg.is_skip if hasattr(cfg, 'is_skip') else False
        
        if self.skip:
            self.denoiser = SkipAdaLNTransformer(
                d_model=self.d_model,
                d_cond=cfg.text_encoder.d_cond,
                nhead=self.nhead,
                nlayer=self.nlayer,
                d_ffn=self.d_ffn,
                dropout=self.dropout,
                shared_aln=False
            )
        else:
            self.denoiser = AdaLNTransformer(
                d_model=self.d_model,
                d_cond=cfg.text_encoder.d_cond,
                nhead=self.nhead,
                nlayer=self.nlayer,
                d_ffn=self.d_ffn,
                dropout=self.dropout,
                shared_aln=False
            )
        
    def forward(self, x, cond_B1D, attn_mask: Tensor = None):
        """
        Args:
            x (Tensor): [B, latent_size, latent_dim]
            cond_B1D (Tensor): [B, 1, cond_dim]
            attn_mask (Tensor, optional): [B, latent_size]
        Returns:
            Tensor: [B, latent_size, latent_dim]
        """
        return self.denoiser.forward(x, cond_B1D, attn_mask)
        

import clip
from models.hmld.textenc import TextEncoder
from models.mld.embeddings import PositionalEncoding, TimestepEmbedding, Timesteps
from models.utils.tools import LearnablePositionalEncoding


class HDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg # cfg = cfg.hmld
        
        self.d_model = cfg.latent_dim
        self.nhead = cfg.nhead
        self.d_ffn = cfg.dim_feedforward
        self.nlayer = cfg.nlayer_self + cfg.nlayer_cross
        self.d_cond = cfg.text_encoder.d_cond
        
        # for ablation
        self.is_skip = cfg.is_skip
        self.is_intergen_clip = cfg.text_encoder.is_intergen_clip
        self.is_lpe = cfg.is_lpe
        
        if self.cfg.arch == 'adaln':
            self.denoiser = AdaLNDenoiser(cfg)
        elif self.cfg.arch == 'parallel':
            self.denoiser = ParallelDenoiser(cfg)
        else:
            raise ValueError(f"Invalid architecture: {self.cfg.arch}")
        
        # text encoder
        self.text_proj = nn.Linear(self.d_cond, self.d_model)        
        # timestep embedding
        self.time_proj = Timesteps(self.d_cond, True,0)
        self.time_embedding = TimestepEmbedding(self.d_cond,
                                                self.d_cond)
        
        
        # PE
        if self.is_lpe:
            self.pe = LearnablePositionalEncoding(self.d_model)
        else:
            self.pe = PositionalEncoding(self.d_model, max_seq_len=1000)
        
        
    def encode_cond(self, time_steps:Tensor, cond_B1D:Tensor)->Tensor:
        """
        Args:
            text (list[str]): [B]
            timesteps (Tensor): [B] or [1]
        Returns:
            time_emb: [B, 1, d_model]
        """
        B = cond_B1D.shape[0]
        time_emb = self.time_proj(time_steps) # BC
        time_emb = self.time_embedding(time_emb).unsqueeze(1) # B1C
        if time_steps.shape[0] != B:
            time_emb = time_emb.expand(B, -1, -1)
        return time_emb
        
        
            
    def forward_adaln_noar(self, x:torch.Tensor, cond_B1D:Tensor, timesteps:Tensor):
        """
        Args:
            x (Tensor): [B, latent_size, latent_dim]
            cond_B1D (Tensor): [B, 1, d_cond]
        Returns:
            Tensor: [B, latent_size, latent_dim]
        """
        time_emb = self.encode_cond(timesteps, cond_B1D)
        # cond_B1C = self.text_proj(cond_B1D)
        # cond_BLC = torch.cat([time_emb, cond_B1C], dim=1)
        cond_B1D = cond_B1D + time_emb
        self.denoiser:AdaLNTransformer
        # x = self.denoiser.forward(x=self.pe(torch.cat([cond_B1D, x], dim=1)),
        #                           cond_B1D=cond_B1D)
        x = self.denoiser.forward(x=self.pe(x), cond_B1D=cond_B1D)
        return x[:,:,:]
    
    
    def get_ar_mask(self) -> torch.Tensor:
        """
        Generate an autoregressive mask for a sequence of given length and latent size.
        
        Args:
            latent_size (int): The size of the latent dimension.
            seq_len (int): The length of the sequence to be masked.
        
        Returns:
            torch.Tensor: A mask tensor of shape [seq_len, seq_len] with True for masked positions.
        """
        seq_len = 3 
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        ar_mask = mask.repeat_interleave(self.cfg.latent_size, dim=0).repeat_interleave(self.cfg.latent_size, dim=1)
        return ar_mask
    
    def forward_adaln_ar(self, x:torch.Tensor, cond_B1D:Tensor, timesteps:Tensor):
        """
        Args:
            x (Tensor): [B, latent_size, latent_dim]
            cond_B1D (Tensor): [B, 1, d_cond]
        Returns:
            Tensor: [B, latent_size, latent_dim]
        """
        time_emb = self.encode_cond(timesteps, cond_B1D)
        cond_B1C = self.text_proj(cond_B1D)
        cond_B1C = cond_B1C + time_emb
        x = self.denoiser.forward(x=x, cond_B1D=cond_B1C)
        return x
    
    def forward_parallel_noar(self, x, cond_B1D:Tensor, timesteps:Tensor):
        """
        Args:
            x (Tensor): [B, latent_size, latent_dim]
            cond_B1D (Tensor): [B, 1, d_cond]
        Returns:
            Tensor: [B, latent_size, latent_dim]
        """
        self.denoiser:ParallelDenoiser
        # time_emb, cond_B1C, cond_B1D = self.encode_cond(text, timesteps)
        time_emb = self.encode_cond(timesteps, cond_B1D)
        cond_B1C = self.text_proj(cond_B1D)
        cond_BLC = torch.cat([time_emb, cond_B1C], dim=1)
        x = self.denoiser.forward(x=self.pe(x), 
                                  cond_BLD=self.pe(cond_BLC))
        return x
    
    def forward_parallel_ar(self, x, cond_B1D:Tensor, timesteps:Tensor):
        """
        Args:
            x (Tensor): [B, latent_size, latent_dim]
            cond_B1D (Tensor): [B, 1, d_cond]
        Returns:
            Tensor: [B, latent_size, latent_dim]
        """
        self.denoiser:ParallelDenoiser
        time_emb = self.encode_cond(timesteps, cond_B1D)
        cond_B1C = self.text_proj(cond_B1D)
        cond_BLC = torch.cat([time_emb, cond_B1C], dim=1)
        x = self.denoiser.forward(x=self.pe(x), 
                                  cond_BLD=self.pe(cond_BLC))
        return x
    
    
    def forward(self, x, cond_B1D:Tensor, timesteps:Tensor):
        if self.cfg.arch == 'adaln' and not self.cfg.is_ar:
            return self.forward_adaln_noar(x, cond_B1D, timesteps)
        elif self.cfg.arch == 'parallel' and not self.cfg.is_ar:
            return self.forward_parallel_noar(x, cond_B1D, timesteps)
        elif self.cfg.arch == 'adaln' and self.cfg.is_ar:
            return self.forward_adaln_ar(x, cond_B1D, timesteps)
        elif self.cfg.arch == 'parallel' and self.cfg.is_ar:
            return self.forward_parallel_ar(x, cond_B1D, timesteps)
        else:
            raise ValueError(f"Invalid architecture: {self.cfg.arch}")
    

from models.hmld.mlp import AdaLnMLP, SkipAdaLnMLP       


class MLPBlock(nn.Module):
    def __init__(self, d_input:int=256, d_model:int=256, d_ffn:int=1024):
        super().__init__()
        
        self.layernorm = nn.LayerNorm(d_model + d_input)
        self.linear1 = nn.Linear(d_model + d_input, d_ffn)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ffn, d_model + d_input)
        
        self.block = nn.Sequential(
            self.layernorm, self.linear1, self.activation, self.linear2
        )
    
    def forward(self, x:Tensor):
        x_orig = x
        x = self.block.forward(x)
        return x + x_orig

class MLPDenoiser(nn.Module):
    def __init__(self, d_input:int=256, d_model:int=256, d_ffn:int=1024, nlayer:int=3):
        super().__init__()
        
        self.blocks = nn.Sequential(
            *[MLPBlock(d_input=d_input, d_model=d_model, d_ffn=d_ffn) for i in range(nlayer)]
        )
        
        self.out_proj = nn.Linear(d_input + d_model, d_input)
        
    
    def forward(self, x:Tensor, cond_B1D:Tensor):
        '''x in shape of [B, L, E], cond_B1D in shape of [B, L, D]'''
        x = torch.cat([x, cond_B1D], dim=-1)
        return self.out_proj(self.blocks.forward(x))
                                                                                                     

class ARDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.d_model = cfg.latent_dim 
        self.d_cond = cfg.text_encoder.d_cond
        self.nhead = cfg.nhead
        self.d_ffn = cfg.dim_feedforward
        self.nlayer_ar = cfg.nlayer_ar
        self.nlayer_mlp = cfg.nlayer_mlp
        self.dropout = cfg.dropout
        self.skip = cfg.is_skip if hasattr(cfg, 'is_skip') else False
        self.skip_mlp = cfg.skip_mlp if hasattr(cfg, 'skip_mlp') else False
        self.multi_mlp = cfg.multi_mlp if hasattr(cfg, 'multi_mlp') else False
        self.mlp_ratio = cfg.mlp_ratio
        self.is_lpe = cfg.is_lpe
        
        if self.multi_mlp:
            self.mlp: nn.Module = MLPDenoiser(d_input=self.d_model, d_model=self.d_model, d_ffn=self.d_ffn, nlayer=3)
            print('use multi mlp========================================================')
        else:
            self.mlp: nn.Module = AdaLnMLP(
                d_cond=self.d_model,
                d_model=self.d_model,
                mlp_ratio=self.mlp_ratio,
                nlayer=self.nlayer_mlp
            ) if not self.skip_mlp else SkipAdaLnMLP(
                d_cond=self.d_model,
                d_model=self.d_model,
                mlp_ratio=self.mlp_ratio,
                nlayer=self.nlayer_mlp
            )
        
        self.arnet = SkipAdaLNTransformer(
            d_model=self.d_model,
            nhead=self.nhead,
            nlayer=self.nlayer_ar,
            d_ffn=self.d_ffn,
            d_cond=self.d_cond,
            dropout=self.dropout
        )
        
        self.text_proj = nn.Linear(self.d_cond, self.d_model)        
        # timestep embedding
        self.time_proj = Timesteps(self.d_cond, True,0)
        self.time_embedding = TimestepEmbedding(self.d_cond,
                                                self.d_cond)
        
        self.diff_time_proj = Timesteps(self.d_model, True,0)
        self.diff_time_embedding = TimestepEmbedding(self.d_model,
                                                self.d_model)
        
        
        self.start_token = nn.Parameter(torch.randn(1, 1, self.d_model))
                
        # PE
        if self.is_lpe:
            self.pe = LearnablePositionalEncoding(self.d_model)
        else:
            self.pe = PositionalEncoding(self.d_model, max_seq_len=1000)
        
        

        
        
    def forward(self, x, cond_B1D, timesteps, mask=None):
        encoded_context = self.encode_context(x, cond_B1D, mask)
        x = self.denoise(x, timesteps, encoded_context)
        return x
        
    def encode_context(self, x0: torch.Tensor, cond_B1D:Tensor, mask:torch.Tensor=None)->Tensor:
        """
        Args:
            x (Tensor): [B, latent_size, d_model] is the latent represnetation with no [s]
            cond_B1D (Tensor): [B, 1, d_cond], is the text condition form CLIP
        Returns:
            cond_token: [B, 1, d_model]
        """
        start_token = self.start_token.expand(cond_B1D.shape[0], -1, -1)
        if x0 is not None:
            x0 = torch.cat([start_token, x0], dim=1) # [B, latent_size + 1, d_model]
        else:
            x0 = start_token
        x = self.arnet.forward(x=x0, cond=cond_B1D, src_mask=mask)
        return x
    
    def denoise(self, x:torch.Tensor, timesteps:Tensor, cond_B1C:Tensor):
        B = cond_B1C.shape[0]
        time_emb = self.diff_time_proj(timesteps) # BC
        if timesteps.shape[0] != B:
            time_emb = time_emb.expand(B, -1)
        time_emb = self.diff_time_embedding(time_emb).unsqueeze(1) # B1C
        # print(time_emb.shape, cond_B1C.shape, '-------------------------')
        cond_B1C = cond_B1C + time_emb
        x = self.mlp.forward(x=x, cond_B1D=cond_B1C)
        return x

                


'''
if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('cfg/hmld/hmld.yaml')
    hdn = HDenoiser(cfg.mld).cuda()
    B = 16
    x = torch.randn(B, 6, 256).cuda()
    text = ['a cat'] * B
    timesteps = torch.randint(0, 1000, (B,)).cuda()
    x = hdn.forward_adaln_noar(x, text, timesteps)
    print(x.shape)
'''
if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('cfg/hmld/hard.yaml')
    hdn = ARDenoiser(cfg.mld).cuda()
    B = 16
    L = 3
    x = torch.randn(B, L, 256).cuda()
    text = ['a cat'] * B
    text_cond = torch.randn(B, 1, 768).cuda()
    timesteps = torch.randint(0, 1000, (B,)).cuda()
    mask = torch.triu(torch.ones(L+ 8, L+8, dtype=torch.bool), diagonal=1).cuda()
    mask = mask[:L,:8].reshape(1, 1, L, 8)
    mask = torch.where(mask, 0, -torch.inf)
    context = hdn.encode_context(x[:, :-1, :], text_cond, mask[:,:,:,:L])
    x = hdn.denoise(x, timesteps, context)
    print(x.shape)