import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from models.hmld.net import CoTransformerLayer, SkipCoTransformer
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from positional_encodings.torch_encodings import PositionalEncoding1D
from models.utils.tools import LearnablePositionalEncoding
from lightning import LightningModule
from models.utils.sparse_loss import weighted_l1_loss_1d




class HEncoder(nn.Module):
    def __init__(self, cfg):
        super(HEncoder, self).__init__()
        self.cfg = cfg
        
        self.d_model = cfg.latent_dim
        self.nhead = cfg.nhead
        self.d_ffn = cfg.dim_feedforward
        self.nlayer_ind = cfg.nlayer_ind
        self.nlayer_top = cfg.nlayer_top
        self.dropout = cfg.dropout
        self.contact = cfg.contact
        self.latent_size = cfg.latent_size
        self.d_joint = cfg.d_joint
        self.d_input = cfg.d_input
        self.max_len = cfg.max_len
        
        self.is_learn_pe = cfg.is_learn_pe
        self.is_skip = cfg.is_skip if hasattr(cfg, 'is_skip') else False
        self.is_two_pe = cfg.is_two_pe if hasattr(cfg, 'is_two_pe') else False        
        if self.is_learn_pe:
            if self.is_two_pe:
                self.pe_a = LearnablePositionalEncoding(self.d_model)
                self.pe_b = LearnablePositionalEncoding(self.d_model)
            else:
                self.pe = LearnablePositionalEncoding(self.d_model)
        else:
            if self.is_two_pe:
                self.pe_a = PositionalEncoding1D(self.d_model)
                self.pe_b = PositionalEncoding1D(self.d_model)
            else:
                self.pe = PositionalEncoding1D(self.d_model)
                
        self.encoder_left = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.d_ffn,
                dropout=self.dropout,
                batch_first=True,
            ),
            num_layers=self.nlayer_ind
        )
        
        self.encoder_right = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.d_ffn,
                dropout=self.dropout,
                batch_first=True,
            ),
            num_layers=self.nlayer_ind
        )
        
        if self.is_skip:
            from models.hmld.net import SkipTransformerEncoder, TransformerEncoderLayer
            nlayer = self.nlayer_ind
            if nlayer % 2 == 0:
                nlayer -= 1
                
            self.encoder_left = SkipTransformerEncoder(
                encoder_layer=TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.d_ffn,
                    dropout=self.dropout,
                    batch_first=True,
                ),
                num_layers=nlayer
            )
            self.encoder_right = SkipTransformerEncoder(
                encoder_layer=TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.d_ffn,
                    dropout=self.dropout,
                    batch_first=True,
                ),
                num_layers=nlayer
            )
        
        self.encoder_top = SkipCoTransformer(
            d_model=self.d_model,
            nhead=self.nhead,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
            nlayer=self.nlayer_top,
            batch_first=True,
            activation='relu',
            device='cuda'
        )
        self.contact_layer = nn.Linear(2*self.d_model, self.d_model)
        self.u1 = nn.Parameter(torch.randn(self.latent_size * 2, self.d_model))
        self.u2 = nn.Parameter(torch.randn(self.latent_size * 2, self.d_model))
        self.u3 = nn.Parameter(torch.randn(self.latent_size * 2, self.d_model))
        
        self.input_proj1 = nn.Linear(self.d_input, self.d_model)
        self.input_proj2 = nn.Linear(self.d_input, self.d_model)
    
    def get_ind_mask(self, length:torch.Tensor):
        mask = torch.arange(self.max_len).unsqueeze(0).to(length.device) >= length.unsqueeze(1)
        mask = torch.cat([torch.zeros(length.shape[0], self.latent_size * 2).bool().to(length.device), 
                          mask],
                         dim=1)
        return mask.bool()
    
    def get_top_mask(self, length:torch.Tensor):
        mask = torch.arange(self.max_len).unsqueeze(0).to(length.device) >= length.unsqueeze(1)
        mask = torch.cat([torch.zeros(length.shape[0], self.latent_size * 4).bool().to(length.device), 
                          mask],
                         dim=1)
        return mask.bool()
    
        
    def forward(self, motion, length):
        x1, x2 = torch.split(motion, [self.d_input, self.d_input], dim=-1)
        
        B, L, D = x1.shape
        
        mask = self.get_ind_mask(length)
        
        x1 = torch.cat([self.u1.repeat(motion.shape[0], 1, 1), self.input_proj1(x1)], dim=1)
        x2 = torch.cat([self.u2.repeat(motion.shape[0], 1, 1), self.input_proj2(x2)], dim=1)
        
        if not self.is_two_pe:
            x1 = self.encoder_left(self.pe(x1), src_key_padding_mask=mask)
            x2 = self.encoder_right(self.pe(x2), src_key_padding_mask=mask)
        else:
            x1 = self.encoder_left(self.pe_a(x1), src_key_padding_mask=mask)
            x2 = self.encoder_right(self.pe_b(x2), src_key_padding_mask=mask)
        
        z1, z2 = x1[:,:self.latent_size*2,:], x2[:,:self.latent_size*2,:]
            
        top_mask = self.get_top_mask(length)
        x1, x2 = self.encoder_top.forward(left=torch.cat([self.u3.repeat(B, 1, 1), x1], dim=1), 
                                          right=torch.cat([self.u3.repeat(B, 1, 1), x2], dim=1), 
                                          left_mask=top_mask, right_mask=top_mask)
        
        x3 = torch.cat([x1, x2], dim=-1)
        z3 = self.contact_layer(x3)[:,:self.latent_size*2,:]
        
        return z1, z2, z3
        

class HDecoder(nn.Module):
    def __init__(self, cfg):
        super(HDecoder, self).__init__()
        self.cfg = cfg
        
        self.d_model = cfg.latent_dim
        self.nhead = cfg.nhead
        self.d_ffn = cfg.dim_feedforward
        self.nlayer_ind = cfg.nlayer_ind
        self.nlayer_top = cfg.nlayer_top
        self.dropout = cfg.dropout
        self.contact = cfg.contact
        self.latent_size = cfg.latent_size
        self.d_joint = cfg.d_joint
        self.d_input = cfg.d_input
        self.max_len = cfg.max_len
        
        self.is_skip = cfg.is_skip if hasattr(cfg, 'is_skip') else False
        self.is_learn_pe = cfg.is_learn_pe
        self.is_two_pe = cfg.is_two_pe if hasattr(cfg, 'is_two_pe') else False
        
        
        if self.is_learn_pe:
            if self.is_two_pe:
                self.pe_a = LearnablePositionalEncoding(self.d_model)
                self.pe_b = LearnablePositionalEncoding(self.d_model)
            else:
                self.pe = LearnablePositionalEncoding(self.d_model)
        else:
            if self.is_two_pe:
                self.pe_a = PositionalEncoding1D(self.d_model)
                self.pe_b = PositionalEncoding1D(self.d_model)
            else:
                self.pe = PositionalEncoding1D(self.d_model)
        
        self.is_decode_cat = cfg.is_decode_cat # if true, [z3, z1]; [z3, z2] else:
        
        
        self.decoder_left = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.d_ffn,
                dropout=self.dropout,
                batch_first=True,
            ),
            num_layers=self.nlayer_ind if not self.is_decode_cat else self.nlayer_top + self.nlayer_ind
        )
        
        self.decoder_right = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.d_ffn,
                dropout=self.dropout,
                batch_first=True,
            ),
            num_layers=self.nlayer_ind if not self.is_decode_cat else self.nlayer_top + self.nlayer_ind
        )
        
        if self.is_skip:
            nlayer = self.nlayer_ind if not self.is_decode_cat else self.nlayer_top + self.nlayer_ind
            if nlayer % 2 == 0:
                nlayer -= 1
                
            from models.hmld.net import SkipTransformerDecoder, TransformerDecoderLayer
            self.decoder_left = SkipTransformerDecoder(
                decoder_layer=TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.d_ffn,
                    dropout=self.dropout,
                    batch_first=True,
                ),
                num_layers=nlayer
            )
            
            self.decoder_right = SkipTransformerDecoder(
                decoder_layer=TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.d_ffn,
                    dropout=self.dropout,
                    batch_first=True,
                ),
                num_layers=nlayer
            )
        
        if not self.is_decode_cat:
            self.top_decoder = SkipCoTransformer(
                d_model=self.d_model,
                nhead=self.nhead,
                d_ffn=self.d_ffn,
                dropout=self.dropout,
                nlayer=self.nlayer_top,
                batch_first=True,
                activation='relu',
                device='cuda'
            )
            
        self.output_proj1 = nn.Linear(self.d_model, self.d_input)
        self.output_proj2 = nn.Linear(self.d_model, self.d_input)
        
    def get_length_mask(self, length:torch.Tensor):
        mask = torch.arange(self.max_len).unsqueeze(0).to(length.device) >= length.unsqueeze(1)
        return mask.bool()
    
    def get_top_mask(self, length:torch.Tensor):
        mask = torch.arange(self.max_len).unsqueeze(0).to(length.device) >= length.unsqueeze(1)
        mask = torch.cat([torch.zeros(length.shape[0], self.latent_size).bool().to(length.device), 
                          mask],
                         dim=1)
        return mask.bool()
    
        
    def forward_cat(self, z1:Tensor, z2:Tensor, z3:Tensor, length:Tensor):
        mask = self.get_length_mask(length)
        
        if self.is_two_pe:
            x1 = self.decoder_left(tgt = self.pe_a(torch.zeros(z1.shape[0], self.max_len, self.d_model).to(z1.device)),
                                memory = torch.cat([z3, z1], dim=1),
                                    tgt_key_padding_mask=mask)
            x2 = self.decoder_right(tgt = self.pe_b(torch.zeros(z2.shape[0], self.max_len, self.d_model).to(z2.device)),
                                    memory = torch.cat([z3, z2], dim=1),
                                    tgt_key_padding_mask=mask)
        else:
            x1 = self.decoder_left(tgt = self.pe(torch.zeros(z1.shape[0], self.max_len, self.d_model).to(z1.device)),
                                memory = torch.cat([z3, z1], dim=1),
                                tgt_key_padding_mask=mask)
            x2 = self.decoder_right(tgt = self.pe(torch.zeros(z2.shape[0], self.max_len, self.d_model).to(z2.device)),
                                    memory = torch.cat([z3, z2], dim=1),
                                    tgt_key_padding_mask=mask)
        x1 = self.output_proj1(x1)
        x2 = self.output_proj2(x2)
        
        x1[mask] = 0
        x2[mask] = 0
        
        return x1, x2
        
        
    def forward_hire(self, z1:Tensor, z2:Tensor, z3:Tensor, length:Tensor):
        top_mask = self.get_top_mask(length)
        mask = self.get_length_mask(length)
                
        query = torch.zeros(z1.shape[0], self.max_len, self.d_model).to(z1.device)
        
        if self.is_two_pe:
            x3, x4 = self.top_decoder(left=self.pe_a(torch.cat([z3, query], dim=1)), 
                                    right=self.pe_b(torch.cat([z3, query], dim=1)), 
                                    left_mask=top_mask, right_mask=top_mask)
        else:
            x3, x4 = self.top_decoder(left=self.pe(torch.cat([z3, query], dim=1)), 
                                    right=self.pe(torch.cat([z3, query], dim=1)), 
                                    left_mask=top_mask, right_mask=top_mask)
        
        
        
        x1 = self.decoder_left.forward(tgt = x3[:,self.latent_size:], memory = torch.cat([z3, z1], dim=1), tgt_key_padding_mask=mask)
        x2 = self.decoder_right.forward(tgt = x4[:,self.latent_size:], memory = torch.cat([z3, z2], dim=1), tgt_key_padding_mask=mask)
        
        x1 = self.output_proj1(x1)
        x2 = self.output_proj2(x2)
        
        x1[mask] = 0
        x2[mask] = 0
        
        return x1, x2
    
    
    def forward(self, z1:Tensor, z2:Tensor, z3:Tensor, length:Tensor):
        if self.is_decode_cat:
            return self.forward_cat(z1, z2, z3, length)
        else:
            return self.forward_hire(z1, z2, z3, length)
    
    
from omegaconf import OmegaConf
        
class CHVAE(LightningModule):
    def __init__(self, cfg):
        super(CHVAE, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg))
        self.name = cfg.vae.name
        self.encoder = HEncoder(cfg.vae)
        self.decoder = HDecoder(cfg.vae)
        self.d_joint = cfg.vae.d_joint
        self.d_input = cfg.vae.d_input
        self.latent_size = cfg.vae.latent_size
        self.margin = cfg.vae.margin
        self.do_norm = cfg.vae.get('do_norm', False)
    def forward(self, motion:torch.Tensor, length:torch.Tensor):
        z1, z2, z3 = self.encoder.forward(motion, length)
        z1_sample = self._sample(z1[:, :self.latent_size], z1[:, self.latent_size:])
        z2_sample = self._sample(z2[:, :self.latent_size], z2[:, self.latent_size:])
        z3_sample = self._sample(z3[:, :self.latent_size], z3[:, self.latent_size:])
        x1, x2 = self.decoder.forward(z1_sample, z2_sample, z3_sample, length)
        return x1, x2
    
    def forward_test(self, batch):
        from utils.utils import MotionNormalizerTorch
        normalizer = MotionNormalizerTorch()
        device = next(self.parameters()).device
        motion, length = batch['motions'], batch['motion_lens']
        motion = torch.cat([normalizer.forward(motion[:,:,0]), normalizer.forward(motion[:,:,1])], dim=-1).to(device).float()
        length = length.to(device)
        z1, z2, z3 = self.encoder.forward(motion, length)
        z1_sample = self._sample(z1[:, :self.latent_size], z1[:, self.latent_size:])
        z2_sample = self._sample(z2[:, :self.latent_size], z2[:, self.latent_size:])
        z3_sample = self._sample(z3[:, :self.latent_size], z3[:, self.latent_size:])
        x1, x2 = self.decoder.forward(z1_sample, z2_sample, z3_sample, length)
        return {'output': torch.cat([x1, x2], dim=-1)}
    
    def _sample(self, mu:torch.Tensor, logvar:torch.Tensor):
        latent_dist = torch.distributions.Normal(mu, logvar.exp().pow(0.5))
        latent_z = latent_dist.rsample()
        return latent_z
    

    def _kl_div(self, mu:torch.Tensor, logvar:torch.Tensor):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def triplet_loss(self, anchor: Tensor, positive:Tensor, negative: Tensor):
        positive_distance = F.pairwise_distance(anchor, positive, p=2)
        negative_distance = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.relu(positive_distance - negative_distance + self.margin)
        return loss.mean()
    
    
    def compute_loss(self, 
                     motion:torch.Tensor, 
                     pos_motion:torch.Tensor,
                     neg_motion:torch.Tensor,
                     length:torch.Tensor):
        z1, z2, z3 = self.encoder.forward(motion, length)
        z1_sample = self._sample(z1[:, :self.latent_size], z1[:, self.latent_size:])
        z2_sample = self._sample(z2[:, :self.latent_size], z2[:, self.latent_size:])
        z3_sample = self._sample(z3[:, :self.latent_size], z3[:, self.latent_size:])
        
        z3_sample_pos = self.encoder.forward(pos_motion, length)[2][:, :self.latent_size]
        z3_sample_neg = self.encoder.forward(neg_motion, length)[2][:, :self.latent_size]
        
        loss_contrast = self.triplet_loss(z3[:,:self.latent_size], z3_sample_pos, z3_sample_neg)
        
        
        x1, x2 = self.decoder.forward(z1_sample, z2_sample, z3_sample, length)
        rec_loss = weighted_l1_loss_1d(x1, motion[...,:self.d_input], length) + weighted_l1_loss_1d(x2, motion[...,self.d_input:], length)
        joint_loss = weighted_l1_loss_1d(x1[:,:,:self.d_joint], motion[:,:,:self.d_joint], length) + \
                     weighted_l1_loss_1d(x2[:,:,:self.d_joint], motion[:,:,self.d_input:self.d_input + self.d_joint], length)
        kl_loss = self._kl_div(z1[:, :self.latent_size], z1[:, self.latent_size:]) + \
                  self._kl_div(z2[:, :self.latent_size], z2[:, self.latent_size:]) + \
                  self._kl_div(z3[:, :self.latent_size], z3[:, self.latent_size:])
        return rec_loss * 0.5, joint_loss * 0.5, kl_loss / 3, loss_contrast
    
    
    def encode(self, motion:torch.Tensor, length:torch.Tensor):
        z1, z2, z3 = self.encoder.forward(motion, length)
        z1_sample = self._sample(z1[:, :self.latent_size], z1[:, self.latent_size:])
        z2_sample = self._sample(z2[:, :self.latent_size], z2[:, self.latent_size:])
        z3_sample = self._sample(z3[:, :self.latent_size], z3[:, self.latent_size:])
        return z1_sample, z2_sample, z3_sample
    
    def training_step(self, batch):
        motion, pos_motion, neg_motion, length = batch['motions'], batch['pos_motions'], batch['neg_motions'], batch['motion_lens']
        rec_loss, joint_loss, kl_loss, loss_contrast = self.compute_loss(motion, pos_motion, neg_motion, length)
        loss = rec_loss + joint_loss + kl_loss * self.cfg.vae.beta_kl + loss_contrast * self.cfg.vae.beta_contrast
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/rec_loss', rec_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/joint_loss', joint_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/kl_loss', kl_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/contrast_loss', loss_contrast, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    
    def validation_step(self, batch):
        motion, pos_motion, neg_motion, length = batch['motions'], batch['pos_motions'], batch['neg_motions'], batch['motion_lens']
        rec_loss, joint_loss, kl_loss, loss_contrast = self.compute_loss(motion, pos_motion, neg_motion, length)
        loss = rec_loss + joint_loss + kl_loss * self.cfg.vae.beta_kl + loss_contrast * self.cfg.vae.beta_contrast
        self.log('val/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val/rec_loss', rec_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val/joint_loss', joint_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val/kl_loss', kl_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val/contrast_loss', loss_contrast, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        from eval.interhuman.evaluator import InterHumanEvaluator
        self.evaluator = InterHumanEvaluator(model=self, is_mm=True)
        result = self.evaluator.evaluation(replication_times=20)
        output = {}
        for k, v in result.items():
            try:
                if len(v) > 1:
                    for i in range(len(v)):
                        output[f'{k}_{i+1}'] = v[i]
                else:
                    output[f'{k}'] = v
            except:
                output[f'{k}'] = v
        return output
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.vae.train.lr)
        
        # Warm-up scheduler for the first 10 epochs 
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, total_iters=1
        )
        
        # Exponential decay scheduler to decay LR by 0.5 every 500 epochs
        gamma = 0.5 ** (1 / 1000)
        decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma
        )
        
        # Combine schedulers: first warm-up, then exponential decay
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[10]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }