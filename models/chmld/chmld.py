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
from models.hmld.hdn import HDenoiser
from models.chmld.chvae import CHVAE
from models.hmld.textenc import TextEncoder

from diffusers import DDPMScheduler, DDIMScheduler
import inspect
import numpy as np
from omegaconf import OmegaConf
from utils.utils import MotionNormalizerTorch

class CHMLD(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.name = cfg.mld.name
        
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        self.guidance_scale = cfg.mld.guidance_scale
        self.guidance_uncodp = cfg.mld.guidance_uncondp
        self.latent_size = cfg.mld.latent_size
        self.latent_dim = cfg.mld.latent_dim
        
        self.z3_scale = cfg.mld.z3_scale
        self.is_vp = cfg.mld.is_vp if hasattr(cfg.mld, 'is_vp') else False
        self.is_sample = cfg.mld.is_sample if hasattr(cfg.mld, 'is_sample') else False
        self.prediction_type = 'v_prediction' if self.is_vp else 'epsilon'
        self.prediction_type = 'sample' if self.is_sample else self.prediction_type
        self.load_vae()
        self.denoiser = HDenoiser(self.cfg.mld)
        
        self.is_dpm = cfg.mld.is_dpm if hasattr(cfg.mld, 'is_dpm') else False
        
        # Noise Scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.NoiseScheduler.num_train_timesteps,
            beta_start=cfg.NoiseScheduler.beta_start,
            beta_end=cfg.NoiseScheduler.beta_end,
            beta_schedule=cfg.NoiseScheduler.beta_schedule,
            clip_sample=cfg.NoiseScheduler.clip_sample,
            prediction_type=self.prediction_type
        )
        
        if not self.is_dpm:
            self.scheduler = DDIMScheduler(
                num_train_timesteps=cfg.NoiseScheduler.num_train_timesteps,
                beta_start=cfg.NoiseScheduler.beta_start,
                beta_end=cfg.NoiseScheduler.beta_end,
                beta_schedule=cfg.NoiseScheduler.beta_schedule,
                clip_sample=cfg.NoiseScheduler.clip_sample,
                set_alpha_to_one=cfg.NoiseScheduler.set_alpha_to_one,
                steps_offset=cfg.NoiseScheduler.steps_offset,
                    prediction_type=self.prediction_type
            )
        else:
            from diffusers import DPMSolverMultistepScheduler
            self.scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=cfg.NoiseScheduler.num_train_timesteps,
                beta_start=cfg.NoiseScheduler.beta_start,
                beta_end=cfg.NoiseScheduler.beta_end,
                beta_schedule=cfg.NoiseScheduler.beta_schedule,
                prediction_type=self.prediction_type,
                algorithm_type=cfg.NoiseScheduler.algorithm_type
            )
            
        self.do_classifier_free_guidance = self.guidance_scale > 1.0  
        
        # text encoder
        self.text_encoder = TextEncoder(cfg.mld)
        
        self.normalizer = MotionNormalizerTorch()
    
    def load_vae(self):
        import os
        from omegaconf import OmegaConf
        self.vae_name = self.cfg.vae.name
        print(f'---------Loading vae from {self.vae_name}---------')
        self.vae_path = f'ckpt/{self.vae_name}/last.ckpt'
        version_id = os.listdir(f'ckpt/{self.vae_name}/lightning_logs/')[0]
        self.vae_cfg = f'ckpt/{self.vae_name}/lightning_logs/{version_id}/hparams.yaml'
        self.vae_cfg = OmegaConf.load(self.vae_cfg)
        
        assert self.latent_size == self.vae_cfg.vae.latent_size, f'latent_size mismatch, {self.latent_size} != {self.vae_cfg.vae.latent_size}'
        assert self.latent_dim == self.vae_cfg.vae.latent_dim, f'latent_dim mismatch, {self.latent_dim} != {self.vae_cfg.vae.latent_dim}'
        
        self.vae = CHVAE.load_from_checkpoint(self.vae_path, cfg=self.vae_cfg)
        self.vae.eval()
        self.vae.to(self.device)
        self.vae.freeze()
        print(f'---------Loaded vae from {self.vae_path}---------')
        
        
        
    def denoise(self, text:list[str], length:Tensor):
        # init latents
        bsz = len(text)
        if self.do_classifier_free_guidance:
            bsz = bsz // 2

        latents = torch.randn(
            (bsz, self.latent_size * 3, self.latent_dim),
            device=length.device,
            dtype=torch.float)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.NoiseScheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(length.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.NoiseScheduler.eta

        cond_B1D = self.text_encoder.encode_text(text).unsqueeze(1)

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            if self.is_dpm:
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser.forward(
                x=latent_model_input,
                cond_B1D=cond_B1D,
                timesteps=t.unsqueeze(0),
            )
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample


        return latents
        
        
    def denoise_single(self, latents, text:list[str]):
        """ do single step denoise
        Args:
            latents (Tensor): [B, latent_size, latent_dim]
            text (list[str]): [B]

        Returns:
            noise_set: dict, {
                'noise': Tensor, [B, latent_size, latent_dim]
                'noise_pred': Tensor, [B, latent_size, latent_dim]
            }
        """
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        cond_B1D = self.text_encoder.encode_text(text).unsqueeze(1)
        noise_pred = self.denoiser.forward(
            x=noisy_latents,
            cond_B1D=cond_B1D,
            timesteps=timesteps,
        )
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
        }
        return n_set
    
    
    
    
    
    def train_diffusion_forward(self, motion, text, length):
        # motion encode
        with torch.no_grad():
            z1, z2, z3 = self.vae.encode(motion, length)
            z3 = z3 * self.z3_scale

            z = torch.cat([z1, z2, z3], dim=1)
        # classifier free guidance: randomly drop text during training
        text = [
            "" if np.random.rand(1) < self.guidance_uncodp else i
            for i in text
        ]
        # diffusion process return with noise and noise_pred
        n_set = self.denoise_single(latents=z, text=text)
        return {**n_set}
    
    
    def forward_test(self, batch):
        device = next(self.parameters()).device
        text = batch['text']
        length = batch['motion_lens'].to(device)
        if self.do_classifier_free_guidance:
            uncond_tokens = [""] * len(length)
            uncond_tokens.extend(text)
            text = uncond_tokens
            
        with torch.no_grad():
            z = self.denoise(text=text, length=length)
            z1, z2, z3 = torch.chunk(z, 3, dim=1)
            x1, x2 = self.vae.decoder.forward(z1=z1, z2=z2, z3=z3 / self.z3_scale, length=length)
            if self.vae.do_norm:    
                x1 = self.normalizer.forward(x1)
                x2 = self.normalizer.forward(x2)
            motion_gen = torch.cat([x1, x2], dim=-1)
        return {'output': motion_gen}
    

    
    def compute_loss(self, motions, text, lengths):

        n_rst = self.train_diffusion_forward(motions, text, lengths)
        noise_pred = n_rst['noise_pred']
        noise = n_rst['noise']
        if self.cfg.mld.is_l1:
            loss = F.l1_loss(noise_pred, noise)
        else:
            loss = F.mse_loss(noise_pred, noise)
        return loss
    
    
    def training_step(self, batch):
        motions = batch['motions']
        text = batch['text']
        lengths = batch['motion_lens']
        loss = self.compute_loss(motions, text, lengths)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        motions = batch['motions']
        text = batch['text']
        lengths = batch['motion_lens']
        loss = self.compute_loss(motions, text, lengths)
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
        
    def on_validation_epoch_end(self):
        from eval.interhuman.evaluator import InterHumanEvaluator
        self.evaluator = InterHumanEvaluator(model=self)
        result = self.evaluator.evaluation(replication_times=1)
        for k, v in result.items():
            try:
                if len(v) > 1:
                    for i in range(len(v)):
                        self.log(f'eval/{k}_{i+1}', v[i], on_epoch=True, prog_bar=True)
                else:
                    self.log(f'eval/{k}', v, on_epoch=True, prog_bar=True)
            except:
                self.log(f'eval/{k}', v, on_epoch=True, prog_bar=True)
                
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.mld.train.lr)
        
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
    
    
    
    
    
    