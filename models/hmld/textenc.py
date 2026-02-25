import clip
from transformers import CLIPModel, CLIPProcessor
import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_cond = cfg.text_encoder.d_cond
        self.clip_version = cfg.text_encoder.clip_version
        self.transformers_clip_version = cfg.text_encoder.transformers_clip_version
        self.is_intergen_clip = cfg.text_encoder.is_intergen_clip
        
        if self.is_intergen_clip == 'intergen':
            self.load_intergen_clip()
        elif self.is_intergen_clip == 'easy':
            self.load_easy_clip()
        elif self.is_intergen_clip == 'transformers':
            self.load_transformers_clip()
        
    def forward(self, text):
        if self.is_intergen_clip:
            if self.is_intergen_clip:
                return self.encode_intergen_text(text)
            else:
                return self.encode_easy_text(text)
        else:
            return self.encode_easy_text(text)
        

    

    
    
    """ load three types of clip"""
    @torch.no_grad()
    def load_easy_clip(self):
        print('---------Load EasyCLIP from ', self.clip_version, '---------')
        self.clip_model, _ = clip.load(self.clip_version, device="cpu", jit=False, download_root='./deps/')
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_dim = self.d_cond
        
    def load_transformers_clip(self):
        print('---------Load TransformersCLIP from ', self.transformers_clip_version, '---------')
        self.clip_model = CLIPModel.from_pretrained(self.transformers_clip_version)
        self.clip_processor = CLIPProcessor.from_pretrained(self.transformers_clip_version)

    def load_intergen_clip(self):

        ##From InterGen
        print('---------Load InterGen Style CLIP from ', self.clip_version, '---------')
        clip_model, _ = clip.load(self.clip_version, device="cpu", jit=False, download_root='./deps/')
        
        self.clip_dim = self.d_cond

        self.clip_token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.clip_positional_embedding = clip_model.positional_embedding
        self.clip_ln_final = clip_model.ln_final
        self.clip_dtype = clip_model.dtype

        for p in self.clip_transformer.parameters():
            p.requires_grad = False
        for p in self.clip_token_embedding.parameters():
            p.requires_grad = False
        for p in self.clip_ln_final.parameters():
            p.requires_grad = False
        
        clipTransLayer = nn.TransformerEncoderLayer(d_model=self.clip_dim,
                                                    nhead=8,
                                                    dim_feedforward=2048,
                                                    dropout=0.1,
                                                    activation="gelu",
                                                    batch_first=True)
        self.clipTrans = nn.TransformerEncoder(clipTransLayer, num_layers=2)
        self.clipln = nn.LayerNorm(self.clip_dim)
    
    @torch.no_grad()
    def encode_easy_text(self, text: str):
        device = next(self.parameters()).device
        text = clip.tokenize(text, truncate=True).to(device)
        text_features = self.clip_model.encode_text(text)
        return text_features # [B, 768]
    
    @torch.no_grad()
    def encode_transformers_text(self, text: str):
        self.clip_model: CLIPModel = self.clip_model
        self.clip_processor: CLIPProcessor = self.clip_processor
        device = next(self.parameters()).device
        text = self.clip_processor(text, return_tensors='pt', padding=True, 
                                        truncation=True).to(device)
        text_encoded = self.clip_model.get_text_features(**text)
        return text_encoded # [B, 768]
    
    
    def encode_intergen_text(self, text: str):
        device = next(self.parameters()).device
        # From InterGen
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x:torch.Tensor = self.clip_token_embedding(text).type(self.clip_dtype)
            pe_tokens:torch.Tensor = x + self.clip_positional_embedding.type(self.clip_dtype)
            x = pe_tokens.permute(1,0,2)
            x = self.clip_transformer(x)
            x = x.permute(1,0,2)
            clip_out = self.clip_ln_final(x).type(self.clip_dtype)
        
        out = self.clipTrans.forward(clip_out)
        out = self.clipln.forward(out)
        feat_clip_text = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return feat_clip_text # [B, 768]
    
    
    def encode_text(self, text: str):
        # return [B, 768]
        if self.is_intergen_clip == 'intergen':
            return self.encode_intergen_text(text)
        elif self.is_intergen_clip == 'easy':
            return self.encode_easy_text(text)
        elif self.is_intergen_clip == 'transformers':
            return self.encode_transformers_text(text)
        else:
            raise ValueError(f"Invalid clip version: {self.is_intergen_clip}")
        