
import os

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.interhuman import InterHumanContrastive, interhuman_contrastive_collate
from models.chmld.chvae import CHVAE

from omegaconf import OmegaConf

def main():

    data_cfg = OmegaConf.load('cfg/interhuman.yaml')
    cfg = OmegaConf.load('cfg/chmld/chmld.yaml')

    import sys
    iargs = sys.argv[1:]
    if iargs:
        cfg = OmegaConf.merge(cfg,  OmegaConf.from_dotlist(iargs))


    
    
    exp_name = cfg.vae.name

        
    if os.path.exists('ckpt/' + exp_name + '/last.ckpt'):
        version_id = os.listdir('ckpt/' + exp_name + '/lightning_logs')[0]
        cfg = OmegaConf.load('ckpt/' + exp_name + '/lightning_logs/' + version_id + '/hparams.yaml')
        model = CHVAE.load_from_checkpoint(
            'ckpt/' + exp_name + '/last.ckpt',
            cfg=cfg
        )
    else:
        model = CHVAE(cfg=cfg)

    train_dataset = InterHumanContrastive(opt=data_cfg.interhuman, normalize=cfg.vae.get('do_norm', False),  
                                          pos_translation=cfg.vae.pos_translation, 
                                          uncontact_translation=cfg.vae.uncontact_translation)
    test_dataset = InterHumanContrastive(opt=data_cfg.interhuman_test, normalize=cfg.vae.get('do_norm', False),
                                          pos_translation=cfg.vae.pos_translation, 
                                          uncontact_translation=cfg.vae.uncontact_translation)

    print(len(train_dataset), len(test_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.vae.train.batch_size,
        shuffle=True,
        collate_fn=interhuman_contrastive_collate,
    )

    # Create DataLoader for the validation set
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.vae.train.batch_size,
        shuffle=False,
        collate_fn=interhuman_contrastive_collate
    )


    trainer = Trainer(
        default_root_dir='ckpt/' + exp_name,
        max_epochs=cfg.vae.train.epochs,
        callbacks=[
            ModelCheckpoint(
                save_top_k=1,
                monitor="val/loss",
                mode="min",
                dirpath="ckpt/" + exp_name,
                filename="best",
                enable_version_counter=False,
                save_last=True
            )
        ],
        check_val_every_n_epoch=10,
        accelerator='cuda',
        devices=1
    )

    trainer.fit(model, train_dataloader, val_dataloader)


    
if __name__ == "__main__":
    main()