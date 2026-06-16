from models.chmld.chmld import CHMLD
from omegaconf import OmegaConf
import os


cfg = OmegaConf.load('cfg/chmld/chmld.yaml')



model = CHMLD(cfg)
import sys
iargs = sys.argv[1:]
if iargs:
    cfg = OmegaConf.merge(cfg,  OmegaConf.from_dotlist(iargs))

    ckpt_file = 'best.ckpt'
    if hasattr(cfg.mld, 'is_best'):
        is_best = cfg.mld.is_best
        
exp_name = cfg.mld.name
if os.path.exists('ckpt/' + exp_name + '/' + ckpt_file):
    version_id = os.listdir('ckpt/' + exp_name + '/lightning_logs/')[0]
    cfg = OmegaConf.load(f'ckpt/{exp_name}/lightning_logs/{version_id}/hparams.yaml')
    model = CHMLD.load_from_checkpoint(
        'ckpt/' + exp_name + '/' + ckpt_file,
        cfg=cfg
    )
model.eval()

guidance_list = [3.5]

metrics = {}

for guidance in guidance_list:
    model.guidance_scale = guidance
    model.do_classifier_free_guidance = guidance > 1.0
    model.name = exp_name + f'_g{guidance}'
    model.eval()
    output = model.on_test_epoch_end()
    print('----------------------------', guidance, '----------------------------')
    for k, v in output.items():
        if k not in metrics:
            metrics[k] = []
        metrics[k].append(v)
    
        
        
print(exp_name, 
      metrics)

