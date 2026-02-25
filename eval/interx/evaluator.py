import torch
from collections import OrderedDict
from eval.interx.metrics import *
from eval.interx.utils import *


def evaluate_matching_score(eval_wrapper,motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating MM Distance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
                # print(text_embeddings.shape)
                # print(motion_embeddings.shape)
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                # print(dist_mat.shape)
                mm_dist_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                # print(argsmax.shape)

                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            mm_dist = mm_dist_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = mm_dist
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times=300, emb_scale=1, divide_by=1):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, 
                           mm_num_times=10, emb_scale=1, divide_by=1):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                if len(batch) == 5:
                    batch[2] = batch[2][0]
                    batch[3] = batch[3][0]
                    batch[4] = batch[4][0]
                motion_embedings = eval_wrapper.get_motion_embeddings(batch)
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times=10):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


from omegaconf import OmegaConf
import os
from torch.utils.data import DataLoader


class InterXEvaluator:
    def __init__(self, 
                 model = None,
                 evalmodel_cfg_file = 'cfg/eval_interx.yaml', 
                 data_cfg_file = 'cfg/interx.yaml',
                 device = 'cuda',
                 batch_size = 32,
                 is_mm = False, 
                 file = None):
        
        
        self.device = device
        self.evalmodel_cfg = OmegaConf.load(evalmodel_cfg_file)
        self.data_cfg = OmegaConf.load(data_cfg_file)
        self.batch_size = batch_size
        self.is_mm = is_mm
        self.file = file if file is not None else os.devnull
        
        
        self.mm_num_samples = 100
        self.mm_num_repeats = 30
        
        
        self.eval_wrapper = EvaluatorModelWrapper(self.evalmodel_cfg)
        self.gt_loader, self.gt_dataset = get_dataset_motion_loader(self.data_cfg, batch_size)
        
        if is_mm:
            self.gt_mm_loader = DataLoader(self.gt_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)
        else:
            self.gt_mm_loader = None
        
        self.eval_motion_loaders = {}
        
        self.model = model
        
        if self.model is None:
            self.eval_motion_loaders['token'] = lambda: (self.gt_loader, self.gt_mm_loader)
            
        else:
            self.sample_motion_loader()
    
    def sample_motion_loader(self):
        self.motion_loader, self.mm_motion_loader = get_motion_loader(self.batch_size, self.model, self.gt_dataset, self.device,
                                                                self.mm_num_samples, self.mm_num_repeats, self.data_cfg, is_mm=self.is_mm)
            
        self.eval_motion_loaders[self.model.name] = lambda: (self.motion_loader, self.mm_motion_loader)
    
    def evaluation(self, replication_times=1):
        from datetime import datetime
        with open(self.file, 'w') as f:
            all_metrics = OrderedDict({'MM Distance': OrderedDict({}),
                                    'R_precision': OrderedDict({}),
                                    'FID': OrderedDict({}),
                                    'Diversity': OrderedDict({}),
                                    'MultiModality': OrderedDict({})})
            for replication in range(replication_times):
                motion_loaders = {}
                mm_motion_loaders = {}
                motion_loaders['ground truth'] = self.gt_loader
                self.sample_motion_loader()
                for motion_loader_name, motion_loader_getter in self.eval_motion_loaders.items():
                    motion_loader, mm_motion_loader = motion_loader_getter()
                    motion_loaders[motion_loader_name] = motion_loader
                    mm_motion_loaders[motion_loader_name] = mm_motion_loader
                    
                print(f'==================== Replication {replication} ====================')
                print(f'==================== Replication {replication} ====================', file=f, flush=True)
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(self.eval_wrapper, motion_loaders, f)

                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                fid_score_dict = evaluate_fid(self.eval_wrapper, self.gt_loader, acti_dict, f)

                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                div_score_dict = evaluate_diversity(acti_dict, f)

                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                if self.is_mm:
                    mm_score_dict = evaluate_multimodality(self.eval_wrapper, mm_motion_loaders, f)
                print(f'!!! DONE !!!')
                print(f'!!! DONE !!!', file=f, flush=True)

                for key, item in mat_score_dict.items():
                    if key not in all_metrics['MM Distance']:
                        all_metrics['MM Distance'][key] = [item]
                    else:
                        all_metrics['MM Distance'][key] += [item]

                for key, item in R_precision_dict.items():
                    if key not in all_metrics['R_precision']:
                        all_metrics['R_precision'][key] = [item]
                    else:
                        all_metrics['R_precision'][key] += [item]

                for key, item in fid_score_dict.items():
                    if key not in all_metrics['FID']:
                        all_metrics['FID'][key] = [item]
                    else:
                        all_metrics['FID'][key] += [item]

                for key, item in div_score_dict.items():
                    if key not in all_metrics['Diversity']:
                        all_metrics['Diversity'][key] = [item]
                    else:
                        all_metrics['Diversity'][key] += [item]

                if self.is_mm:
                    for key, item in mm_score_dict.items():
                            all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]

            results_dict = {}

            for metric_name, metric_dict in all_metrics.items():
                print('========== %s Summary ==========' % metric_name)
                print('========== %s Summary ==========' % metric_name, file=f, flush=True)

                for model_name, values in metric_dict.items():
                    mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                    if model_name == self.model.name:
                        results_dict[metric_name] = mean 
                    if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                        print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                        print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                    elif isinstance(mean, np.ndarray):
                        line = f'---> [{model_name}]'
                        for i in range(len(mean)):
                            line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                        print(line)
                        print(line, file=f, flush=True)
        return results_dict
    
if __name__ == '__main__':

    
    from torch import nn
    from utils.interx_utils import InterxNormalizerTorch
    
    class SampleModel(nn.Module):
        def __init__(self, device='cuda'):
            super().__init__()
            self.name = 'SampleModel'
            self.device = device
        
        def forward_test(self, motions, caption, motion_lens):
            # return B, L, 2D will be reshaped automatically
            output = motions.to(self.device)  + 0.02 * torch.randn_like(motions).to(self.device)
            return output
        
    evaluator = InterXEvaluator(model=SampleModel(), device='cuda',
                                    is_mm=True)
    evaluator.evaluation()