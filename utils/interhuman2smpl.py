# This script is used to translate the 262 representation to the smpl 69 representation. 

from utils.body_model import BodyModel
from utils.paramUtil import relax_hand_pose
from utils.joints2smpl import joints2smpl

import numpy as np
import torch

import os

J2S = joints2smpl(cuda=False)

from torch import nn
class InterHuman2SMPL(nn.Module):
    def __init__(self):
        super().__init__()
        self.bm = BodyModel(bm_fname="deps/body_models/smplx/neutral/model.npz", num_betas=16).cuda()
        self.bm.eval()
        self.num_joints_smpl = 22
        self.num_joints_hands = 30
        
    def interhuman2mesh(self, motion:torch.Tensor):
        B, T, D = motion.shape
        smpl_data1 = J2S.joint2smpl(input_joints=motion.reshape(B*T, D).detach().cpu()[:,:66])
        smpl_data2 = J2S.joint2smpl(input_joints=motion.reshape(B*T, D).detach().cpu()[:,262:262+66])
        
        mesh1 = self.bm.forward(
            root_orient=torch.Tensor(smpl_data1['poses'][:, :3*1]).cuda(),
            pose_body=torch.Tensor(smpl_data1['poses'][:, 3*1:(3*self.num_joints_smpl)]).cuda(),
            pose_hand=torch.Tensor(smpl_data1['poses'][:, -3*self.num_joints_hands:]).cuda(),
            trans=torch.Tensor(smpl_data1['trans']).cuda()       
        ).v
        mesh2 = self.bm.forward(
            root_orient=torch.Tensor(smpl_data2['poses'][:, :3*1]).cuda(),
            pose_body=torch.Tensor(smpl_data2['poses'][:, 3*1:(3*self.num_joints_smpl)]).cuda(),
            pose_hand=torch.Tensor(smpl_data2['poses'][:, -3*self.num_joints_hands:]).cuda(),
            trans=torch.Tensor(smpl_data2['trans']).cuda()
        ).v
        return mesh1.reshape(B, T, -1, 3), mesh2.reshape(B, T, -1, 3), self.bm.f



def to_np_array(motion: torch.Tensor):
    if type(motion) is torch.Tensor:
        return motion.detach().cpu().numpy()
    elif type(motion) is np.ndarray:
        return motion
    else:
        raise TypeError

def to_torch(motion: np.ndarray):
    if type(motion) is torch.Tensor:
        return motion.detach().cpu()
    elif type(motion) is np.ndarray:
        return torch.Tensor(motion)
    else:
        raise TypeError    


def interhuman2smpl(motion:torch.Tensor, length:int=-1):
    """
    Args:
        motion (torch.Tensor | np.ndarray): [L, 262]
            3 * 22 +  22 * 3 + 21 * 6 +  + 4
    """
    smpl_data = J2S.joint2smpl(input_joints=motion[:length, :66].detach().cpu())
    return {'trans':smpl_data['trans'], 
            'poses':smpl_data['poses']}
    
    

def dump_interhuman_npz(motion, 
                        name:str,
                        folder:str = './',
                        gender='neutral', betas=torch.zeros(16,), fps=30, dataset='test'):
    smpl_data = interhuman2smpl(motion)
    smpl_data.update({
        'gender': gender,
        'betas': betas,
        'mocap_framerate': fps,
        'dataset':dataset
    })
    np.savez(os.path.join(folder, name+'.npz'), **smpl_data)
    
def dump_interhuman_cat_npzs(motions,
                             name:str,
                             folder:str = './',
                             gender='neutral', betas=torch.zeros(16,), fps=30, dataset='test'):
    """ motions in shape of [L, 262 * 2]"""
    motion1 = motions[:, :262]
    motion2 = motions[:, 262:]
    dump_interhumman_npzs(motion1, motion2, name, folder, gender, betas, fps, dataset)
    
    
def dump_interhumman_npzs(motion1,
                          motion2, 
                          name:str,
                          folder:str = './',
                          gender='neutral', betas=torch.zeros(16,), fps=30, dataset='test'):
    dump_interhuman_npz(motion1, name + 'p1', folder, gender, betas, fps, dataset)
    dump_interhuman_npz(motion2, name + 'p2', folder, gender, betas, fps, dataset)