import numpy as np
import torch
from utils.rotation_conversions import (
    rotation_6d_to_axis_angle, 
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_axis_angle
)
import os

def to_torch(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    else:
        raise TypeError("Input should be a torch.Tensor or a numpy.ndarray")

from torch import nn
class InterX2SMPL(nn.Module):
    """ The default input shape is [T, 56, 12] """
    def __init__(self):
        super().__init__()
        from utils.body_model import BodyModel
        self.bm = BodyModel(bm_fname="deps/body_models/smplx/neutral/model.npz", num_betas=16).cuda()
        self.bm.eval()

    def interx2smpl(self, interx_motion):
        # T, 56, 12
        interx_motion = to_torch(interx_motion)
        poses1 = rotation_6d_to_axis_angle(interx_motion[:, :-1, :6]) # T, 55, 3
        poses2 = rotation_6d_to_axis_angle(interx_motion[:, :-1, 6:]) # T, 55, 3
        trans1 = interx_motion[:, -1, :3]
        trans2 = interx_motion[:, -1, 6:9]
        dict1 = {
            'root_orient': poses1[:, :1, :].reshape(-1, 3),
            'pose_body': poses1[:, 1:22, :].reshape(-1, 3*21),
            'pose_hand': poses1[:, -30:, :].reshape(-1, 3*30),
            'trans': trans1.reshape(-1, 3),
        }
        
        dict2 = {
            'root_orient': poses2[:, :1, :].reshape(-1, 3),
            'pose_body': poses2[:, 1:22, :].reshape(-1, 3*21),
            'pose_hand': poses2[:, -30:, :].reshape(-1, 3*30),
            'trans': trans2.reshape(-1, 3),
        }
        
        return dict1, dict2
    
    def interx2mesh(self, interx_motion):
        ''' Args:
            interx_motion: torch.Tensor of shape (B, T, 56, 12)
        '''
        B, T, _, _ = interx_motion.shape
        interx_motion = interx_motion.reshape(B*T, 56, 12)
        dict1, dict2 = self.interx2smpl(interx_motion)
        
        
        mesh1 = self.bm.forward(
            root_orient=dict1['root_orient'],
            pose_body=dict1['pose_body'],
            pose_hand=dict1['pose_hand'],
            trans=dict1['trans']
        ).v
        mesh2 = self.bm.forward(
            root_orient=dict2['root_orient'],
            pose_body=dict2['pose_body'],
            pose_hand=dict2['pose_hand'],
            trans=dict2['trans']
        ).v
        return mesh1.reshape(B, T, -1, 3), mesh2.reshape(B, T, -1, 3), self.bm.f
    
    
def interx2smpl(interx_motion,                             
                name:str,
                folder:str = './',
                gender='neutral', 
                betas=torch.zeros(16,), fps=30, dataset='test'):
    """ The default input shape is [T, 56, 12] with no normalization"""
    interx_motion = to_torch(interx_motion)
    poses1 = rotation_6d_to_axis_angle(interx_motion[:, :-1, :6])
    poses2 = rotation_6d_to_axis_angle(interx_motion[:, :-1, 6:])
    trans1 = interx_motion[:, -1, :3]
    trans2 = interx_motion[:, -1, 6:9]
    
    dict1 = {
        'poses': poses1.numpy().reshape(-1, 3*55),
        'trans': trans1.numpy(),
        'gender': gender,
        'betas': betas,
        'mocap_framerate': fps,
        'dataset': dataset
    }
    
    dict2 = {
        'poses': poses2.numpy().reshape(-1, 3*55),
        'trans': trans2.numpy(),
        'gender': gender,
        'betas': betas,
        'mocap_framerate': fps,
        'dataset': dataset
    }
    
    np.savez(os.path.join(folder, name+'_p1.npz'), **dict1)
    np.savez(os.path.join(folder, name+'_p2.npz'), **dict2)
    
    
def aa2smpl(aa_motion, 
            name:str, 
            folder:str = './', 
            gender='neutral', 
            betas=torch.zeros(16,), 
            fps=30, 
            dataset='test'):
    interx_motion = to_torch(aa_motion)
    poses1 = (interx_motion[:, :-1, :3])
    poses2 = (interx_motion[:, :-1, 3:])
    trans1 = interx_motion[:, -1, :3]
    trans2 = interx_motion[:, -1, 3:]
    
    dict1 = {
        'poses': poses1.numpy().reshape(-1, 3*55),
        'trans': trans1.numpy(),
        'gender': gender,
        'betas': betas,
        'mocap_framerate': fps,
        'dataset': dataset
    }
    
    dict2 = {
        'poses': poses2.numpy().reshape(-1, 3*55),
        'trans': trans2.numpy(),
        'gender': gender,
        'betas': betas,
        'mocap_framerate': fps,
        'dataset': dataset
    }
    
    np.savez(os.path.join(folder, name+'_p1.npz'), **dict1)
    np.savez(os.path.join(folder, name+'_p2.npz'), **dict2)