import torch
import smplx
import numpy as np
from utils.constant import *
from scipy.interpolate import interp1d
from torch import nn, einsum
import pytorch3d.transforms as T


class JointsToSMPLX(nn.Module):
    def __init__(self, input_dim=72, output_dim=132, hidden_dim=64, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def optimize_smpl(pose_pred, joints, joints_ind, hand_pca=45):
    device = joints.device
    len = joints.shape[0]

    smpl_model = smplx.create('./deps', model_type='smplx',
                              gender='neutral', ext='npz',
                              num_betas=10,
                              use_pca=False,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=True,
                              create_right_hand_pose=True,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=len,
                              ).to(device)
    smpl_model.eval()

    # weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 100]).reshape(nb_joints, 1).repeat(1, 3).to(device)
    joints = joints.reshape(len, -1, 3) + torch.tensor(pelvis_shift).to(device)
    pose_input = torch.nn.Parameter(pose_pred.detach(), requires_grad=True)
    transl = torch.nn.Parameter(torch.zeros(pose_pred.shape[0], 3).to(device), requires_grad=True)
    # left_hand = torch.nn.Parameter(torch.zeros(pose_pred.shape[0], hand_pca).to(device), requires_grad=True)
    # right_hand = torch.nn.Parameter(torch.zeros(pose_pred.shape[0], hand_pca).to(device), requires_grad=True)
    left_hand = torch.from_numpy(relaxed_hand_pose[:45].reshape(1, -1).repeat(pose_pred.shape[0], axis=0)).to(device)
    right_hand = torch.from_numpy(relaxed_hand_pose[45:].reshape(1, -1).repeat(pose_pred.shape[0], axis=0)).to(device)
    optimizer = torch.optim.Adam(params=[pose_input, transl], lr=0.05)
    loss_fn = nn.MSELoss()
    vertices_output = None
    for step in range(100):
        smpl_output = smpl_model(transl=transl, body_pose=pose_input[:, 3:], global_orient=pose_input[:, :3], return_verts=True,
                                 left_hand_pose=left_hand,# @ left_hand_components[:hand_pca],
                                 right_hand_pose=right_hand,# @ right_hand_components[:hand_pca],
                                 )
        joints_output = smpl_output.joints[:, joints_ind].reshape(len, -1, 3)
        vertices_output = smpl_output.vertices[:, ::10].detach().cpu()
        loss = loss_fn(joints[:, :], joints_output[:, :])
        # loss = torch.mean((joints - joints_output) ** 2 * weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(loss.item())





    #left_hand = left_hand @ left_hand_components[:hand_pca]
    #right_hand = right_hand @ right_hand_components[:hand_pca]

    return pose_input.detach().cpu(), transl.detach().cpu(), left_hand.detach().cpu(), right_hand.detach().cpu(), vertices_output


def joints_to_smpl(model, joints, joints_ind, interp_s=1):
    joints = interpolate_joints(joints, scale=interp_s)
    # joints = interpolate_joints(joints, scale=0.33)
    # joints = interpolate_joints(joints, scale=interp_s * 3)
    input_len = joints.shape[0]
    joints = joints.reshape(input_len, -1, 3)
    joints = joints.permute(1, 0, 2)
    trans_np = joints[0].detach().cpu()
    joints = joints - joints[0]
    joints = joints.permute(1, 0, 2)
    joints = joints.reshape(input_len, -1)
    pose_pred = model(joints)
    pose_pred = pose_pred.reshape(-1, 6)
    pose_pred = T.matrix_to_axis_angle(T.rotation_6d_to_matrix(pose_pred)).reshape(input_len, -1)
    # pose_pred = pose_pred[:seq_len]
    pose_output, transl, left_hand, right_hand, vertices = optimize_smpl(pose_pred, joints, joints_ind)

    transl = trans_np - torch.Tensor(pelvis_shift) + transl

    vertices = vertices + transl.reshape(-1, 1, 3)


    return pose_output, transl, left_hand, right_hand, vertices


def interpolate_joints(joints, scale):
    if scale == 1:
        return joints
    device = joints.device
    joints = joints.detach().cpu().numpy()
    in_len = joints.shape[0]
    out_len = int(in_len * scale)
    joints = joints.reshape(in_len, -1)
    x = np.array(range(in_len))
    xnew = np.linspace(0, in_len - 1, out_len)
    f = interp1d(x, joints, axis=0)
    joints_new = f(xnew)
    joints_new = torch.from_numpy(joints_new).to(device).float()

    return  joints_new


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


class JIK(nn.Module):
    def __init__(self,
                 j2s_path:str='deps/model_joints_to_smpl_wrist.pth',
                 palm_len = 0.05,
                 joints_idx = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 40 ]):
        super().__init__()
        # for joint 24, we can simply by adding a null one
        self.j2s_model = JointsToSMPLX()
        self.j2s_model.load_state_dict(torch.load(j2s_path))
        
        self.idx_l_elbow = 18
        self.idx_r_elbow = 19
        self.idx_l_wrist = 20
        self.idx_r_wrist = 21

        
        self.palm_len = palm_len
        self.joints_idx = joints_idx
    
    def forward(self, joints:torch.Tensor):
        """ joints should be in the shape of [L, 66] or [L, 72]"""
        joints = to_torch(joints)
        if len(joints.shape) != 2:
            joints = joints.reshape(joints.shape[0], -1)
        if joints.shape[-1] != 24 * 3:
            joints = self.add_psedo_wrist(joints)
        
        poses, trans, _, _, _ = joints_to_smpl(
            model=self.j2s_model, 
            joints=joints,
            joints_ind=self.joints_idx
        )
        
        return {'trans':trans.numpy(), 'poses':self.fill_up_poses(pose=poses).numpy()}
    
    
    def add_psedo_wrist(self, joints:torch.Tensor):
        """joints in shape of [L, 66]"""
        joints = joints.reshape(-1, 22, 3)
        vec_l_elbow = joints[:, self.idx_l_wrist] - joints[:, self.idx_l_elbow] # L, 3
        vec_r_elbow = joints[:, self.idx_r_wrist] - joints[:, self.idx_r_elbow] # L, 3

        norm_l_elbow = torch.norm(vec_l_elbow, dim=-1, keepdim=True)
        norm_r_elbow = torch.norm(vec_r_elbow, dim=-1, keepdim=True)

        # Avoid division by zero: If norm is zero, keep vector zero
        norm_l_elbow = torch.where(norm_l_elbow == 0, torch.ones_like(norm_l_elbow), norm_l_elbow)
        norm_r_elbow = torch.where(norm_r_elbow == 0, torch.ones_like(norm_r_elbow), norm_r_elbow)

        # Normalize only if the vector is nonzero
        vec_l_elbow = torch.where(norm_l_elbow > 0, vec_l_elbow / norm_l_elbow * self.palm_len, vec_l_elbow)
        vec_r_elbow = torch.where(norm_r_elbow > 0, vec_r_elbow / norm_r_elbow * self.palm_len, vec_r_elbow)
        # print(vec_l_elbow)
        # print(vec_r_elbow)
        joint_l_hand = joints[:, self.idx_l_wrist] + vec_l_elbow
        joint_r_hand = joints[:, self.idx_l_wrist] + vec_r_elbow
        
        joints = torch.cat([joints,
                            joint_l_hand.unsqueeze(1),
                            joint_r_hand.unsqueeze(1)], dim=1) # L, 24, 3
        
        return joints.reshape(joints.shape[0], -1) # L, 72
    
    
    
    def fill_up_poses(self, pose:torch.Tensor):
        ''' fill up the 66 into a 165 ones'''
        B, _ = pose.shape
        poses_hand = torch.Tensor(relaxed_hand_pose).expand((B, -1))
        poses_jaw_eye = torch.zeros(B, 3 * 3)
        return torch.cat([pose, poses_jaw_eye, poses_hand], dim=-1)
        

    def dump_interhuman_npz(self,
                            motion, 
                            name:str,
                            folder:str = './',
                            gender='neutral', betas=torch.zeros(16,), fps=30, dataset='test'):
        smpl_data = self.forward(motion)
        smpl_data.update({
            'gender': gender,
            'betas': betas,
            'mocap_framerate': fps,
            'dataset':dataset
        })
        np.savez(os.path.join(folder, name+'.npz'), **smpl_data)
        

    def dump_interhumman_npzs(self,
                            motion1,
                            motion2, 
                            name:str,
                            folder:str = './',
                            gender='neutral', betas=torch.zeros(16,), fps=30, dataset='test'):
        self.dump_interhuman_npz(motion1, name + 'p1')
        self.dump_interhuman_npz(motion2, name + 'p2')