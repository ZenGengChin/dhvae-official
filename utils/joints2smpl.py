import numpy as np
import os
import torch
from utils.j2s import config
import smplx
import h5py
from utils.j2s.smplify import SMPLify3D
from tqdm import tqdm
from utils.rotation_conversions import  matrix_to_rotation_6d, axis_angle_to_matrix, rotation_6d_to_axis_angle
from utils.constant import relaxed_hand_pose
import argparse

class joints2smpl:

    def __init__(self, cuda=True):
        
        if not cuda: 
            self.device = 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.num_joints = 22  # for HumanML3D
        self.joint_category = "AMASS"
        self.num_smplify_iters = 150
        self.fix_foot = False
        smplmodel = smplx.create(config.SMPL_MODEL_DIR,
                                 model_type="smpl", gender="neutral", ext="pkl").to(self.device)

        # ## --- load the mean pose as original ----
        smpl_mean_file = config.SMPL_MEAN_FILE

        self.file = h5py.File(smpl_mean_file, 'r')

        #

        # # #-------------initialize SMPLify
        self.smplify = SMPLify3D(smplxmodel=smplmodel,
                            joints_category=self.joint_category,
                            num_iters=self.num_smplify_iters,
                            step_size=0.03,
                            device=self.device)


    def npy2smpl(self, npy_path):
        out_path = npy_path.replace('.npy', '_rot.npy')
        motions = np.load(npy_path, allow_pickle=True)[None][0]
        n_samples = motions['motion'].shape[0]
        all_thetas = []
        for sample_i in tqdm(range(n_samples)):
            thetas, _ = self.joint2smpl(motions['motion'][sample_i].transpose(2, 0, 1))  # [nframes, njoints, 3]
            all_thetas.append(thetas.cpu().numpy())
        motions['motion'] = np.concatenate(all_thetas, axis=0)
        print('motions', motions['motion'].shape)

        print(f'Saving [{out_path}]')
        np.save(out_path, motions)
        exit()



    def joint2smpl(self, input_joints, init_params=None):
        ''' input_joints in shape of [L, J, 3] // [L, 66]
            return: [L, 69]
        '''
        batch_size = len(input_joints)
        
        if len(input_joints.shape) != 3:
            input_joints = input_joints.reshape(batch_size, -1, 3)
        
        
        self.init_mean_pose = torch.from_numpy(self.file['pose'][:]).unsqueeze(0).repeat(batch_size, 1).float().to(self.device)
        self.init_mean_shape = torch.from_numpy(self.file['shape'][:]).unsqueeze(0).repeat(batch_size, 1).float().to(self.device)
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)
        
        _smplify = self.smplify # if init_params is None else self.smplify_fast
        pred_pose = torch.zeros(batch_size, 72).to(self.device)
        pred_betas = torch.zeros(batch_size, 10).to(self.device)
        pred_cam_t = torch.zeros(batch_size, 3).to(self.device)
        keypoints_3d = torch.zeros(batch_size, self.num_joints, 3).to(self.device)

        # joints3d = input_joints[idx]  # *1.2 #scale problem [check first]
        keypoints_3d = torch.Tensor(input_joints).to(self.device).float()

        # if idx == 0:
        if init_params is None:
            pred_betas = self.init_mean_shape
            pred_pose = self.init_mean_pose
            pred_cam_t = self.cam_trans_zero
        else:
            pred_betas = init_params['betas']
            pred_pose = init_params['pose']
            pred_cam_t = init_params['cam']

        if self.joint_category == "AMASS":
            confidence_input = torch.ones(self.num_joints)
            # make sure the foot and ankle
            if self.fix_foot == True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        else:
            print("Such category not settle down!")

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss = _smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(self.device),
            # seq_ind=idx
        )

        thetas = new_opt_pose.reshape(batch_size, 24, 3)
        # thetas = matrix_to_rotation_6d(axis_angle_to_matrix(thetas))  # [bs, 24, 6]
        root_loc = torch.Tensor(keypoints_3d[:, 0])  # [bs, 3]
        root_loc += torch.tensor([0,0.3515,0]).to(root_loc.device)
        #root_loc = torch.cat([root_loc, torch.zeros_like(root_loc)], dim=-1).unsqueeze(1)  # [bs, 1, 6]
        #thetas = torch.cat([thetas, root_loc], dim=1).unsqueeze(0).permute(0, 2, 3, 1)  # [1, 25, 6, 196]
        thetas = thetas[:, :self.num_joints,:].reshape((-1, 3*self.num_joints)) # [196, 132]
        # result = torch.cat([root_loc, thetas], dim=1)
        thetas = thetas.detach().cpu()
        
        return {'trans':root_loc.detach().cpu().numpy(), 'poses':self.fill_up_poses(pose=thetas).numpy()}
        #return thetas.clone().detach(), {'pose': new_opt_joints[0, :24].flatten().clone().detach(), 'betas': new_opt_betas.clone().detach(), 'cam': new_opt_cam_t.clone().detach()}

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
        """motion in shape of any just start with [L, 66 + ....]"""
        motion = motion.reshape(-1, 66)
        smpl_data = self.joint2smpl(motion)
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
        self.dump_interhuman_npz(motion1, name + 'p1', folder, gender, betas, fps, dataset)
        self.dump_interhuman_npz(motion2, name + 'p2', folder, gender, betas, fps, dataset)

