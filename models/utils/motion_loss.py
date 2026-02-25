""" This defines the motion related loss functions"""

import torch.nn.functional as F
import torch


def JointLossInterHuman(motion_rec, 
                        motion_gt, 
                        channel_first=False):
    """
    motion_rec: [B, L, C]
    motion_gt: [B, L, C]
    """
    if channel_first:
        motion_rec = motion_rec.permute(0, 2, 1)
        motion_gt = motion_gt.permute(0, 2, 1)
        
    return F.l1_loss(motion_rec, motion_gt)


def JointVelLossInterHuman(motion_rec, motion_gt, channel_first=False):
    """
    motion_rec: [B, L, C]
    motion_gt: [B, L, C]
    """
    if channel_first:
        motion_rec = motion_rec.permute(0, 2, 1)
        motion_gt = motion_gt.permute(0, 2, 1)
    
    motion_rec_vel = motion_rec[:-1,:66] - motion_rec[1:]
    motion_gt_vel = motion_gt[:-1] - motion_gt[1:]
    return F.l1_loss(motion_rec_vel, motion_gt_vel)
        
        
def DistanceMapLossInterHuman(motion_rec, motion_gt, channel_first=False):
    """
    motion_rec: [B, L, C]
    motion_gt: [B, L, C]
    """
    return 0