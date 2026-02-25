import numpy as np

### CONSTANTS ###

HML_RAW_OFFSETS = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])
HML_KINEMATIC_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
HML_LEFT_HAND_CHAIN = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
HML_RIGHT_HAND_CHAIN = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]
HML_TGT_SKEL_ID = '000021'
HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]
NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints
HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 
                                                                  'left_hip', 
                                                                  'right_hip', 
                                                                  'left_knee', 
                                                                  'right_knee', 
                                                                  'left_ankle', 
                                                                  'right_ankle', 
                                                                  'left_foot', 
                                                                  'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_ROOT_HORIZONTAL_MASK = np.concatenate(([True]*(1+2) + [False],
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(3)),
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(6)),
                                np.zeros_like(HML_ROOT_BINARY.repeat(3)),
                                [False] * 4))
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK
HML_TRAJ_MASK = np.zeros_like(HML_ROOT_MASK)
HML_TRAJ_MASK[1:3] = True
NUM_HML_FEATS = 263
L_IDX1, L_IDX2 = 5, 8 # Lower legs
FID_R, FID_L = [8, 11], [7, 10] # Right/Left foot
FACE_JOINT_INDX = [2, 1, 17, 16] # Face direction, r_hip, l_hip, sdr_r, sdr_l
R_HIP, L_HIP = 2, 1 # l_hip, r_hip
JOINTS_NUM = 22

### FUNCTIONS ###

def expand_mask(mask, shape):
    """
    expands a mask of shape (num_feat, seq_len) to the requested shape (usually, (batch_size, num_feat, 1, seq_len))
    """
    _, num_feat, _, _ = shape
    return np.ones(shape) * mask.reshape((1, num_feat, 1, -1))

def get_joints_mask(join_names):
    joins_mask = np.array([joint_name in join_names for joint_name in HML_JOINT_NAMES])
    mask = np.concatenate(([False]*(1+2+1),
                                joins_mask[1:].repeat(3),
                                np.zeros_like(joins_mask[1:].repeat(6)),
                                np.zeros_like(joins_mask.repeat(3)),
                                [False] * 4))
    return mask

def get_batch_joint_mask(shape, joint_names):
    return expand_mask(get_joints_mask(joint_names), shape)

def get_in_between_mask(shape, lengths, prefix_end, suffix_end):
    mask = np.ones(shape)  # True means use gt motion
    for i, length in enumerate(lengths):
        start_idx, end_idx = int(prefix_end * length), int(suffix_end * length)
        mask[i, :, :, start_idx: end_idx] = 0  # do inpainting in those frames
    return mask

def get_prefix_mask(shape, prefix_length=20):
    _, num_feat, _, seq_len = shape
    prefix_mask = np.concatenate((np.ones((num_feat, prefix_length)), np.zeros((num_feat, seq_len - prefix_length))), axis=-1)
    return expand_mask(prefix_mask, shape)

def get_inpainting_mask(mask_name, shape, **kwargs):
    mask_names = mask_name.split(',')
    
    mask = np.zeros(shape)
    if 'in_between' in mask_names:
        mask = np.maximum(mask, get_in_between_mask(shape, **kwargs))
    
    if 'root' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_MASK, shape))
    
    if 'root_horizontal' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_HORIZONTAL_MASK, shape))

    if 'prefix' in mask_names:
        mask = np.maximum(mask, get_prefix_mask(shape, **kwargs))

    if 'upper_body' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_UPPER_BODY_MASK, shape))
    
    if 'lower_body' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_LOWER_BODY_MASK, shape))
    
    return np.maximum(mask, get_batch_joint_mask(shape, mask_names))



relax_hand_pose = [0.11167845129966736, 0.04289207234978676, -0.41644084453582764, 
                   0.10881128907203674, -0.06598565727472305, -0.756219744682312, 
                   -0.0963931530714035, -0.09091583639383316, -0.18845966458320618, 
                   -0.11809506267309189, 0.050943851470947266, -0.5295845866203308, 
                   -0.14369848370552063, 0.055241718888282776, -0.704857349395752,
                   -0.019182899966835976, -0.0923367589712143, -0.3379131853580475, 
                   -0.45703303813934326, -0.1962839663028717, -0.6254575848579407, 
                   -0.21465237438678741, -0.06599827855825424, -0.5068942308425903, 
                   -0.36972442269325256, -0.0603446289896965, -0.07949023693799973, 
                   -0.14186954498291016, -0.08585254102945328, -0.6355276107788086,
                   -0.3033415675163269, -0.05788097903132439, -0.6313892006874084, 
                   -0.17612087726593018, -0.13209305703639984, -0.3733545243740082, 
                   0.850964367389679, 0.2769227623939514, -0.09154807031154633, 
                   -0.4998386800289154, 0.026556432247161865, 0.052880801260471344, 
                   0.5355585217475891, 0.045960985124111176, -0.27735769748687744, 
                   0.11167845129966736, -0.04289207234978676, 0.41644084453582764, 
                   0.10881128907203674, 0.06598565727472305, 0.756219744682312, 
                   -0.0963931530714035, 0.09091583639383316, 0.18845966458320618, 
                   -0.11809506267309189, -0.050943851470947266, 0.5295845866203308, 
                   -0.14369848370552063, -0.055241718888282776, 0.704857349395752, 
                   -0.019182899966835976, 0.0923367589712143, 0.3379131853580475, 
                   -0.45703303813934326, 0.1962839663028717, 0.6254575848579407, 
                   -0.21465237438678741, 0.06599827855825424, 0.5068942308425903, 
                   -0.36972442269325256, 0.0603446289896965, 0.07949023693799973, 
                   -0.14186954498291016, 0.08585254102945328, 0.6355276107788086, 
                   -0.3033415675163269, 0.05788097903132439, 0.6313892006874084, 
                   -0.17612087726593018, 0.13209305703639984, 0.3733545243740082, 
                   0.850964367389679, -0.2769227623939514, 0.09154807031154633, 
                   -0.4998386800289154, -0.026556432247161865, -0.052880801260471344, 
                   0.5355585217475891, -0.045960985124111176, 0.27735769748687744]

t2m_adj_list = [
    [1, 2, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [1, 7],
    [2, 8],
    [3, 9],
    [4, 10],
    [5, 11],
    [6, 12, 13, 14],
    [7],
    [8],
    [9, 15],
    [9, 16],
    [9, 17],
    [12],
    [13, 18],
    [14, 19],
    [16, 20],
    [17, 21],
    [18],
    [19],
]