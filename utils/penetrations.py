import numpy as np
from scipy.spatial import cKDTree
import trimesh
from tqdm import tqdm
import torch

def compute_contact_penetration_fast(
    vertsA, vertsB, lengths, eps=0.05, pen_thresh=0.05, device=None
):
    """
    Fast batch contact + penetration evaluation with CUDA acceleration.

    Args:
        vertsA: (B, L, V, 3) numpy or torch
        vertsB: (B, L, V, 3)
        lengths: (B,) valid frames
        eps: contact distance threshold
        pen_thresh: penetration threshold (distance < pen_thresh counts as penetration)
        device: torch device ('cuda' or 'cpu'), auto-detected if None

    Returns:
        contact_score, penetration_score
    """
    # Convert to torch if needed
    is_torch = isinstance(vertsA, torch.Tensor)
    if not is_torch:
        vertsA = torch.from_numpy(vertsA).float()
        vertsB = torch.from_numpy(vertsB).float()
        lengths = torch.from_numpy(lengths).long()
    
    # Auto-detect device
    if device is None:
        device = vertsA.device if is_torch else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move to device
    vertsA = vertsA.to(device)
    vertsB = vertsB.to(device)
    lengths = lengths.to(device)

    B, L, V, _ = vertsA.shape
    total_contact = 0.0
    total_penetration = 0.0
    total_count = 0

    for b in range(B):
        T = int(lengths[b])
        if T == 0:
            continue
            
        # Get all frames for this batch
        VA = vertsA[b, :T]  # (T, V, 3)
        VB = vertsB[b, :T]  # (T, V, 3)
        
        # Compute pairwise distances: (T, V, V)
        # Using cdist for efficient batch computation
        dists_AB = torch.cdist(VA, VB, p=2)  # (T, V, V)
        dists_BA = dists_AB.transpose(-1, -2)  # (T, V, V)
        
        # Get minimum distances (A->B and B->A)
        min_dists_AB, _ = dists_AB.min(dim=-1)  # (T, V)
        min_dists_BA, _ = dists_BA.min(dim=-1)  # (T, V)
        
        # Compute contact and penetration
        contact_AB = (min_dists_AB < eps).float()
        penetration_AB = (min_dists_AB < pen_thresh).float()
        contact_BA = (min_dists_BA < eps).float()
        penetration_BA = (min_dists_BA < pen_thresh).float()
        
        # Accumulate
        total_contact += contact_AB.mean().item() + contact_BA.mean().item()
        total_penetration += penetration_AB.mean().item() + penetration_BA.mean().item()
        total_count += 2

    if total_count == 0:
        return 0.0, 0.0
    
    contact_score = total_contact / total_count
    penetration_score = total_penetration / total_count
    return contact_score, penetration_score

x_min, x_max = -5.0, 5.0
y_min, y_max = -5.0, 5.0
z_min, z_max = 0.0, 2.0 
pitch = 0.02
Nx = int(np.ceil((x_max - x_min) / pitch))
Ny = int(np.ceil((y_max - y_min) / pitch))
Nz = int(np.ceil((z_max - z_min) / pitch))
grid_shape = (Nx, Ny, Nz)

def mesh_to_fixed_voxel(mesh, pitch=pitch, solid=False):
    voxelized = mesh.voxelized(pitch)
    if solid:
        voxelized = voxelized.fill()

    voxel_matrix = np.zeros(grid_shape, dtype=np.uint8)

    coords_world = voxelized.points  # (N,3)

    mask = (
        (coords_world[:,0] >= x_min) & (coords_world[:,0] <= x_max) &
        (coords_world[:,1] >= y_min) & (coords_world[:,1] <= y_max) &
        (coords_world[:,2] >= z_min) & (coords_world[:,2] <= z_max)
    )
    coords_world = coords_world[mask]

    ix = np.floor((coords_world[:,0] - x_min)/pitch).astype(int)
    iy = np.floor((coords_world[:,1] - y_min)/pitch).astype(int)
    iz = np.floor((coords_world[:,2] - z_min)/pitch).astype(int)

    ix = np.clip(ix, 0, Nx-1)
    iy = np.clip(iy, 0, Ny-1)
    iz = np.clip(iz, 0, Nz-1)

    voxel_matrix[ix, iy, iz] = 1
    return voxel_matrix



def compute_voxel_overlap_fixed(VA, VB):
    """
    VA, VB: solid voxel grid (Nx, Ny, Nz)
    return: overlap ratio
    """
    intersection = np.logical_and(VA, VB).sum()
    min_volume = min(VA.sum(), VB.sum())
    return intersection / max(min_volume, 1e-8)


def contact_penetration_voxel_batch_fixed(vertsA, vertsB, lengths, faces, pitch=pitch, device=None):
    """
    Compute voxel-based contact + penetration for BxL mesh sequences.
    In contact: any overlap > 0 is considered contact.
    In penetration: returns the actual penetration volume in ml.
    Optimized with GPU acceleration for voxel operations.
    
    vertsA, vertsB: (B, L, V, 3) numpy or torch
    lengths: (B,) valid frames
    faces: (F, 3)
    device: torch device, auto-detected if None
    """
    # Convert to numpy for trimesh (trimesh requires numpy)
    is_torch = isinstance(vertsA, torch.Tensor)
    if is_torch:
        vertsA_np = vertsA.cpu().numpy()
        vertsB_np = vertsB.cpu().numpy()
        lengths_np = lengths.cpu().numpy()
        if device is None:
            device = vertsA.device
    else:
        vertsA_np = vertsA
        vertsB_np = vertsB
        lengths_np = lengths
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    faces_np = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else faces

    B, L, V, _ = vertsA_np.shape
    contact_values = []
    penetration_values = []

    # Each voxel volume = (0.02m)^3 = 0.000008 m³ = 8ml
    voxel_volume_ml = (pitch ** 3) * 1000  # Convert m³ to ml
    
    # For small matrices, numpy is faster. For large ones, use GPU.
    # Grid shape is (500, 500, 100) = 25M elements, so GPU should help
    use_gpu = device != 'cpu' and torch.cuda.is_available()
    
    # Process in batches to reduce overhead
    for b in range(B):
        T = int(lengths_np[b])
        if T == 0:
            continue
        
        # Batch process voxel matrices on GPU if available
        if use_gpu and T > 1:
            # Collect all voxel matrices first
            VA_contact_list = []
            VB_contact_list = []
            VA_pen_list = []
            VB_pen_list = []
            
            for t in range(T):
                meshA = trimesh.Trimesh(vertices=vertsA_np[b, t], faces=faces_np, process=False)
                meshB = trimesh.Trimesh(vertices=vertsB_np[b, t], faces=faces_np, process=False)
                
                VA_contact_list.append(mesh_to_fixed_voxel(meshA, pitch=pitch, solid=False))
                VB_contact_list.append(mesh_to_fixed_voxel(meshB, pitch=pitch, solid=False))
                VA_pen_list.append(mesh_to_fixed_voxel(meshA, pitch=pitch, solid=True))
                VB_pen_list.append(mesh_to_fixed_voxel(meshB, pitch=pitch, solid=True))
            
            # Batch convert to torch and compute on GPU
            VA_contact_batch = torch.stack([torch.from_numpy(v).to(device) for v in VA_contact_list])
            VB_contact_batch = torch.stack([torch.from_numpy(v).to(device) for v in VB_contact_list])
            VA_pen_batch = torch.stack([torch.from_numpy(v).to(device) for v in VA_pen_list])
            VB_pen_batch = torch.stack([torch.from_numpy(v).to(device) for v in VB_pen_list])
            
            # Compute overlaps in batch
            overlap_contact_batch = (VA_contact_batch & VB_contact_batch).sum(dim=(1, 2, 3)).cpu().numpy()
            overlap_pen_batch = (VA_pen_batch & VB_pen_batch).sum(dim=(1, 2, 3)).cpu().numpy()
            
            # Convert to contact/penetration values
            for t in range(T):
                contact_val = 1.0 if overlap_contact_batch[t] > 0 else 0.0
                contact_values.append(contact_val)
                penetration_val = overlap_pen_batch[t] * voxel_volume_ml
                penetration_values.append(penetration_val)
        else:
            # Process frame by frame (original method, faster for single frames or CPU)
            for t in range(T):
                meshA = trimesh.Trimesh(vertices=vertsA_np[b, t], faces=faces_np, process=False)
                meshB = trimesh.Trimesh(vertices=vertsB_np[b, t], faces=faces_np, process=False)

                # contact -> surface voxels (just presence of any overlap counts)
                VA_contact = mesh_to_fixed_voxel(meshA, pitch=pitch, solid=False)
                VB_contact = mesh_to_fixed_voxel(meshB, pitch=pitch, solid=False)
                
                if use_gpu:
                    VA_contact_t = torch.from_numpy(VA_contact).to(device)
                    VB_contact_t = torch.from_numpy(VB_contact).to(device)
                    overlap_contact = (VA_contact_t & VB_contact_t).sum().item()
                else:
                    overlap_contact = np.logical_and(VA_contact, VB_contact).sum()
                contact_val = 1.0 if overlap_contact > 0 else 0.0
                contact_values.append(contact_val)

                # penetration -> solid voxels (returns volume in ml)
                VA_pen = mesh_to_fixed_voxel(meshA, pitch=pitch, solid=True)
                VB_pen = mesh_to_fixed_voxel(meshB, pitch=pitch, solid=True)
                
                if use_gpu:
                    VA_pen_t = torch.from_numpy(VA_pen).to(device)
                    VB_pen_t = torch.from_numpy(VB_pen).to(device)
                    overlap_pen = (VA_pen_t & VB_pen_t).sum().item()
                else:
                    overlap_pen = np.logical_and(VA_pen, VB_pen).sum()
                penetration_val = overlap_pen * voxel_volume_ml  # Convert voxels to ml
                penetration_values.append(penetration_val)

    if len(contact_values) == 0:
        return 0.0, 0.0
    
    return np.mean(contact_values), np.mean(penetration_values)