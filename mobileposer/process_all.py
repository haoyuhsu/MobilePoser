import os
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import glob

from mobileposer.articulate.model import ParametricModel
from mobileposer.articulate import math
from mobileposer.config import paths, datasets


# specify target FPS
TARGET_FPS = 30

# left wrist, right wrist, left thigh, right thigh, head, pelvis
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
body_model = ParametricModel(paths.smpl_file)


def _syn_acc(v, smooth_n=4):
    """Synthesize accelerations from vertex positions."""
    mid = smooth_n // 2
    scale_factor = TARGET_FPS ** 2 

    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * scale_factor for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))

    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * scale_factor / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def process_all(split='train', dataset_name='custom', data_root='/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/data/final_data_per_sequence_smpl/motion_data', num_splits=1):
    """Preprocess customized dataset with SMPL parameters.
    
    Args:
        split: 'train' or 'test'
        dataset_name: Dataset postfix name (e.g., 'humanml', 'fit3d', 'interx')
        data_root: Root directory containing train/test subdirectories
        num_splits: Number of splits to divide the data into
    """
    
    def _foot_ground_probs(joint):
        """Compute foot-ground contact probabilities."""
        dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
        dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
        lfoot_contact = (dist_lfeet < 0.008).int()
        rfoot_contact = (dist_rfeet < 0.008).int()
        lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
        rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
        return torch.stack((lfoot_contact, rfoot_contact), dim=1)

    data_dir = os.path.join(data_root, split)
    
    if not os.path.exists(data_dir):
        print(f"{dataset_name} data directory not found: {data_dir}")
        return
    
    # Filter files by dataset name - files contain the dataset name in their filename
    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl') and dataset_name in f])
    
    total_files = len(file_list)
    
    if total_files == 0:
        print(f"No files found for dataset '{dataset_name}' in {data_dir}")
        return
    
    files_per_split = total_files // num_splits if num_splits > 1 else total_files

    print(f'Processing {dataset_name} {split} split: {total_files} files into {num_splits} splits (~{files_per_split} files each)')
    
    for split_idx in range(num_splits):

        start_idx = split_idx * files_per_split
        if split_idx == num_splits - 1:
            # Last split gets remaining files
            end_idx = total_files
        else:
            end_idx = (split_idx + 1) * files_per_split
        
        split_file_list = file_list[start_idx:end_idx]

        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, out_contact, out_fname, out_vpos = [], [], [], [], [], [], [], [], []

        print(f'\nProcessing split {split_idx:03d} ({len(split_file_list)} files)')
        for pkl_file in tqdm(split_file_list):
            file_path = os.path.join(data_dir, pkl_file)
            try:
                data = pickle.load(open(file_path, 'rb'))
            except:
                print(f'Failed to load {pkl_file}')
                continue
            
            smpl_params = data['smpl_params']
            
            # Extract SMPL parameters
            global_orient = torch.from_numpy(smpl_params['global_orient']).float()  # (N, 3)
            body_pose = torch.from_numpy(smpl_params['body_pose']).float()  # (N, 23, 3)
            transl = torch.from_numpy(smpl_params['transl']).float()  # (N, 3)
            
            # Use zero shape (default body shape)
            shape = torch.zeros(10)  # (10,)
            
            # Combine global_orient and body_pose to get full pose (N, 24, 3)
            pose_aa = torch.cat([global_orient.unsqueeze(1), body_pose], dim=1)  # (N, 24, 3)
            
            seq_len = pose_aa.shape[0]
            
            # Skip sequences that are too short
            if seq_len <= 12:
                print(f'\tDiscard {pkl_file} with length {seq_len}')
                continue
            
            # Convert to rotation matrices for forward kinematics
            p = math.axis_angle_to_rotation_matrix(pose_aa).view(-1, 24, 3, 3)
            
            # Forward kinematics to get joints and vertices
            grot, joint, vert = body_model.forward_kinematics(p, shape, transl, calc_mesh=True)
            
            # Synthesize IMU accelerations from vertices with smoothing
            vacc = _syn_acc(vert[:, vi_mask])  # N, 6, 3
            
            # Extract virtual IMU orientations
            vrot = grot[:, ji_mask]  # N, 6, 3, 3
            
            # Store vertex positions for runtime IMU simulation
            vpos = vert[:, vi_mask]  # N, 6, 3
            
            # Compute foot contact labels
            contact = _foot_ground_probs(joint)  # N, 2
            
            # Store outputs in MobilePoser format
            out_pose.append(p.clone())  # N, 24, 3, 3
            out_tran.append(transl.clone())  # N, 3
            out_shape.append(shape.clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            out_vacc.append(vacc)  # N, 6, 3
            out_vrot.append(vrot)  # N, 6, 3, 3
            out_contact.append(contact)  # N, 2
            out_fname.append(pkl_file.replace('.pkl', ''))  # Store filename without extension
            out_vpos.append(vpos)  # N, 6, 3 - vertex positions for IMU simulation
        
        print(f'Saving {len(out_pose)} sequences...')
        
        # Save in MobilePoser format (single .pt file with dict)
        data = {
            'joint': out_joint,
            'pose': out_pose,
            'shape': out_shape,
            'tran': out_tran,
            'acc': out_vacc,
            'ori': out_vrot,
            'contact': out_contact,
            'fname': out_fname,
            'vpos': out_vpos  # vertex positions for IMU simulation
        }
        
        # Determine save directory (test split goes to eval subdirectory)
        save_dir = paths.processed_datasets / 'eval' if split == 'test' else paths.processed_datasets
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save with split index suffix
        if num_splits > 1:
            data_path = save_dir / f"{dataset_name}_{split}_{split_idx:03d}.pt"
        else:
            data_path = save_dir / f"{dataset_name}_{split}.pt"
        
        torch.save(data, data_path)
        print(f"Processed {dataset_name} {split} dataset saved at: {data_path}")

        import gc; gc.collect()  # Clean up memory after each split

    print(f'\nAll {num_splits} splits of {dataset_name} {split} dataset saved')
    


def create_directories():
    paths.processed_datasets.mkdir(exist_ok=True, parents=True)
    paths.eval_dir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="humanml", type=str, help="Name of the dataset to process (e.g., 'humanml', 'lingo')")
    args = parser.parse_args()

    # create dataset directories
    create_directories()

    process_all(split='train', dataset_name=args.dataset, num_splits=1)
    process_all(split='test', dataset_name=args.dataset, num_splits=1)
