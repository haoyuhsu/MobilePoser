import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from typing import List
import random
import lightning as L
from tqdm import tqdm 
import sys
from pathlib import Path

# Add imu_synthesis directory to path to import simulation utilities
sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/imu-human-mllm/imu_synthesis')
from get_imu_readings import simulate_imu_readings

import mobileposer.articulate as art
from mobileposer.config import *
from mobileposer.utils import *
from mobileposer.helpers import *


class PoseDataset(Dataset):
    def __init__(self, fold: str='train', evaluate: str=None, finetune: str=None, dataset_source: str='lingo'):
        super().__init__()
        self.fold = fold
        self.evaluate = evaluate
        self.finetune = finetune
        self.dataset_source = dataset_source
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
        self.combos = list(amass.combos.items())   # use IMUPoser combos
        self.data = self._prepare_dataset()

    def _get_data_files(self, data_folder):
        if self.fold == 'train':
            return self._get_train_files(data_folder)
        elif self.fold == 'test':
            return self._get_test_files()
        else:
            raise ValueError(f"Unknown data fold: {self.fold}.")

    def _get_train_files(self, data_folder):
        if self.finetune:
            return [datasets.finetune_datasets[self.finetune]]
        else:
            # Check for multiple split files (e.g., humanml_train_000.pt, humanml_train_001.pt)
            split_pattern = f"{self.dataset_source}_train_*.pt"
            split_files = sorted([x.name for x in data_folder.glob(split_pattern)])
            
            if split_files:
                # Multiple split files found, load all of them
                return split_files
            elif self.dataset_source in datasets.train_datasets:
                # Single file mapping
                return [datasets.train_datasets[self.dataset_source]]
            else:
                # Fallback to loading all files in the folder
                return [x.name for x in data_folder.iterdir() if not x.is_dir()]

    def _get_test_files(self):
        test_key = self.evaluate if self.evaluate else self.dataset_source
        return [datasets.test_datasets[test_key]]

    def _prepare_dataset(self):
        """Load raw data without combo-specific processing."""
        data_folder = paths.processed_datasets / ('eval' if (self.finetune or self.evaluate or self.fold == 'test') else '')
        data_files = self._get_data_files(data_folder)
        
        # Store raw IMU data (acc, ori) and outputs without combo expansion
        data = {key: [] for key in ['acc', 'ori', 'pose_outputs', 'joint_outputs', 'tran_outputs', 'vel_outputs', 'foot_outputs', 'fnames', 'vpos']}
        
        for data_file in tqdm(data_files):
            try:
                file_data = torch.load(data_folder / data_file)
                self._process_file_data(file_data, data)
            except Exception as e:
                print(f"Error processing {data_file}: {e}.")
        return data

    def _process_file_data(self, file_data, data):
        """Process file data without combo expansion."""
        accs, oris, poses, trans = file_data['acc'], file_data['ori'], file_data['pose'], file_data['tran']
        joints = file_data.get('joint', [None] * len(poses))
        foots = file_data.get('contact', [None] * len(poses))
        fnames = file_data.get('fname', [f"sample_{i}" for i in range(len(poses))])
        vpositions = file_data.get('vpos', [None] * len(poses))  # vertex positions for IMU simulation
        
        # Minimum frames required for IMU simulation
        MIN_FRAMES = 4

        for acc, ori, pose, tran, joint, foot, fname, vpos in zip(accs, oris, poses, trans, joints, foots, fnames, vpositions):
            
            # Skip sequences that are too short
            if len(acc) < MIN_FRAMES:
                continue
            
            # Normalize acc and ori
            acc, ori = acc[:, :6] / amass.acc_scale, ori[:, :6]    # use all 6 IMUs, filter later
            
            # Convert pose to global rotation
            pose_global, joint = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216))
            pose = pose if self.evaluate else pose_global.view(-1, 24, 3, 3)
            joint = joint.view(-1, 24, 3)
            
            # During evaluation/testing: use entire sequence
            # During training: use first K frames (window_length)
            if self.evaluate:
                # Use entire sequence for evaluation
                data['acc'].append(acc)
                data['ori'].append(ori)
                data['pose_outputs'].append(pose)
                data['joint_outputs'].append(joint)
                data['tran_outputs'].append(tran)
                data['fnames'].append(fname)
                data['vpos'].append(vpos)
            else:
                # Training: use only first window_length frames
                data_len = min(datasets.window_length, len(acc))
                
                data['acc'].append(acc[:data_len])
                data['ori'].append(ori[:data_len])
                data['pose_outputs'].append(pose[:data_len])
                data['joint_outputs'].append(joint[:data_len])
                data['tran_outputs'].append(tran[:data_len])
                data['fnames'].append(fname)
                data['vpos'].append(vpos[:data_len] if vpos is not None else None)
                
                # Process translation data if needed
                if not (self.evaluate or self.finetune):
                    tran_chunk = tran[:data_len]
                    joint_chunk = joint[:data_len]
                    foot_chunk = foot[:data_len]
                    
                    root_vel = torch.cat((torch.zeros(1, 3), tran_chunk[1:] - tran_chunk[:-1]))
                    vel = torch.cat((torch.zeros(1, 24, 3), torch.diff(joint_chunk, dim=0)))
                    vel[:, 0] = root_vel
                    data['vel_outputs'].append(vel * (datasets.fps / amass.vel_scale))
                    data['foot_outputs'].append(foot_chunk)

    def _apply_combo_mask(self, acc, ori, combo_indices):
        """Apply combo mask to acceleration and orientation data."""
        combo_acc = torch.zeros_like(acc)
        combo_ori = torch.zeros_like(ori)
        combo_acc[:, combo_indices] = acc[:, combo_indices]
        combo_ori[:, combo_indices] = ori[:, combo_indices]
        imu_input = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)  # [N, 60]
        return imu_input

    def __getitem__(self, idx):
        
        # Get raw data
        joint = self.data['joint_outputs'][idx].float()
        tran = self.data['tran_outputs'][idx].float()
        fname = self.data['fnames'][idx]
        
        # Get pre-computed ori (joint rotations) - always available
        ori = self.data['ori'][idx].float()  # N, 6, 3, 3
        
        # Check if we have vertex positions for runtime IMU simulation
        vpos = self.data['vpos'][idx]
        
        if vpos is not None:
            # Runtime IMU simulation with noise
            vpos_full = vpos.float()  # N, 6, 3
            ori_full = ori.float()    # N, 6, 3, 3
            
            # Simulate IMU readings for ALL 6 IMUs first
            # Note: simulate_imu_readings expects (N, num_imus, 3) for positions and (N, num_imus, 3, 3) for rotations
            a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(
                vpos_full, 
                ori_full, 
                fps=datasets.fps,
                noise_raw_traj=False,
                noise_syn_imu=False,
                noise_est_orient=False,
                skip_ESKF=True,
                device='cpu'
            )
            
            # Normalize acceleration
            acc = a_sim[:, :5] / amass.acc_scale  # N, 5, 3
            ori = R_sim[:, :5]  # N, 5, 3, 3
        else:
            # Use pre-computed accelerations and orientations
            # acc = self.data['acc'][idx][:, :5].float()
            # ori = self.data['ori'][idx][:, :5].float()
            raise ValueError(f"Vertex positions not available for {fname}, cannot simulate IMU readings. Please ensure vpos is included in the dataset for runtime simulation.")
        
        if self.evaluate:
            combo_name, combo_indices = 'global', [0, 1, 2, 3, 4]     # use global combo for consistent evaluation
        else:
            # Sample a random combo at runtime
            combo_name, combo_indices = random.choice(self.combos)

        # Apply combo mask
        imu = self._apply_combo_mask(acc, ori, combo_indices)
        
        # Prepare full filename with combo
        full_fname = f"{combo_name}/{fname}"
        
        # Prepare pose output
        num_pred_joints = len(amass.pred_joints_set)
        pose = art.math.rotation_matrix_to_r6d(self.data['pose_outputs'][idx]).reshape(-1, num_pred_joints, 6)[:, amass.pred_joints_set].reshape(-1, 6*num_pred_joints)

        if self.evaluate or self.finetune:
            return imu, pose, joint, tran, full_fname

        vel = self.data['vel_outputs'][idx].float()
        contact = self.data['foot_outputs'][idx].float()

        return imu, pose, joint, tran, vel, contact, full_fname

    def __len__(self):
        return len(self.data['acc'])


def pad_seq(batch):
    """Pad sequences to same length for RNN."""
    def _pad(sequence):
        padded = nn.utils.rnn.pad_sequence(sequence, batch_first=True)
        lengths = [seq.shape[0] for seq in sequence]
        return padded, lengths

    inputs, poses, joints, trans = zip(*[(item[0], item[1], item[2], item[3]) for item in batch])
    inputs, input_lengths = _pad(inputs)
    poses, pose_lengths = _pad(poses)
    joints, joint_lengths = _pad(joints)
    trans, tran_lengths = _pad(trans)
    
    outputs = {'poses': poses, 'joints': joints, 'trans': trans}
    output_lengths = {'poses': pose_lengths, 'joints': joint_lengths, 'trans': tran_lengths}

    if len(batch[0]) > 5: # include velocity and foot contact, if available
        vels, foots = zip(*[(item[4], item[5]) for item in batch])

        # foot contact 
        foot_contacts, foot_contact_lengths = _pad(foots)
        outputs['foot_contacts'], output_lengths['foot_contacts'] = foot_contacts, foot_contact_lengths

        # root velocities
        vels, vel_lengths = _pad(vels)
        outputs['vels'], output_lengths['vels'] = vels, vel_lengths

    return (inputs, input_lengths), (outputs, output_lengths)


class PoseDataModule(L.LightningDataModule):
    def __init__(self, finetune: str = None, dataset_source: str = 'lingo'):
        super().__init__()
        self.finetune = finetune
        self.dataset_source = dataset_source
        self.hypers = finetune_hypers if self.finetune else train_hypers

    def setup(self, stage: str):
        if stage == 'fit':
            dataset = PoseDataset(fold='train', finetune=self.finetune, dataset_source=self.dataset_source)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        elif stage == 'test':
            self.test_dataset = PoseDataset(fold='test', finetune=self.finetune, dataset_source=self.dataset_source)

    def _dataloader(self, dataset):
        return DataLoader(
            dataset, 
            batch_size=self.hypers.batch_size, 
            collate_fn=pad_seq, 
            num_workers=self.hypers.num_workers, 
            shuffle=True, 
            drop_last=True
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)
