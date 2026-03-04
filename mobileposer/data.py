import math
import os
import pickle
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
sys.path.append('/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/code/imu-human-mllm/imu_synthesis')
from get_imu_readings import simulate_imu_readings

import mobileposer.articulate as art
from mobileposer.config import *
from mobileposer.utils import *
from mobileposer.helpers import *


class PoseDataset(Dataset):
    def __init__(self, fold: str='train', evaluate: str=None, finetune: str=None, dataset_source: str='lingo', combo: str=None):
        super().__init__()
        self.fold = fold
        self.evaluate = evaluate
        self.finetune = finetune
        self.dataset_source = dataset_source
        self.combo = combo  # Fixed combo for evaluation, None means random sampling during training
        self.bodymodel = art.model.create_body_model()
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
            split_pattern = f"{self.dataset_source}_train.pt"
            split_files = sorted([x.name for x in data_folder.glob(split_pattern)])
            
            if split_files:
                # Multiple split files found, load all of them
                return split_files
            elif self.dataset_source in datasets.train_datasets:
                print(f"Load {self.dataset_source} training data")
                data_files = datasets.train_datasets[self.dataset_source]
                return data_files if isinstance(data_files, list) else [data_files]
            else:
                # Fallback to loading all files in the folder
                print(f"No specific training files found for {self.dataset_source}, loading all files in {data_folder}.")
                return [x.name for x in data_folder.iterdir() if not x.is_dir()]

    def _get_test_files(self):
        test_key = self.evaluate if self.evaluate else self.dataset_source
        data_files = datasets.test_datasets[test_key]
        return data_files if isinstance(data_files, list) else [data_files]

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
                
                # save memory
                del file_data
                import gc; gc.collect()

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
            if self.evaluate:
                a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(   # use augmentation-free simulation for evaluation
                    vpos_full, 
                    ori_full, 
                    fps=datasets.fps,
                    noise_raw_traj=False,
                    noise_syn_imu=False,
                    noise_est_orient=False,
                    skip_ESKF=True,
                    device='cpu'
                )
            else:
                # a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(   # use noisy simulation for training
                #     vpos_full, 
                #     ori_full, 
                #     fps=datasets.fps,
                #     noise_raw_traj=True,
                #     noise_syn_imu=True,
                #     noise_est_orient=True,
                #     skip_ESKF=True,
                #     device='cpu'
                # )
                a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(   # use augmentation-free simulation for training
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
            if self.combo:
                combo_name = self.combo
                combo_indices = amass.combos[self.combo]
            else:
                combo_name, combo_indices = 'global', [0, 1, 2, 3, 4]     # use global combo as default
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


# ---------------------------------------------------------------------------
# SMPL-X IMU reorder: SMPL-X 6-IMU order → MobilePoser/IMUPoser 6-IMU order
#   SMPL-X:    [left_hip(0), right_hip(1), left_ear(2), right_ear(3), left_elbow(4), right_elbow(5)]
#   IMUPoser:  [lw(0),       rw(1),        lp(2),       rp(3),        h(4),          extra(5)]
# Semantic mapping:
#   left_elbow(4)  → slot 0 (left arm / lw)
#   right_elbow(5) → slot 1 (right arm / rw)
#   left_hip(0)    → slot 2 (left pocket / lp)
#   right_hip(1)   → slot 3 (right pocket / rp)
#   left_ear(2)    → slot 4 (head / h)
#   right_ear(3)   → slot 5 (extra – dropped for 5-IMU model input)
# ---------------------------------------------------------------------------
SMPLX_TO_IMUPOSER_6 = [4, 5, 0, 1, 2, 3]

MIN_FRAMES_SMPLX = 4


class SMPLXPoseDataset(Dataset):
    """
    Dataset that pre-loads SMPL-X per-sequence .pkl files and preprocesses
    everything (FK, IMU simulation, velocity, foot contact) in __init__.

    __getitem__ only does combo masking + r6d conversion → fast iteration.

    Each .pkl contains:
      - motion_data_smpl85: (N, 85) = transl(3) + pose(72) + betas(10)
      - imu_traj:           (N, 6, 6) = 6 IMUs × (rot_aa(3) + pos(3))

    Produces the **same output format** as PoseDataset so it plugs into
    the existing training loop and pad_seq collate function.
    """

    def __init__(self, fold: str = 'train', evaluate: str = None,
                 finetune: str = None, dataset_source: str = 'lingo',
                 combo: str = None):
        super().__init__()
        self.fold = fold
        self.evaluate = evaluate
        self.finetune = finetune
        self.dataset_source = dataset_source
        self.combo = combo
        self.combos = list(amass.combos.items())
        self.num_pred_joints = len(amass.pred_joints_set)

        # Body model for FK (24-joint interface, SMPL-X under the hood)
        bodymodel = art.model.create_body_model(device=torch.device('cpu'))

        # Locate .pkl directory
        data_dir = Path(paths.smplx_data_path)
        if fold == 'test' or evaluate:
            data_dir = data_dir / 'test'
        else:
            data_dir = data_dir / 'train'

        if not data_dir.exists():
            raise FileNotFoundError(f"SMPL-X data dir not found: {data_dir}")

        # Collect filenames (optionally filter by dataset_source)
        fnames = sorted(f for f in os.listdir(data_dir) if f.endswith('.pkl'))
        if dataset_source:
            valid = [s for s in dataset_source.split(',')]
            fnames = [f for f in fnames if any(v in f for v in valid)]
        print(f"SMPLXPoseDataset [{fold}]: Found {len(fnames)} .pkl files in {data_dir} for dataset source '{dataset_source}'.")

        window_length = datasets.window_length if fold == 'train' else None

        # Preprocess all sequences up-front
        self.data = self._preprocess_all(data_dir, fnames, window_length, bodymodel)
        print(f"SMPLXPoseDataset [{fold}]: {len(self.data['acc'])} sequences preprocessed")

    # ---- preprocessing (runs once in __init__) --------------------------
    def _preprocess_all(self, data_dir, fnames, window_length, bodymodel):
        """Load every pkl, run FK + IMU sim, store results in lists."""
        keys = ['acc', 'ori', 'pose_outputs', 'joint_outputs',
                'tran_outputs', 'vel_outputs', 'foot_outputs', 'fnames']
        data = {k: [] for k in keys}

        for fn in tqdm(fnames, desc=f"Preprocessing SMPL-X [{self.fold}]"):
            try:
                self._preprocess_one(data_dir / fn, fn, window_length,
                                     bodymodel, data)
            except Exception as e:
                print(f"  Skipping {fn}: {e}")

        return data

    def _preprocess_one(self, fpath, fn, window_length, bodymodel, data):
        with open(fpath, 'rb') as f:
            raw = pickle.load(f)

        smpl85 = torch.from_numpy(raw['motion_data_smpl85']).float()
        imu_traj = torch.from_numpy(raw['imu_traj']).float()
        N = smpl85.shape[0]

        # pad short sequences
        if N < MIN_FRAMES_SMPLX:
            pad = MIN_FRAMES_SMPLX - N
            smpl85 = torch.cat([smpl85, smpl85[-1:].expand(pad, -1)], 0)
            imu_traj = torch.cat([imu_traj, imu_traj[-1:].expand(pad, -1, -1)], 0)
            N = MIN_FRAMES_SMPLX

        # truncate for training
        if window_length is not None:
            N = min(window_length, N)
            smpl85 = smpl85[:N]
            imu_traj = imu_traj[:N]

        # ---- parse smpl85 ----
        transl = smpl85[:, :3]
        pose_aa = smpl85[:, 3:75].reshape(N, 24, 3)

        # ---- FK ----
        pose_rotmat = art.math.axis_angle_to_rotation_matrix(
            pose_aa.reshape(-1, 3)
        ).reshape(N, 24, 3, 3)

        with torch.no_grad():
            pose_global, joint = bodymodel.forward_kinematics(
                pose=pose_rotmat.view(N, -1)
            )
        pose_global = pose_global.view(N, 24, 3, 3)
        joint = joint.view(N, 24, 3)

        # global rotation for training, local for eval
        pose_out = pose_rotmat if self.evaluate else pose_global

        # ---- IMU simulation ----
        imu_rot_aa = imu_traj[:, :, :3]
        imu_pos = imu_traj[:, :, 3:6]
        imu_rot = art.math.axis_angle_to_rotation_matrix(
            imu_rot_aa.reshape(-1, 3)
        ).reshape(N, 6, 3, 3)

        imu_pos = imu_pos[:, SMPLX_TO_IMUPOSER_6]
        imu_rot = imu_rot[:, SMPLX_TO_IMUPOSER_6]

        a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(
            imu_pos, imu_rot,
            fps=datasets.fps,
            noise_raw_traj=False, noise_syn_imu=False,
            noise_est_orient=False, skip_ESKF=True,
            device='cpu',
        )

        acc = a_sim[:, :5] / amass.acc_scale   # (N, 5, 3)
        ori = R_sim[:, :5]                      # (N, 5, 3, 3)

        # ---- velocity + foot contact (training only) ----
        if not (self.evaluate or self.finetune):
            root_vel = torch.cat([torch.zeros(1, 3), transl[1:] - transl[:-1]], 0)
            vel = torch.cat([torch.zeros(1, 24, 3), torch.diff(joint, dim=0)], 0)
            vel[:, 0] = root_vel
            data['vel_outputs'].append(vel * (datasets.fps / amass.vel_scale))
            data['foot_outputs'].append(self._foot_ground_probs(joint))

        data['acc'].append(acc)
        data['ori'].append(ori)
        data['pose_outputs'].append(pose_out)
        data['joint_outputs'].append(joint)
        data['tran_outputs'].append(transl)
        data['fnames'].append(fn.replace('.pkl', ''))

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _foot_ground_probs(joint):
        """Foot-ground contact from joint displacement (same heuristic as process.py)."""
        dist_l = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
        dist_r = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
        lc = torch.cat([torch.zeros(1), (dist_l < 0.008).float()])
        rc = torch.cat([torch.zeros(1), (dist_r < 0.008).float()])
        return torch.stack([lc, rc], dim=1)  # (N, 2)

    def _apply_combo_mask(self, acc, ori, combo_indices):
        combo_acc = torch.zeros_like(acc)
        combo_ori = torch.zeros_like(ori)
        combo_acc[:, combo_indices] = acc[:, combo_indices]
        combo_ori[:, combo_indices] = ori[:, combo_indices]
        return torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1).float()

    # ---- __getitem__ (fast: only combo mask + r6d) ----------------------
    def __len__(self):
        return len(self.data['acc'])

    def __getitem__(self, idx):
        acc = self.data['acc'][idx].float()
        ori = self.data['ori'][idx].float()
        joint = self.data['joint_outputs'][idx].float()
        transl = self.data['tran_outputs'][idx].float()
        fname = self.data['fnames'][idx]

        # ---- combo selection ----
        if self.evaluate:
            if self.combo:
                combo_name = self.combo
                combo_indices = amass.combos[self.combo]
            else:
                combo_name, combo_indices = 'global', [0, 1, 2, 3, 4]
        else:
            combo_name, combo_indices = random.choice(self.combos)

        imu = self._apply_combo_mask(acc, ori, combo_indices)
        full_fname = f"{combo_name}/{fname}"

        # ---- r6d pose output ----
        pose_r6d = art.math.rotation_matrix_to_r6d(
            self.data['pose_outputs'][idx]
        ).reshape(-1, self.num_pred_joints, 6
        )[:, amass.pred_joints_set].reshape(-1, 6 * self.num_pred_joints)

        if self.evaluate or self.finetune:
            return imu, pose_r6d, joint, transl, full_fname

        vel = self.data['vel_outputs'][idx].float()
        contact = self.data['foot_outputs'][idx].float()

        return imu, pose_r6d, joint, transl, vel, contact, full_fname


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

    def _make_dataset(self, fold, evaluate=None):
        """Instantiate the right dataset class based on body_model_config."""
        if body_model_config.model_type == 'smplx':
            return SMPLXPoseDataset(
                fold=fold,
                evaluate=evaluate,
                finetune=self.finetune,
                dataset_source=self.dataset_source,
            )
        else:
            return PoseDataset(
                fold=fold,
                evaluate=evaluate,
                finetune=self.finetune,
                dataset_source=self.dataset_source,
            )

    def setup(self, stage: str):
        if stage == 'fit':
            dataset = self._make_dataset(fold='train')
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        elif stage == 'test':
            self.test_dataset = self._make_dataset(fold='test')

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
