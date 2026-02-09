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
        data_folder = paths.processed_datasets / ('eval' if (self.finetune or self.evaluate) else '')
        data_files = self._get_data_files(data_folder)
        
        # Store raw IMU data (acc, ori) and outputs without combo expansion
        data = {key: [] for key in ['acc', 'ori', 'pose_outputs', 'joint_outputs', 'tran_outputs', 'vel_outputs', 'foot_outputs', 'fnames']}
        
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

        for acc, ori, pose, tran, joint, foot, fname in zip(accs, oris, poses, trans, joints, foots, fnames):
            # Normalize acc and ori
            acc, ori = acc[:, :5] / amass.acc_scale, ori[:, :5]
            
            # Convert pose to global rotation
            pose_global, joint = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216))
            pose = pose if self.evaluate else pose_global.view(-1, 24, 3, 3)
            joint = joint.view(-1, 24, 3)
            
            # Split data into windows
            data_len = len(acc) if self.evaluate else datasets.window_length
            
            # Store raw acc and ori (not combo-masked yet)
            data['acc'].extend(torch.split(acc, data_len))
            data['ori'].extend(torch.split(ori, data_len))
            data['pose_outputs'].extend(torch.split(pose, data_len))
            data['joint_outputs'].extend(torch.split(joint, data_len))
            data['tran_outputs'].extend(torch.split(tran, data_len))
            data['fnames'].extend([fname] * len(torch.split(acc, data_len)))
            
            # Process translation data if needed
            if not (self.evaluate or self.finetune):
                root_vel = torch.cat((torch.zeros(1, 3), tran[1:] - tran[:-1]))
                vel = torch.cat((torch.zeros(1, 24, 3), torch.diff(joint, dim=0)))
                vel[:, 0] = root_vel
                data['vel_outputs'].extend(torch.split(vel * (datasets.fps / amass.vel_scale), data_len))
                data['foot_outputs'].extend(torch.split(foot, data_len))

    def _apply_combo_mask(self, acc, ori, combo_indices):
        """Apply combo mask to acceleration and orientation data."""
        combo_acc = torch.zeros_like(acc)
        combo_ori = torch.zeros_like(ori)
        combo_acc[:, combo_indices] = acc[:, combo_indices]
        combo_ori[:, combo_indices] = ori[:, combo_indices]
        imu_input = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)  # [N, 60]
        return imu_input

    def __getitem__(self, idx):

        if self.evaluate:
            combo_name, combo_indices = 'global', [0, 1, 2, 3, 4]     # use global combo for consistent evaluation
        else:
            # Sample a random combo at runtime
            combo_name, combo_indices = random.choice(self.combos)
        
        # Get raw data
        acc = self.data['acc'][idx].float()
        ori = self.data['ori'][idx].float()
        joint = self.data['joint_outputs'][idx].float()
        tran = self.data['tran_outputs'][idx].float()
        fname = self.data['fnames'][idx]
        
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
