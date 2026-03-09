"""
Finetune MobilePoser modules (joints, poser) on DIP-IMU / IMUPoser datasets
using pre-computed acc/ori from .pt files.

Usage examples:
  # Finetune on DIP-IMU
  python -m mobileposer.finetune_dipimu \
      --train-data data/processed_datasets/eval/dip_train.pt \
      --checkpoint-dir checkpoints/all_MotionGV_no_noise \
      --output-dir checkpoints/finetuned_dip \
      --epochs 15 --lr 5e-5 --batch-size 32

  # Finetune on IMUPoser
  python -m mobileposer.finetune_dipimu \
      --train-data data/processed_datasets/eval/imuposer_train.pt \
      --checkpoint-dir checkpoints/all_MotionGV_no_noise \
      --output-dir checkpoints/finetuned_imuposer \
      --epochs 15 --lr 5e-5 --batch-size 32

  # Finetune a single module
  python -m mobileposer.finetune_dipimu \
      --train-data data/processed_datasets/eval/dip_train.pt \
      --checkpoint-dir checkpoints/all_MotionGV_no_noise \
      --output-dir checkpoints/finetuned_dip \
      --module poser
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

import mobileposer.articulate as art
from mobileposer.config import amass, datasets, model_config, joint_set, finetune_hypers
from mobileposer.constants import MODULES
from mobileposer.utils.file_utils import get_best_checkpoint


torch.set_float32_matmul_precision('medium')


# ---------------------------------------------------------------------------
# Dataset: loads pre-computed acc/ori directly from .pt files (no vpos needed)
# ---------------------------------------------------------------------------
class PrecomputedIMUDataset(Dataset):
    """
    Dataset for DIP-IMU / IMUPoser .pt files that already contain
    pre-computed acc and ori (no vertex-position-based IMU simulation).
    """

    def __init__(self, pt_path: str, fold: str = 'train', combo: str = None):
        super().__init__()
        self.fold = fold
        self.combo = combo
        self.combos = list(amass.combos.items())
        self.bodymodel = art.model.create_body_model(device=torch.device('cpu'))

        raw = torch.load(pt_path, weights_only=False)
        self.data = self._prepare(raw, fold)

    def _prepare(self, raw, fold):
        accs = raw['acc']       # list of (N, 6, 3)
        oris = raw['ori']       # list of (N, 6, 3, 3)
        poses = raw['pose']     # list of (N, 24, 3, 3)
        trans = raw['tran']     # list of (N, 3)
        joints_raw = raw.get('joint', [None] * len(poses))
        fnames = raw.get('fname', [f"sample_{i}" for i in range(len(poses))])
        foots = raw.get('contact', [None] * len(poses))

        is_eval = (fold == 'test')
        window_length = datasets.window_length if not is_eval else None

        data = {k: [] for k in [
            'acc', 'ori', 'pose_outputs', 'joint_outputs',
            'tran_outputs', 'vel_outputs', 'foot_outputs', 'fnames'
        ]}

        for acc, ori, pose, tran, jnt, foot, fname in zip(
                accs, oris, poses, trans, joints_raw, foots, fnames):

            if len(acc) < 4:
                continue

            # Normalise acc; keep first 6 IMUs
            acc = acc[:, :6] / amass.acc_scale
            ori = ori[:, :6]

            # Forward kinematics → global rotations + joint positions
            pose_global, jnt = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216))
            pose_out = pose if is_eval else pose_global.view(-1, 24, 3, 3)
            jnt = jnt.view(-1, 24, 3)

            if is_eval:
                # full sequence for evaluation
                data['acc'].append(acc)
                data['ori'].append(ori)
                data['pose_outputs'].append(pose_out)
                data['joint_outputs'].append(jnt)
                data['tran_outputs'].append(tran)
                data['fnames'].append(fname)
            else:
                L = min(window_length, len(acc)) if window_length else len(acc)
                data['acc'].append(acc[:L])
                data['ori'].append(ori[:L])
                data['pose_outputs'].append(pose_out[:L])
                data['joint_outputs'].append(jnt[:L])
                data['tran_outputs'].append(tran[:L])
                data['fnames'].append(fname)

                # Compute velocity & foot contact for training
                root_vel = torch.cat([torch.zeros(1, 3), tran[1:L] - tran[:L-1]])
                vel = torch.cat([torch.zeros(1, 24, 3), torch.diff(jnt[:L], dim=0)])
                vel[:, 0] = root_vel
                data['vel_outputs'].append(vel * (datasets.fps / amass.vel_scale))

                if foot is not None and len(foot) >= L:
                    data['foot_outputs'].append(foot[:L])
                else:
                    # heuristic foot contact from joint displacement
                    dist_l = torch.norm(jnt[1:L, 10] - jnt[:L-1, 10], dim=1)
                    dist_r = torch.norm(jnt[1:L, 11] - jnt[:L-1, 11], dim=1)
                    lc = torch.cat([torch.zeros(1), (dist_l < 0.008).float()])
                    rc = torch.cat([torch.zeros(1), (dist_r < 0.008).float()])
                    data['foot_outputs'].append(torch.stack([lc, rc], dim=1))

        return data

    def _apply_combo_mask(self, acc, ori, combo_indices):
        combo_acc = torch.zeros_like(acc)
        combo_ori = torch.zeros_like(ori)
        combo_acc[:, combo_indices] = acc[:, combo_indices]
        combo_ori[:, combo_indices] = ori[:, combo_indices]
        return torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)

    def __len__(self):
        return len(self.data['acc'])

    def __getitem__(self, idx):
        acc = self.data['acc'][idx][:, :5].float()
        ori = self.data['ori'][idx][:, :5].float()
        joint = self.data['joint_outputs'][idx].float()
        tran = self.data['tran_outputs'][idx].float()
        fname = self.data['fnames'][idx]

        # Combo selection
        if self.fold == 'test':
            combo_name = self.combo if self.combo else 'global'
            combo_indices = amass.combos[combo_name]
        else:
            combo_name, combo_indices = random.choice(self.combos)

        imu = self._apply_combo_mask(acc, ori, combo_indices)
        full_fname = f"{combo_name}/{fname}"

        # r6d pose
        num_pred_joints = len(amass.pred_joints_set)
        pose = art.math.rotation_matrix_to_r6d(
            self.data['pose_outputs'][idx]
        ).reshape(-1, num_pred_joints, 6)[:, amass.pred_joints_set].reshape(-1, 6 * num_pred_joints)

        if self.fold == 'test':
            return imu, pose, joint, tran, full_fname

        vel = self.data['vel_outputs'][idx].float()
        contact = self.data['foot_outputs'][idx].float()
        return imu, pose, joint, tran, vel, contact, full_fname


# ---------------------------------------------------------------------------
# Collate (same as existing pad_seq)
# ---------------------------------------------------------------------------
def pad_seq(batch):
    def _pad(sequence):
        padded = nn.utils.rnn.pad_sequence(sequence, batch_first=True)
        lengths = [seq.shape[0] for seq in sequence]
        return padded, lengths

    inputs, poses, joints, trans = zip(*[(b[0], b[1], b[2], b[3]) for b in batch])
    inputs, input_lengths = _pad(inputs)
    poses, _ = _pad(poses)
    joints, _ = _pad(joints)
    trans, _ = _pad(trans)

    outputs = {'poses': poses, 'joints': joints, 'trans': trans}
    output_lengths = {}

    if len(batch[0]) > 5:
        vels, foots = zip(*[(b[4], b[5]) for b in batch])
        foot_contacts, _ = _pad(foots)
        vels, _ = _pad(vels)
        outputs['foot_contacts'] = foot_contacts
        outputs['vels'] = vels

    return (inputs, input_lengths), (outputs, output_lengths)


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class FinetuneDataModule(L.LightningDataModule):
    def __init__(self, train_pt: str, batch_size: int = 32, num_workers: int = 8):
        super().__init__()
        self.train_pt = train_pt
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == 'fit':
            full = PrecomputedIMUDataset(self.train_pt, fold='train')
            train_size = int(0.9 * len(full))
            val_size = len(full) - train_size
            self.train_ds, self.val_ds = random_split(full, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          collate_fn=pad_seq, num_workers=self.num_workers,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=min(self.batch_size, len(self.val_ds)),
                          collate_fn=pad_seq, num_workers=self.num_workers,
                          shuffle=False, drop_last=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training .pt file (e.g. data/processed_datasets/eval/dip_train.pt)')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory with pre-trained module checkpoints (e.g. checkpoints/all_MotionGV_no_noise)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save finetuned checkpoints')
    parser.add_argument('--module', type=str, default=None, choices=['joints', 'poser'],
                        help='Finetune a single module. If omitted, finetunes joints then poser.')
    parser.add_argument('--epochs', type=int, default=finetune_hypers.num_epochs)
    parser.add_argument('--lr', type=float, default=finetune_hypers.lr)
    parser.add_argument('--batch-size', type=int, default=finetune_hypers.batch_size)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    seed_everything(42, workers=True)

    # Prepare data module
    datamodule = FinetuneDataModule(
        train_pt=args.train_data,
        batch_size=args.batch_size,
        num_workers=finetune_hypers.num_workers,
    )

    modules_to_train = [args.module] if args.module else ['joints', 'poser']

    for module_name in modules_to_train:
        print()
        print('=' * 60)
        print(f'  Finetuning: {module_name}')
        print(f'  From:       {args.checkpoint_dir}/{module_name}')
        print(f'  Data:       {args.train_data}')
        print(f'  Epochs:     {args.epochs}  LR: {args.lr}  BS: {args.batch_size}')
        print('=' * 60)

        # Find best pretrained checkpoint
        src_dir = Path(args.checkpoint_dir) / module_name
        best_ckpt = get_best_checkpoint(src_dir)
        if best_ckpt is None:
            print(f'  !! No checkpoint found in {src_dir}, skipping.')
            continue
        ckpt_path = src_dir / best_ckpt
        print(f'  Loading pretrained: {ckpt_path}')

        # Load pretrained module
        module_cls = MODULES[module_name]
        model = module_cls.from_pretrained(str(ckpt_path))
        model.hypers = type('H', (), {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'num_epochs': args.epochs,
            'accelerator': 'gpu',
            'device': args.device,
            'num_workers': finetune_hypers.num_workers,
        })()

        # Override lr from hypers
        # The model's configure_optimizers reads self.hypers.lr
        save_dir = Path(args.output_dir) / module_name
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            monitor='validation_step_loss',
            save_top_k=3,
            mode='min',
            dirpath=save_dir,
            save_weights_only=True,
            filename='{epoch}-{validation_step_loss:.4f}',
        )

        trainer = L.Trainer(
            max_epochs=args.epochs,
            min_epochs=args.epochs,
            devices=[args.device],
            accelerator='gpu',
            callbacks=[checkpoint_cb],
            deterministic=True,
        )

        trainer.fit(model, datamodule=datamodule)
        torch.cuda.empty_cache()

        print(f'  ✓ Saved finetuned {module_name} checkpoints to {save_dir}')

    print('\nDone.')


if __name__ == '__main__':
    main()
