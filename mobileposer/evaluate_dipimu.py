"""
Evaluate MobilePoser on DIP-IMU / IMUPoser test sets using pre-computed
acc/ori from .pt files.

Usage examples:
  # Evaluate on DIP-IMU test set
  python -m mobileposer.evaluate_dipimu \
      --test-data data/processed_datasets/eval/dip_test.pt \
      --model checkpoints/finetuned_dip/combined_model.pth \
      --combo global

  # Evaluate on IMUPoser test set
  python -m mobileposer.evaluate_dipimu \
      --test-data data/processed_datasets/eval/imuposer_test.pt \
      --model checkpoints/finetuned_imuposer/combined_model.pth

  # Evaluate with module-level checkpoints (auto-combines)
  python -m mobileposer.evaluate_dipimu \
      --test-data data/processed_datasets/eval/dip_test.pt \
      --checkpoint-dir checkpoints/finetuned_dip \
      --base-checkpoint-dir checkpoints/all_MotionGV_no_noise

  # Save per-sequence predictions
  python -m mobileposer.evaluate_dipimu \
      --test-data data/processed_datasets/eval/dip_test.pt \
      --model checkpoints/finetuned_dip/combined_model.pth \
      --save-predictions predictions/dip_finetuned
"""

import os
import pickle
import torch
import tqdm
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

import mobileposer.articulate as art
from mobileposer.config import paths, amass, datasets, model_config, joint_set
from mobileposer.constants import MODULES
from mobileposer.models import MobilePoserNet
from mobileposer.utils.file_utils import get_best_checkpoint
from mobileposer.finetune_dipimu import PrecomputedIMUDataset


class PoseEvaluator:
    """Compute standard pose-estimation metrics (same as evaluate.py)."""

    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(
            paths.smpl_file, joint_mask=torch.tensor([2, 5, 16, 20]), fps=datasets.fps
        )

    def eval(self, pose_p, pose_t, tran_p=None, tran_t=None):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        tran_p = tran_p.clone().view(-1, 3)
        tran_t = tran_t.clone().view(-1, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)

        errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
        return torch.stack([
            errs[9], errs[3], errs[9], errs[0] * 100,
            errs[7] * 100, errs[1] * 100, errs[4] / 100, errs[6]
        ])

    @staticmethod
    def print(errors):
        names = [
            'SIP Error (deg)', 'Angular Error (deg)', 'Masked Angular Error (deg)',
            'Positional Error (cm)', 'Masked Positional Error (cm)', 'Mesh Error (cm)',
            'Jitter Error (100m/s^3)', 'Distance Error (cm)',
        ]
        for i, name in enumerate(names):
            print('  %s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))


def load_model_from_modules(ft_dir: str, base_dir: str):
    """
    Build a MobilePoserNet by loading finetuned joints/poser from `ft_dir`
    and velocity/foot_contact from `base_dir`.

    Falls back to base_dir when a module directory in ft_dir is missing or
    empty (no checkpoint files).
    """
    modules = {}
    for module_name in ['joints', 'poser', 'velocity', 'foot_contact']:
        # Try finetuned dir first, then fall back to base dir
        src = Path(ft_dir) / module_name
        best = get_best_checkpoint(src) if src.exists() else None

        if best is None:
            # Finetuned dir missing or empty → fall back to base
            src = Path(base_dir) / module_name
            best = get_best_checkpoint(src) if src.exists() else None
            if best is None:
                print(f'  Warning: no checkpoint for {module_name} in either '
                      f'{ft_dir} or {base_dir}')
                continue
            print(f'  Loading {module_name} (base): {src / best}')
        else:
            print(f'  Loading {module_name} (finetuned): {src / best}')

        ckpt = src / best
        modules[module_name] = MODULES[module_name].load_from_checkpoint(str(ckpt))

    # Build combined net
    model = MobilePoserNet(
        poser=modules.get('poser'),
        joints=modules.get('joints'),
        foot_contact=modules.get('foot_contact'),
        velocity=modules.get('velocity'),
    )
    return model.to(model_config.device)


def load_model_from_weights(model_path: str):
    """Load a combined MobilePoserNet from a single .pth state dict."""
    device = model_config.device
    model = MobilePoserNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception:
        model = MobilePoserNet.load_from_checkpoint(model_path)
    return model.to(device)


@torch.no_grad()
def evaluate(model, dataset, combo: str = 'global', save_dir: str = None):
    device = model_config.device

    # Load all sequences
    xs, ys, fnames = zip(*[
        (imu.to(device), (pose.to(device), tran), fname)
        for imu, pose, joint, tran, fname in dataset
    ])

    evaluator = PoseEvaluator()
    offline_errs = []

    model.eval()
    for idx, (x, y, fname) in enumerate(tqdm.tqdm(list(zip(xs, ys, fnames)))):
        model.reset()
        pose_p, joint_p, tran_p, _ = model.forward_offline(x.unsqueeze(0), [x.shape[0]])

        pose_t, tran_t = y
        pose_t_rotmat = art.math.r6d_to_rotation_matrix(pose_t).view(-1, 24, 3, 3)

        # Optionally save predictions
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            pose_p_aa = art.math.rotation_matrix_to_axis_angle(pose_p).reshape(-1, 24, 3)
            pose_t_aa = art.math.rotation_matrix_to_axis_angle(pose_t_rotmat).reshape(-1, 24, 3)
            data_dict = {
                'gt': {
                    'global_orient': pose_t_aa[:, 0].cpu().numpy(),
                    'body_pose': pose_t_aa[:, 1:24].reshape(-1, 69).cpu().numpy(),
                    'transl': tran_t.cpu().numpy() if isinstance(tran_t, torch.Tensor) else tran_t,
                },
                'recon': {
                    'global_orient': pose_p_aa[:, 0].cpu().numpy(),
                    'body_pose': pose_p_aa[:, 1:24].reshape(-1, 69).cpu().numpy(),
                    'transl': tran_p.cpu().numpy(),
                },
            }
            safe_fname = fname.replace('/', '_')
            with open(os.path.join(save_dir, f'{safe_fname}.pkl'), 'wb') as f:
                pickle.dump(data_dict, f)

        pose_t_flat = pose_t_rotmat.view(-1, 24 * 3 * 3)
        offline_errs.append(evaluator.eval(pose_p, pose_t_flat, tran_p=tran_p, tran_t=tran_t))

    print('\n============ Evaluation Results ============')
    evaluator.print(torch.stack(offline_errs).mean(dim=0))
    print('============================================\n')


def main():
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, default=None,
                       help='Path to combined model weights (.pth)')
    group.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory with finetuned module checkpoints')
    parser.add_argument('--base-checkpoint-dir', type=str, default=None,
                        help='Base checkpoint dir for velocity/foot_contact '
                             '(required when using --checkpoint-dir)')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test .pt file')
    parser.add_argument('--combo', type=str, default='global',
                        choices=list(amass.combos.keys()),
                        help='IMU combo configuration')
    parser.add_argument('--save-predictions', type=str, default=None,
                        help='Directory to save per-sequence predictions')
    args = parser.parse_args()

    # Load model
    if args.model:
        print(f'Loading model from: {args.model}')
        model = load_model_from_weights(args.model)
    else:
        if args.base_checkpoint_dir is None:
            parser.error('--base-checkpoint-dir is required when using --checkpoint-dir')
        print(f'Building model from module checkpoints:')
        print(f'  Finetuned: {args.checkpoint_dir}')
        print(f'  Base:      {args.base_checkpoint_dir}')
        model = load_model_from_modules(args.checkpoint_dir, args.base_checkpoint_dir)

    # Load dataset
    print(f'Loading test data: {args.test_data}')
    dataset = PrecomputedIMUDataset(args.test_data, fold='test', combo=args.combo)
    print(f'Test sequences: {len(dataset)}')

    # Evaluate
    print(f'Evaluating with combo: {args.combo}')
    evaluate(model, dataset, combo=args.combo, save_dir=args.save_predictions)


if __name__ == '__main__':
    main()
