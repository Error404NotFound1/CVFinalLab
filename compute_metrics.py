"""
计算 Baseline 和优化模型的 per-class IoU，输出到 JSON 供 generate_figures.py 使用
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ckpt', type=str, required=True)
    parser.add_argument('--opt-ckpt', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output', type=str, default='Report/figures/metrics.json')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline eval, use cached result')
    return parser.parse_args()

def get_device(device_str):
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_str)

def load_model(ckpt_path, device, model_type='baseline'):
    if model_type == 'baseline':
        sys.path.insert(0, 'baseline/Pytorch-UNet')
        from unet import UNet
        sys.path.pop(0)
        model = UNet(n_channels=3, n_classes=3, bilinear=False)
    else:
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=3,
            classes=3,
        )

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state:
        sd = state['model_state_dict']
    elif 'state_dict' in state:
        sd = state['state_dict']
    else:
        sd = state
    # Filter out non-model keys
    sd = {k: v for k, v in sd.items() if k != 'mask_values'}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

def compute_per_class_iou(model, dataloader, device, num_classes=3):
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            masks = batch['mask'].numpy()
            preds = model(images).argmax(1).cpu().numpy()

            for c in range(num_classes):
                pred_c = (preds == c)
                mask_c = (masks == c)
                intersection[c] += (pred_c & mask_c).sum()
                union[c] += (pred_c | mask_c).sum()

    iou = intersection / (union + 1e-8)
    return iou

def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Load dataset
    sys.path.insert(0, 'optimization/Pytorch-UNet')
    from utils.data_loading import OxfordPetDataset
    sys.path.pop(0)

    dataset = OxfordPetDataset(root=args.data_root, split='test', image_size=256, augment=False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"Test set size: {len(dataset)}")

    # Baseline
    if args.skip_baseline:
        baseline_iou = np.array([0.77173364, 0.87276553, 0.47134519])
        baseline_miou = baseline_iou.mean()
        print(f"\nUsing cached Baseline per-class IoU: {baseline_iou}")
        print(f"  Baseline mIoU: {baseline_miou:.4f}")
    else:
        print("\nEvaluating Baseline...")
        baseline_model = load_model(args.baseline_ckpt, device, 'baseline')
        baseline_iou = compute_per_class_iou(baseline_model, dataloader, device)
        baseline_miou = baseline_iou.mean()
        print(f"  Baseline per-class IoU: {baseline_iou}")
        print(f"  Baseline mIoU: {baseline_miou:.4f}")

    # Optimized
    print("\nEvaluating Optimized...")
    opt_model = load_model(args.opt_ckpt, device, 'optimized')
    opt_iou = compute_per_class_iou(opt_model, dataloader, device)
    opt_miou = opt_iou.mean()
    print(f"  Optimized per-class IoU: {opt_iou}")
    print(f"  Optimized mIoU: {opt_miou:.4f}")

    # Save
    results = {
        'classes': ['Foreground', 'Background', 'Boundary'],
        'baseline_iou': baseline_iou.tolist(),
        'optimized_iou': opt_iou.tolist(),
        'baseline_miou': float(baseline_miou),
        'optimized_miou': float(opt_miou),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()
