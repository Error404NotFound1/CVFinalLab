"""
生成预测结果对比图

使用方法：
    python generate_predictions.py --baseline-ckpt baseline/Pytorch-UNet/checkpoints/best.pth \
                                    --opt-ckpt optimization/Pytorch-UNet/checkpoints/best.pth \
                                    --data-root ./data \
                                    --output-dir Report/figures

功能：
    1. 自动找出 IoU 最高/最低的图像
    2. 生成 Baseline vs 优化模型的预测对比
    3. 输出网格拼接图
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ckpt', type=str, required=True, help='Baseline checkpoint')
    parser.add_argument('--opt-ckpt', type=str, required=True, help='优化模型 checkpoint')
    parser.add_argument('--data-root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--output-dir', type=str, default='Report/figures', help='输出目录')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    parser.add_argument('--num-samples', type=int, default=6, help='每类图片数量')
    parser.add_argument('--skip-eval', action='store_true', help='跳过评估，使用预设索引')
    return parser.parse_args()


def get_device(device_str):
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_str)


def load_models(baseline_ckpt, opt_ckpt, device):
    """加载两个模型"""
    print("加载模型...")

    # 导入模型类
    sys.path.insert(0, 'baseline/Pytorch-UNet')
    from unet import UNet as BaselineUNet
    sys.path.pop(0)

    sys.path.insert(0, 'optimization/Pytorch-UNet')
    import segmentation_models_pytorch as smp
    sys.path.pop(0)

    # 加载 Baseline
    baseline_model = BaselineUNet(n_channels=3, n_classes=3, bilinear=False)
    state_dict = torch.load(baseline_ckpt, map_location=device, weights_only=False)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    baseline_model.load_state_dict(state_dict)
    baseline_model.to(device)
    baseline_model.eval()
    print(f"  ✓ Baseline 模型加载完成")

    # 加载优化模型
    opt_model = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                         in_channels=3, classes=3)
    state_dict = torch.load(opt_ckpt, map_location=device, weights_only=False)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    opt_model.load_state_dict(state_dict)
    opt_model.to(device)
    opt_model.eval()
    print(f"  ✓ 优化模型加载完成")

    return baseline_model, opt_model


def load_dataset(data_root):
    """加载测试集"""
    print("加载数据集...")
    sys.path.insert(0, 'optimization/Pytorch-UNet')
    from utils.data_loading import OxfordPetDataset
    sys.path.pop(0)

    test_dataset = OxfordPetDataset(root=data_root, split='test',
                                    image_size=256, augment=False)
    print(f"  ✓ 测试集大小: {len(test_dataset)}")
    return test_dataset


def compute_iou_per_image(baseline_model, opt_model, dataset, device):
    """计算每张图的 IoU"""
    print("计算每张图的 IoU...")

    baseline_ious = []
    opt_ious = []
    improvements = []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='评估'):
            image, mask_true = batch['image'].to(device), batch['mask'].to(device)

            # Baseline 预测
            pred_baseline = baseline_model(image).argmax(1)
            iou_baseline = compute_iou(pred_baseline, mask_true, num_classes=3)

            # 优化模型预测
            pred_opt = opt_model(image).argmax(1)
            iou_opt = compute_iou(pred_opt, mask_true, num_classes=3)

            baseline_ious.append(iou_baseline)
            opt_ious.append(iou_opt)
            improvements.append(iou_opt - iou_baseline)

    return np.array(baseline_ious), np.array(opt_ious), np.array(improvements)


def compute_iou(pred, target, num_classes=3):
    """计算 mIoU"""
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)


def visualize_mask(mask, num_classes=3):
    """将 mask 转换为彩色可视化"""
    colors = np.array([
        [255, 100, 100],  # 前景 - 浅红
        [100, 255, 100],  # 背景 - 浅绿
        [100, 100, 255],  # 轮廓 - 浅蓝
    ])

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        vis[mask == c] = colors[c]
    return vis


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反归一化"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def generate_comparison_grid(baseline_model, opt_model, dataset, indices,
                              device, output_path, title):
    """生成对比网格图"""
    num_samples = len(indices)
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image, mask_gt = sample['image'], sample['mask']
        image_tensor = image.unsqueeze(0).to(device)

        # 预测
        with torch.no_grad():
            pred_baseline = baseline_model(image_tensor).argmax(1)[0].cpu().numpy()
            pred_opt = opt_model(image_tensor).argmax(1)[0].cpu().numpy()

        # 反归一化图像
        image_vis = denormalize(image).permute(1, 2, 0).numpy()
        mask_gt_np = mask_gt.numpy()

        # 计算 IoU
        iou_baseline = compute_iou(torch.from_numpy(pred_baseline).unsqueeze(0),
                                   torch.from_numpy(mask_gt_np).unsqueeze(0))
        iou_opt = compute_iou(torch.from_numpy(pred_opt).unsqueeze(0),
                             torch.from_numpy(mask_gt_np).unsqueeze(0))

        # 绘制
        axes[i, 0].imshow(image_vis)
        axes[i, 0].set_title('Input', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(visualize_mask(mask_gt_np))
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(visualize_mask(pred_baseline))
        axes[i, 2].set_title(f'Baseline\nmIoU={iou_baseline:.3f}', fontsize=10)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(visualize_mask(pred_opt))
        axes[i, 3].set_title(f'Optimized\nmIoU={iou_opt:.3f}', fontsize=10)
        axes[i, 3].axis('off')

    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='#ff6464', label='Foreground'),
        mpatches.Patch(facecolor='#64ff64', label='Background'),
        mpatches.Patch(facecolor='#6464ff', label='Boundary')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.98), fontsize=11, frameon=False)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存到: {output_path}")


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型和数据
    baseline_model, opt_model = load_models(args.baseline_ckpt, args.opt_ckpt, device)
    dataset = load_dataset(args.data_root)

    if args.skip_eval:
        success_indices = np.array([3414, 906, 521])
        failure_indices = np.array([505, 644, 1093])
        baseline_problem_indices = np.array([1858, 2005, 2292, 2302, 1690, 1093])
    else:
        # 计算每张图的 IoU
        baseline_ious, opt_ious, improvements = compute_iou_per_image(
            baseline_model, opt_model, dataset, device
        )

        # 选择图像索引
        # 1. 成功案例：提升最大的
        success_indices = np.argsort(improvements)[-args.num_samples//2:][::-1]

        # 2. 失败案例：两个模型都很低的
        both_low = (baseline_ious < 0.4) & (opt_ious < 0.4)
        failure_indices = np.where(both_low)[0][:args.num_samples//2]

        # 3. Baseline 问题案例：Baseline 低但优化模型有改善的
        baseline_problem_indices = np.argsort(baseline_ious)[:args.num_samples]

    print(f"\n选择的图像索引:")
    print(f"  成功案例: {success_indices.tolist()}")
    print(f"  失败案例: {failure_indices.tolist()}")
    print(f"  Baseline 问题: {baseline_problem_indices.tolist()}")

    # 生成图表
    print("\n生成对比图...")

    # 图4: Baseline 预测结果（展示问题）
    generate_comparison_grid(
        baseline_model, opt_model, dataset, baseline_problem_indices,
        device, output_dir / '03_baseline_predictions.jpg',
        'Baseline Model Prediction Analysis'
    )

    # 图5: 成功案例
    generate_comparison_grid(
        baseline_model, opt_model, dataset, success_indices,
        device, output_dir / '05_success_cases.jpg',
        'Optimized Model Success Cases'
    )

    # 图5: 失败案例
    if len(failure_indices) > 0:
        generate_comparison_grid(
            baseline_model, opt_model, dataset, failure_indices,
            device, output_dir / '05_failure_cases.jpg',
            'Failure Case Analysis'
        )

    # 图5: 综合对比（混合成功和失败）
    mixed_indices = np.concatenate([success_indices[:3], failure_indices[:3]])
    generate_comparison_grid(
        baseline_model, opt_model, dataset, mixed_indices,
        device, output_dir / '05_comparison.jpg',
        'Baseline vs Optimized Model Comparison'
    )

    print("\n✅ 预测对比图生成完成！")


if __name__ == '__main__':
    main()
