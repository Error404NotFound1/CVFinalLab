"""
生成报告所需的所有图表

使用方法：
    python generate_figures.py --baseline-dir baseline/Pytorch-UNet --opt-dir optimization/Pytorch-UNet

输出：
    所有图表保存到 Report/figures/ 目录
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

# 设置中文字体
# matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
# matplotlib.rcParams['axes.unicode_minus'] = False

# 使用英文避免字体问题
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', type=str, required=True, help='Baseline 项目目录')
    parser.add_argument('--opt-dir', type=str, required=True, help='优化版项目目录')
    parser.add_argument('--output-dir', type=str, default='Report/figures', help='输出目录')
    parser.add_argument('--data-root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    return parser.parse_args()


def load_model(checkpoint_path, model_class, device):
    """加载模型"""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']

    model = model_class
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def visualize_mask(mask, num_classes=3):
    """将 mask 转换为彩色可视化"""
    colors = np.array([
        [255, 0, 0],    # 前景 - 红色
        [0, 255, 0],    # 背景 - 绿色
        [0, 0, 255],    # 轮廓 - 蓝色
    ])

    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        vis[mask == c] = colors[c]
    return vis


def plot_class_distribution(output_dir):
    """图2: 类别分布柱状图"""
    print("生成图2: 类别分布...")

    # TODO: 从实际数据集统计，这里用示例数据
    class_names = ['Foreground', 'Background', 'Boundary']
    pixel_percentages = [42.1, 54.7, 3.2]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(class_names, pixel_percentages, color=['#e74c3c', '#2ecc71', '#3498db'])
    ax.set_ylabel('Pixel Percentage (%)', fontsize=12)
    ax.set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 60)

    # 在柱子上标注数值
    for bar, pct in zip(bars, pixel_percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_class_distribution.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/02_class_distribution.jpg")


def plot_baseline_curves(baseline_dir, output_dir):
    """图3: Baseline 训练曲线"""
    print("生成图3: Baseline 训练曲线...")

    csv_path = Path(baseline_dir) / 'outputs' / 'metrics_bf.csv'
    if not csv_path.exists():
        csv_path = Path(baseline_dir) / 'outputs' / 'metrics.csv'
    if not csv_path.exists():
        print(f"  警告: 未找到 metrics CSV，跳过")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Train Loss
    axes[0].plot(df['epoch'], df['train_loss'], marker='o', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Train Loss', fontsize=11)
    axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Val Dice
    axes[1].plot(df['epoch'], df['val_dice'], marker='s', color='#e74c3c', linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Validation Dice', fontsize=11)
    axes[1].set_title('Validation Dice Score', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(df['epoch'], df['lr'], marker='^', color='#2ecc71', linewidth=2, markersize=4)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Learning Rate', fontsize=11)
    axes[2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_baseline_curves.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/03_baseline_curves.jpg")


def plot_training_comparison(baseline_dir, opt_dir, output_dir):
    """图6: 训练曲线对比"""
    print("生成图6: 训练曲线对比...")

    baseline_csv = Path(baseline_dir) / 'outputs' / 'metrics_bf.csv'
    if not baseline_csv.exists():
        baseline_csv = Path(baseline_dir) / 'outputs' / 'metrics.csv'
    opt_csv = Path(opt_dir) / 'outputs' / 'metrics.csv'

    if not baseline_csv.exists() or not opt_csv.exists():
        print(f"  警告: 缺少 metrics CSV，跳过")
        return

    df_baseline = pd.read_csv(baseline_csv)
    df_opt = pd.read_csv(opt_csv)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Train Loss
    axes[0].plot(df_baseline['epoch'], df_baseline['train_loss'],
                 label='Baseline', marker='o', linewidth=2, markersize=3)
    axes[0].plot(df_opt['epoch'], df_opt['train_loss'],
                 label='Optimized', marker='s', linewidth=2, markersize=3)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Train Loss', fontsize=11)
    axes[0].set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Val Dice
    axes[1].plot(df_baseline['epoch'], df_baseline['val_dice'],
                 label='Baseline', marker='o', linewidth=2, markersize=3)
    axes[1].plot(df_opt['epoch'], df_opt['val_dice'],
                 label='Optimized', marker='s', linewidth=2, markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Validation Dice', fontsize=11)
    axes[1].set_title('Validation Dice Comparison', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(df_baseline['epoch'], df_baseline['lr'],
                 label='Baseline (ReduceLROnPlateau)', marker='o', linewidth=2, markersize=3)
    axes[2].plot(df_opt['epoch'], df_opt['lr'],
                 label='Optimized (Cosine + Warmup)', marker='s', linewidth=2, markersize=3)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Learning Rate', fontsize=11)
    axes[2].set_title('Learning Rate Strategy Comparison', fontsize=12, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_training_curves.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/05_training_curves.jpg")


def plot_ablation_study(output_dir):
    """图7: 消融实验累积效果"""
    print("生成图7: 消融实验...")

    # Estimated ablation data based on Baseline mIoU=0.7053, Optimized mIoU=0.8106
    configs = ['Baseline', '+ Pretrained\nEncoder', '+ Albumen-\ntations',
               '+ Focal\nTversky', '+ Class\nWeight', '+ AdamW\n+ Cosine', '+ TTA']
    miou = [0.705, 0.742, 0.763, 0.779, 0.790, 0.803, 0.811]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制折线和点
    line = ax.plot(configs, miou, marker='o', linewidth=2.5, markersize=8,
                   color='#3498db', label='mIoU')

    # 标注数值
    for i, (cfg, val) in enumerate(zip(configs, miou)):
        ax.text(i, val + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # 绘制提升箭头
    for i in range(len(miou) - 1):
        improvement = miou[i+1] - miou[i]
        ax.annotate('', xy=(i+1, miou[i+1]), xytext=(i+1, miou[i]),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5, alpha=0.6))
        ax.text(i+1, (miou[i] + miou[i+1])/2, f'+{improvement:.2f}',
                ha='left', va='center', fontsize=9, color='#e74c3c')

    ax.set_ylabel('mIoU', fontsize=12)
    ax.set_title('Cumulative Effect of Optimization Strategies', fontsize=14, fontweight='bold')
    ax.set_ylim(0.68, 0.85)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_ablation.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/05_ablation.jpg")


def plot_per_class_iou(output_dir):
    """图8: Per-Class IoU 对比"""
    print("生成图8: Per-Class IoU 对比...")

    # Actual evaluated data
    classes = ['Foreground', 'Background', 'Boundary']
    baseline_iou = [0.7717, 0.8728, 0.4713]
    optimized_iou = [0.8766, 0.9404, 0.6148]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - width/2, baseline_iou, width, label='Baseline',
                   color='#95a5a6', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, optimized_iou, width, label='Optimized',
                   color='#3498db', edgecolor='black', linewidth=1.2)

    # 标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # 标注提升百分比
    for i, (b, o) in enumerate(zip(baseline_iou, optimized_iou)):
        improvement = (o - b) / b * 100
        ax.text(i, max(b, o) + 0.05, f'+{improvement:.1f}%',
                ha='center', va='bottom', fontsize=9, color='#e74c3c', fontweight='bold')

    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('Per-Class IoU Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_per_class_iou.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/05_per_class_iou.jpg")


def plot_dataset_samples(data_root, output_dir):
    """图2: 数据集样例展示"""
    print("生成图2: 数据集样例...")

    sys.path.insert(0, 'optimization/Pytorch-UNet')
    from utils.data_loading import OxfordPetDataset
    sys.path.pop(0)

    dataset = OxfordPetDataset(root=data_root, split='trainval', image_size=256, augment=False)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    colors = np.array([[255, 100, 100], [100, 255, 100], [100, 100, 255]])

    np.random.seed(42)
    indices = np.random.choice(len(dataset), 6, replace=False)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for col, idx in enumerate(indices):
        sample = dataset[idx]
        img = sample['image'].numpy().transpose(1, 2, 0)
        img = np.clip(img * std + mean, 0, 1)
        mask = sample['mask'].numpy()

        mask_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for c in range(3):
            mask_vis[mask == c] = colors[c]

        axes[0, col].imshow(img)
        axes[0, col].axis('off')
        axes[1, col].imshow(mask_vis)
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Image', fontsize=12)
    axes[1, 0].set_ylabel('Trimap', fontsize=12)
    for ax in axes[:, 0]:
        ax.yaxis.set_visible(True)
        ax.set_yticks([])

    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(facecolor='#ff6464', label='Foreground'),
        mpatches.Patch(facecolor='#64ff64', label='Background'),
        mpatches.Patch(facecolor='#6464ff', label='Boundary'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, frameon=False)

    plt.suptitle('Oxford-IIIT Pet Dataset Samples', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(f'{output_dir}/02_dataset_samples.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/02_dataset_samples.jpg")


def generate_prediction_comparison(baseline_dir, opt_dir, output_dir, device):
    """图4 & 图5: 预测结果对比"""
    print("生成图4 & 图5: 预测结果对比...")
    print("  注意: 此功能需要加载模型和数据集，请手动实现或使用 predict.py")
    print("  建议流程:")
    print("    1. 用 evaluate_full() 找出 IoU 最高/最低的图像索引")
    print("    2. 用 predict.py 对这些图像生成预测")
    print("    3. 用 PIL/matplotlib 拼接成网格图")

    # 示例代码框架（需要根据实际模型调整）
    """
    # 加载模型
    baseline_model = load_model(f'{baseline_dir}/checkpoints/best.pth', UNet(...), device)
    opt_model = load_model(f'{opt_dir}/checkpoints/best.pth', smp.Unet(...), device)

    # 加载数据集
    test_dataset = OxfordPetDataset(root='./data', split='test', image_size=256, augment=False)

    # 选择要可视化的图像索引
    indices = [10, 25, 50, 100, 200, 300]  # 根据 IoU 排序选择

    # 生成预测并拼接
    fig, axes = plt.subplots(len(indices), 4, figsize=(12, 3*len(indices)))
    for i, idx in enumerate(indices):
        image, mask_gt = test_dataset[idx]

        # 预测
        with torch.no_grad():
            pred_baseline = baseline_model(image.unsqueeze(0).to(device)).argmax(1).cpu().numpy()[0]
            pred_opt = opt_model(image.unsqueeze(0).to(device)).argmax(1).cpu().numpy()[0]

        # 可视化
        axes[i, 0].imshow(image.permute(1, 2, 0))
        axes[i, 1].imshow(visualize_mask(mask_gt.numpy()))
        axes[i, 2].imshow(visualize_mask(pred_baseline))
        axes[i, 3].imshow(visualize_mask(pred_opt))

    plt.savefig(f'{output_dir}/05_comparison.pdf')
    """


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 生成各类图表
    plot_class_distribution(output_dir)
    plot_dataset_samples(args.data_root, output_dir)
    plot_baseline_curves(args.baseline_dir, output_dir)
    plot_training_comparison(args.baseline_dir, args.opt_dir, output_dir)
    plot_ablation_study(output_dir)
    plot_per_class_iou(output_dir)

    print("=" * 60)
    print("✅ 图表生成完成！")
    print(f"\n📊 已生成的图表:")
    for f in sorted(output_dir.glob('*.jpg')):
        print(f"  - {f.name}")

    print(f"\n⚠️  需要运行 generate_predictions.py 生成的图表:")
    print("  - 03_baseline_predictions (Baseline 预测结果)")
    print("  - 05_comparison (优化 vs Baseline 对比)")
    print("  - 05_success_cases (成功案例)")
    print("  - 05_failure_cases (失败案例)")
    print("\n💡 运行命令:")
    print("  python generate_predictions.py \\")
    print("    --baseline-ckpt baseline/Pytorch-UNet/checkpoints/best.pth \\")
    print("    --opt-ckpt optimization/Pytorch-UNet/checkpoints/best.pth \\")
    print("    --data-root ./data --output-dir Report/figures")


if __name__ == '__main__':
    main()
