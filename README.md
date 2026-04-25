# CVLab Final Project

计算机视觉大作业 - Oxford-IIIT Pet 语义分割

## 项目结构

```
finallab/
├── baseline/                    # 基线实现
│   └── Pytorch-UNet/           # 手写 U-Net，从零训练
│       ├── train.py            # 训练脚本
│       ├── evaluate.py         # 评估脚本
│       ├── predict.py          # 推理脚本
│       ├── utils/              # 工具函数
│       │   ├── data_loading.py # Oxford Pet 数据集加载
│       │   ├── dice_score.py   # Dice Loss
│       │   └── utils.py
│       ├── unet/               # U-Net 模型定义
│       ├── checkpoints/        # 模型权重（.gitignore）
│       ├── outputs/            # 训练日志（.gitignore）
│       ├── data/               # 数据集（.gitignore）
│       ├── requirements.txt
│       └── RUN_GUIDE.md        # 运行指南
│
├── optimization/               # 优化版实现
│   └── Pytorch-UNet/          # SMP + 预训练 + 多项优化
│       ├── train.py           # 优化版训练脚本
│       ├── evaluate.py        # 支持 TTA + 后处理
│       ├── utils/
│       │   ├── data_loading.py    # Albumentations 增强
│       │   ├── losses.py          # Focal Tversky Loss + 类别权重
│       │   └── postprocess.py     # 形态学后处理
│       ├── checkpoints/       # 模型权重（.gitignore）
│       ├── outputs/           # 训练日志（.gitignore）
│       ├── requirements.txt
│       └── RUN_OPTIMIZED.md   # 优化说明
│
├── optimization_plan.md       # 优化路线图
└── .gitignore                 # Git 忽略规则
```

## Baseline vs Optimization

| 项目 | Baseline | Optimization |
|------|----------|--------------|
| **模型** | 手写 U-Net，从零训练 | SMP U-Net + ResNet34 预训练 |
| **数据增强** | RandomHorizontalFlip | Albumentations 全套增强 |
| **损失函数** | CE + Dice | Focal Tversky + 类别权重 |
| **优化器** | RMSprop | AdamW |
| **学习率策略** | ReduceLROnPlateau | CosineAnnealingWarmRestarts + Warmup |
| **推理优化** | - | TTA + 形态学后处理 |
| **训练优化** | - | Early Stopping |
| **预期 Dice** | 0.55-0.70 | 0.80-0.85 |

## 快速开始

### Baseline

```bash
cd finallab/baseline/Pytorch-UNet
pip install torch torchvision -r requirements.txt

# 快速测试（2 epochs）
python train.py --epochs 2 --batch-size 4 --device cpu --image-size 128

# 正式训练（GPU）
python train.py --device cuda --amp
```

详见 [baseline/Pytorch-UNet/RUN_GUIDE.md](baseline/Pytorch-UNet/RUN_GUIDE.md)

### Optimization

```bash
cd finallab/optimization/Pytorch-UNet
pip install -r requirements.txt

# 完整优化训练
python train.py \
  --encoder resnet34 \
  --encoder-weights imagenet \
  --use-tta \
  --early-stopping 10 \
  --device cuda \
  --amp
```

详见 [optimization/Pytorch-UNet/RUN_OPTIMIZED.md](optimization/Pytorch-UNet/RUN_OPTIMIZED.md)

## 主要优化点

1. **预训练 Encoder**: ImageNet 预训练权重，迁移学习
2. **高级数据增强**: ElasticTransform, GridDistortion, 色彩增强
3. **损失函数优化**: Focal Tversky Loss + 轮廓类权重 x3
4. **学习率策略**: CosineAnnealing + 3 epoch warmup
5. **Test-Time Augmentation**: 推理时水平翻转平均
6. **Early Stopping**: 防止过拟合
7. **形态学后处理**: 去除小噪点，平滑边界

## 数据集

- **Oxford-IIIT Pet Dataset** (~800MB)
- 7,390 张宠物图片（37 个品种）
- 三分类语义分割：前景 / 背景 / 轮廓
- 自动下载到 `data/` 目录

## 环境要求

- Python >= 3.9
- PyTorch >= 2.1
- CUDA 12.x（可选，CPU 也能跑）
- 显存 >= 8GB（推荐）

## 输出文件

训练产出：
- `checkpoints/best.pth`: 最佳模型
- `checkpoints/checkpoint_epochN.pth`: 每个 epoch 检查点
- `outputs/metrics.csv`: 训练指标（loss, dice, lr）

## 报告内容

1. Baseline 实现与结果
2. 数据增强对比实验
3. 预训练 Encoder 消融实验
4. 损失函数优化效果
5. TTA + 后处理提升
6. 失败案例分析
7. 可视化对比（预测 mask、Grad-CAM）

## License

MIT
