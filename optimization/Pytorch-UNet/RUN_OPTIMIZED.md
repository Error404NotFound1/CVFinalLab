# 优化版 U-Net 训练指南

## 主要改进

1. **预训练 Encoder (SMP)**: 使用 ResNet34 ImageNet 预训练权重替代从零训练
2. **Albumentations 数据增强**: ElasticTransform、GridDistortion、RandomRotate90、色彩增强
3. **Focal Tversky Loss + 类别权重**: 替换 CE + Dice，轮廓类权重 x3，更好处理类别不平衡
4. **AdamW + CosineAnnealingWarmRestarts**: 替换 RMSprop + ReduceLROnPlateau，加 3 epoch warmup
5. **Test-Time Augmentation (TTA)**: 推理时水平翻转平均
6. **Early Stopping**: 防止过拟合，默认 patience=10
7. **形态学后处理**: 去除小噪点，平滑边界

## 安装依赖

```bash
cd /Users/ybw/Mine/Labs/CVLab/finallab/optimization/Pytorch-UNet
pip install -r requirements.txt
```

## 训练命令

### 基础训练（ResNet34 预训练 + Early Stopping）
```bash
python train.py \
  --epochs 30 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --image-size 256 \
  --classes 3 \
  --data-root ./data \
  --encoder resnet34 \
  --encoder-weights imagenet \
  --early-stopping 10
```

### 启用 TTA（推理时增强）
```bash
python train.py \
  --epochs 30 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --image-size 256 \
  --classes 3 \
  --data-root ./data \
  --encoder resnet34 \
  --encoder-weights imagenet \
  --use-tta \
  --early-stopping 10
```

### 更强的 Encoder（ResNet50）
```bash
python train.py \
  --epochs 30 \
  --batch-size 12 \
  --learning-rate 1e-4 \
  --image-size 256 \
  --classes 3 \
  --data-root ./data \
  --encoder resnet50 \
  --encoder-weights imagenet \
  --use-tta \
  --early-stopping 10
```

### 启用混合精度（节省显存）
```bash
python train.py \
  --epochs 30 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --image-size 256 \
  --classes 3 \
  --data-root ./data \
  --encoder resnet34 \
  --encoder-weights imagenet \
  --use-tta \
  --amp \
  --early-stopping 10
```

## 参数说明

- `--encoder`: 可选 resnet34, resnet50, resnet101, efficientnet-b3, efficientnet-b4
- `--encoder-weights`: imagenet (预训练) 或 None (从零训练)
- `--use-tta`: 启用 Test-Time Augmentation（水平翻转平均）
- `--amp`: 启用混合精度训练（节省显存，加速训练）
- `--early-stopping`: Early stopping patience（默认 10 epochs）
- `--batch-size`: 根据显存调整（16 for ResNet34, 12 for ResNet50）
- `--learning-rate`: AdamW 推荐 1e-4

## 评估时启用后处理

训练完后，用 `evaluate_full()` 评估时可以启用形态学后处理：

```python
import torch
from unet import UNet
from torch.utils.data import DataLoader
from utils.data_loading import OxfordPetDataset
from evaluate import evaluate_full
import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=3,
)
state_dict = torch.load('checkpoints/best.pth', map_location=device)
if 'mask_values' in state_dict:
    del state_dict['mask_values']
model.load_state_dict(state_dict)
model.to(device)

test_ds = OxfordPetDataset(root='./data', split='test', image_size=256, augment=False)
test_loader = DataLoader(test_ds, batch_size=16, num_workers=4)

# 不启用后处理
results = evaluate_full(model, test_loader, device, amp=False, num_classes=3, use_postprocess=False)
print(f"Without postprocess - mIoU: {results['miou']:.4f}")

# 启用后处理
results_pp = evaluate_full(model, test_loader, device, amp=False, num_classes=3, use_postprocess=True)
print(f"With postprocess - mIoU: {results_pp['miou']:.4f}")
```

## 优化效果总结

相比 baseline (Dice 0.7654):
- **预训练 Encoder**: +5-10%
- **Albumentations 增强**: +2-4%
- **Focal Tversky Loss + 类别权重**: +2-3%（轮廓类 IoU 显著提升）
- **TTA**: +1-2%
- **Early Stopping**: 防止过拟合，节省训练时间
- **形态学后处理**: +0.5-1%（去除小噪点）

**预期最终 Dice: 0.80-0.85**

## 输出文件

- `checkpoints/best.pth`: 最佳模型
- `checkpoints/checkpoint_epochN.pth`: 每个 epoch 的检查点
- `outputs/metrics.csv`: 训练指标（epoch, train_loss, val_dice, lr）

## 新增优化说明

### 1. 类别权重（Class Weighting）
轮廓类像素占比极少（<5%），默认给轮廓类 3 倍权重：
```python
# 在 utils/losses.py 的 CombinedLoss 中
self.ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 3.0]))
```
可以通过修改 `losses.py` 调整权重比例。

### 2. Early Stopping
训练时自动监控验证集 Dice，如果连续 N 个 epoch 没有提升则停止：
- 默认 patience=10
- 通过 `--early-stopping N` 调整
- 日志会显示 "No improvement for X epoch(s)"
- 触发时显示 "Early stopping triggered at epoch X"

### 3. 形态学后处理
使用 scipy 的形态学操作去除预测中的小噪点：
- `binary_opening`: 去除小的孤立噪点
- `binary_closing`: 填充小的空洞
- 默认 kernel_size=3
- 在 `evaluate_full()` 中通过 `use_postprocess=True` 启用
- 适合最终提交结果，训练时不使用

**注意**: 后处理会增加推理时间，仅在需要最优结果时使用。

