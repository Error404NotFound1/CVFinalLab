# Baseline 运行指导

## 环境要求

- Python >= 3.9
- PyTorch >= 2.1（自带 torchvision）
- 显卡（可选）：NVIDIA GPU + CUDA，或 Apple Silicon MPS，或纯 CPU

## 安装依赖

```bash
cd finallab/baseline/Pytorch-UNet

# 先安装 PyTorch（根据你的硬件选一个）：

# --- GPU (Linux, CUDA 12.x) ---
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# --- CPU only (macOS / 无卡 Linux) ---
pip install torch torchvision

# --- Apple Silicon (macOS, 自动启用 MPS) ---
pip install torch torchvision

# 再装其他依赖
pip install -r requirements.txt
```

## Docker 运行脚本

仓库里新增了 `scripts/run_docker.sh`，它会自动完成两件事：先构建镜像，再在容器里启动训练。

### 最简用法

```bash
bash scripts/run_docker.sh
```

这条命令等价于下面两步：

1. `docker build -t pytorch-unet .`：根据当前目录下的 `Dockerfile` 构建镜像，镜像名叫 `pytorch-unet`。
2. `docker run ... pytorch-unet bash -lc "python train.py --device auto --amp"`：启动容器并执行训练命令。

### 命令含义

脚本默认会把整个项目目录挂载到容器里的 `/workspace/unet`，所以容器内训练产生的文件会直接写回宿主机当前仓库。

脚本默认执行的训练命令是：

```bash
python train.py --device auto --amp
```

含义如下：

`python train.py`：在容器中启动训练入口。

`--device auto`：自动选择可用设备，优先用 CUDA，其次 MPS，最后 CPU。

`--amp`：启用混合精度训练。这个参数只在 CUDA 上真正生效。

### 改成你自己的训练命令

如果你想改参数，直接编辑 `scripts/run_docker.sh` 里的这一行：

```bash
python train.py --device auto --amp
```

例如可以改成：

```bash
python train.py --device cuda --epochs 2 --batch-size 4 --image-size 128
```

这里的含义是：

`--device cuda`：强制使用 GPU。

`--epochs 2`：只跑 2 个 epoch，适合快速检查。

`--batch-size 4`：把 batch size 调小，减少显存压力。

`--image-size 128`：把输入尺寸缩小，加快测试速度。

### 这个脚本帮你做了什么

`docker build`：构建运行环境。

`-v "$PWD:/workspace/unet"`：把当前项目目录挂载进容器，这样代码、数据、`checkpoints/`、`outputs/` 都会保留在宿主机。

`--shm-size=8g`：给容器更大的共享内存，避免 DataLoader 在多进程读取数据时出问题。

`--ulimit memlock=-1`：放宽锁页内存限制，和 PyTorch / CUDA 的容器运行习惯一致。

`python train.py --device auto --amp`：容器启动后直接执行训练命令。

### 代理配置

当前 Dockerfile 已经把代理写死为 `http://172.19.135.130:5000`，所以你重新构建镜像后，镜像内部的 `pip`、`apt` 和容器运行时默认都会带这个代理。

同时，训练进程会用宿主机当前用户的 UID/GID 启动，这样挂载进来的项目目录、`data/`、`outputs/` 和 `checkpoints/` 都能正常写入。

如果后面代理地址变了，只需要改 [Dockerfile](Dockerfile) 里这两行：

```dockerfile
ARG HTTP_PROXY=http://172.19.135.130:5000
ARG HTTPS_PROXY=http://172.19.135.130:5000
```

然后重新执行：

```bash
bash scripts/run_docker.sh
```

另外，`requirements.txt` 里的 `matplotlib` 已经调整为 `>=3.7.5`，这是为了兼容当前基础镜像里的 Python 3.8；否则会出现“找不到满足条件的版本”的报错。

## 运行训练

### 快速测试（确认 pipeline 跑通，2-3 分钟）

```bash
# CPU 模式，2 个 epoch，小 batch
python train.py --epochs 2 --batch-size 4 --device cpu --image-size 128
```

首次运行会自动下载 Oxford-IIIT Pet Dataset（~800MB）到 `./data/` 目录。

### 正式训练（有 GPU）

```bash
# 默认配置：30 epochs, batch=16, lr=1e-4, 256x256, 3 类
python train.py --device cuda --amp
```

### 正式训练（无 GPU，纯 CPU）

```bash
# 缩小图片 + 减少 batch 以加速
python train.py --device cpu --epochs 20 --batch-size 8 --image-size 128
```

### macOS Apple Silicon

```bash
python train.py --device mps --batch-size 16
```

## 全部参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs, -e` | 30 | 训练轮数 |
| `--batch-size, -b` | 16 | 批大小 |
| `--learning-rate, -l` | 1e-4 | 学习率 |
| `--image-size` | 256 | 输入图片尺寸 |
| `--classes, -c` | 3 | 分类数（3=前景/背景/轮廓） |
| `--device` | auto | 设备选择：auto/cpu/cuda/mps |
| `--amp` | False | 启用混合精度（仅 CUDA 有效） |
| `--bilinear` | False | 用双线性上采样替代转置卷积 |
| `--validation, -v` | 10.0 | 验证集比例（百分比） |
| `--data-root` | ./data | 数据集存放目录 |
| `--load, -f` | - | 加载已有 checkpoint 继续训练 |
| `--wandb` | False | 启用 wandb 在线日志 |

## 训练产出

```
Pytorch-UNet/
├── checkpoints/
│   ├── checkpoint_epoch1.pth
│   ├── checkpoint_epoch2.pth
│   ├── ...
│   └── best.pth              ← 验证集 Dice 最高的模型
├── outputs/
│   └── metrics.csv            ← 每 epoch 的 train_loss, val_dice, lr
└── data/
    └── oxford-iiit-pet/       ← 自动下载的数据集
```

## 评估最佳模型

训练完后，可以用 `evaluate.py` 中的 `evaluate_full()` 做完整评估（mIoU + per-class IoU + pixel accuracy）。

用法示例（在 Python 中）：

```python
import torch
from unet import UNet
from torch.utils.data import DataLoader
from utils.data_loading import OxfordPetDataset
from evaluate import evaluate_full

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(n_channels=3, n_classes=3, bilinear=False)
state_dict = torch.load('checkpoints/best.pth', map_location=device)
del state_dict['mask_values']
model.load_state_dict(state_dict)
model.to(device)

test_ds = OxfordPetDataset(root='./data', split='test', image_size=256, augment=False)
test_loader = DataLoader(test_ds, batch_size=16, num_workers=4)

results = evaluate_full(model, test_loader, device, amp=False, num_classes=3)
print(f"mIoU: {results['miou']:.4f}")
print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
for cls, iou in results['per_class_iou'].items():
    print(f"  {cls}: {iou:.4f}")
```

## 预期性能

| 指标 | 预期范围 |
|------|---------|
| Val Dice | 0.55 - 0.70 |
| mIoU | 0.45 - 0.60 |
| Pixel Accuracy | 0.80 - 0.88 |

轮廓类（boundary）的 IoU 会明显低于前景和背景类，这是正常的——边缘像素极少，是后续优化的重点。

## 常见问题

**Q: 下载数据集很慢/超时？**
数据集来自 Oxford 官网。如果网络不好，可以手动下载后放到 `./data/oxford-iiit-pet/` 目录。

**Q: CPU 训练太慢？**
用 `--image-size 128 --batch-size 8 --epochs 10` 快速跑通看效果，正式训练建议上 GPU。

**Q: MPS 报错？**
某些 PyTorch 版本的 MPS 不支持 `channels_last` 格式。如果报错，用 `--device cpu` 回退。

**Q: wandb 需要登录？**
默认不启用 wandb。加 `--wandb` 才会用，且使用匿名模式不需要账号。
