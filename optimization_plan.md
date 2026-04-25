# Baseline 改造 + Optimization 全流程规划

## 第 0 步：让 baseline 跑通（必须）

改 `utils/data_loading.py`，新增 `OxfordPetDataset` 类：
- 内部用 `torchvision.datasets.OxfordIIITPet(download=True)` 自动下载数据
- trimap 1/2/3 → 0/1/2 三分类映射
- `torchvision.transforms` 做 Resize + Normalize
- 训练集加 `RandomHorizontalFlip`

改 [train.py](finallab/baseline/Pytorch-UNet/train.py) 默认参数：
- batch_size 从 1 → 16（GPU 显存够用）
- lr 从 1e-5 → 1e-4（RMSprop 的合理起点）
- scale 从 0.5 → 1.0（原图尺寸）
- epochs 从 5 → 30
- `device` 检测加 `'cuda' if torch.cuda.is_available() else 'cpu'`

**跑通后产出：一个能从 0 开始训练、在 test 集上评估的完整 pipeline。**

---

## 第 1 层：数据侧优化（体现数据处理能力）

### 1.1 高级数据增强（Albumentations）
- 替换 torchvision.transforms → Albumentations
- 加 `ElasticTransform`、`GridDistortion`、`RandomRotate90`、`RandomScale`
- 加色彩增强：`HueSaturationValue`、`RandomBrightnessContrast`
- **报告写法**：对比 baseline 增强 vs 高级增强的 train/val loss 曲线和 mIoU 提升，解释为什么几何增强对非刚性物体（宠物）特别有效

### 1.2 数据质量分析
- 统计每个类别的像素占比分布 → 做柱状图
- 发现轮廓类（trimap=2）像素占比极低 → 自然引出类别不平衡问题 → 引出后续 loss 优化
- **报告写法**：这是论文级的数据分析，不是简单跑代码就完事

---

## 第 2 层：模型侧优化（体现模型理解深度）

### 2.1 迁移学习：预训练 Encoder（最大加分项）
- 当前 U-Net 从 0 训练（所有权重随机初始化）
- 改 encoder 为预训练 backbone：ResNet34 → ResNet50 → EfficientNet-B3/B4
- **实现方式**：
  - 方案 A：用 `timm` 库替换手写 U-Net 的 encoder 部分
  - 方案 B：直接引入 `segmentation-models-pytorch (SMP)`，用 `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet")`
  - 方案 B 更优雅，代码量更小，且正好可以对比"手写 U-Net vs SMP 预训练 U-Net"的差距
- **报告写法**：做 encoder 消融实验，画参数-精度权衡图

### 2.2 多解码器架构对比
- baseline：标准 U-Net decoder
- 实验：FPN（Feature Pyramid Network）、DeepLabV3+（空洞卷积）
- 用 SMP 只需改 `arch` 参数：`smp.FPN()`, `smp.DeepLabV3Plus()`
- **报告写法**：同一 encoder 下不同解码器的对比，体现对分割架构的全面理解

---

## 第 3 层：训练策略优化（体现工程调参能力）

### 3.1 损失函数替换
- baseline：CrossEntropyLoss + DiceLoss（仓库自带的）
- 实验路线：
  - CE + Dice → 纯 Tversky Loss（α, β 可调，针对 false positive/negative 的权衡）
  - Focal Tversky Loss（进一步关注难分样本）
- **报告写法**：展示类别不平衡问题 → 解释为什么 Dice/Tversky 能缓解 → 用消融表格证明

### 3.2 学习率策略
- baseline：ReduceLROnPlateau（仓库已有）
- 实验：CosineAnnealingLR、OneCycleLR
- 加 warmup（前 3-5 epoch 线性升到目标 lr）
- **报告写法**：画 lr schedule 曲线 + 对应 epoch 的 mIoU，解释 warmup 的作用

### 3.3 优化器对比
- baseline：RMSprop
- 实验：AdamW（带 weight decay）、SGD + momentum
- **报告写法**：同一配置下不同优化器的收敛速度对比

---

## 第 4 层：推理侧优化（体现完整 pipeline 意识）

### 4.1 Test-Time Augmentation (TTA)
- 推理时对图像做水平翻转 + 多尺度缩放 → 对每个变体做预测 → 平均 logits → 取 argmax
- 通常能免费涨 1-2 个 mIoU 点
- **报告写法**：展示 TTA 带来的 per-class IoU 提升，特别关注轮廓类的提升

### 4.2 阈值调优
- baseline：直接 argmax（阈值=0.5）
- 实验：在 validation set 上搜索最优类别阈值
- **报告写法**：不同阈值下的 IoU 折线图

---

## 第 5 层：可视化与分析（体现科研思维）

### 5.1 训练过程可视化
- loss 曲线（train + val，每个优化实验一张图）
- mIoU 曲线
- 学习率变化曲线

### 5.2 预测结果对比
- 选 6-8 张测试图，横向排列：
  - 原图 | GT | baseline 预测 | 各优化版本预测
  - 特别选几张 baseline 失败的图（轮廓模糊、小宠物、复杂背景）

### 5.3 Grad-CAM / 特征图可视化
- 用 `torchcam` 或手写 Grad-CAM
- 展示模型关注区域是否覆盖宠物主体
- **报告写法**：迁移学习后模型关注区域更准确 = 预训练特征的知识迁移

### 5.4 失败案例分析
- 挑出 IoU 最低的 5 张图
- 分析原因：遮挡、光照、相似颜色背景等
- **报告写法**：体现批判性思维，不是一味吹嘘结果

---

## 工作量汇总表

| 层级 | 改动 | 预期提升 | 报告页数 |
|------|------|---------|---------|
| 0. Baseline | 跑通原始仓库 | mIoU ~0.50 | 1-2 |
| 1.1 数据增强 | Albumentations | +2~4% | 2 |
| 1.2 数据分析 | 像素分布统计 | N/A（定性） | 1 |
| 2.1 预训练 Encoder | SMP pretrained | +5~10% | 3 |
| 2.2 多架构对比 | FPN/DeepLabV3+ | +1~3% | 2 |
| 3.1 损失函数 | Tversky/Focal | +2~3% | 2 |
| 3.2 LR 策略 | CosineAnneal | +1% | 1 |
| 4.1 TTA | 翻转+多尺度 | +1~2% | 1 |
| 5. 可视化分析 | Grad-CAM/失败案例 | N/A | 3-4 |

**总计：从 baseline 的 ~0.50 mIoU → 优化后的 ~0.65-0.75 mIoU，15-20 页报告内容**

---

## 推荐实验顺序

```
Step 0: 改 dataset → 跑通 baseline (1-2 小时)
Step 1: 加预训练 encoder (最大提升，最先做)
Step 2: 加 Albumentations 增强
Step 3: 换损失函数
Step 4: 调 LR schedule + 优化器
Step 5: 多架构对比 (FPN/DeepLab)
Step 6: TTA + 阈值调优
Step 7: 可视化 + 报告
```

前 3 步做完就已经有非常好的报告素材了。后面的都是加分项。
