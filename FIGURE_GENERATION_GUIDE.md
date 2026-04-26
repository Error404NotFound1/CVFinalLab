# 报告图表生成指南

## 📊 图表生成流程

### 第 1 步：生成统计类图表

运行 `generate_figures.py` 生成训练曲线、消融实验等统计图表：

```bash
cd /Users/ybw/Mine/Labs/CVLab/finallab

python generate_figures.py \
    --baseline-dir baseline/Pytorch-UNet \
    --opt-dir optimization/Pytorch-UNet \
    --output-dir Report/figures
```

**生成的图表**：
- ✅ `02_class_distribution.pdf` - 类别分布柱状图
- ✅ `03_baseline_curves.pdf` - Baseline 训练曲线
- ✅ `05_training_curves.pdf` - 训练曲线对比
- ✅ `05_ablation.pdf` - 消融实验累积效果
- ✅ `05_per_class_iou.pdf` - Per-Class IoU 对比

---

### 第 2 步：生成预测对比图

运行 `generate_predictions.py` 生成模型预测结果对比：

```bash
python generate_predictions.py \
    --baseline-ckpt baseline/Pytorch-UNet/checkpoints/best.pth \
    --opt-ckpt optimization/Pytorch-UNet/checkpoints/best.pth \
    --data-root ./data \
    --output-dir Report/figures \
    --device cuda \
    --num-samples 6
```

**生成的图表**：
- ✅ `03_baseline_predictions.pdf` - Baseline 预测问题分析
- ✅ `05_success_cases.pdf` - 优化模型成功案例
- ✅ `05_failure_cases.pdf` - 失败案例分析
- ✅ `05_comparison.pdf` - Baseline vs 优化模型综合对比

---

### 第 3 步：手动补充数据集展示图（可选）

如果需要展示数据集样本，可以手动挑选几张图：

```python
import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet

dataset = OxfordIIITPet(root='./data', split='trainval', target_types='segmentation')

# 挑选索引
indices = [10, 50, 100, 200, 500, 1000]

fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for i, idx in enumerate(indices):
    image, mask = dataset[idx]
    axes[0, i].imshow(image)
    axes[0, i].axis('off')
    axes[1, i].imshow(mask)
    axes[1, i].axis('off')

plt.savefig('Report/figures/02_dataset_samples.pdf', dpi=300, bbox_inches='tight')
```

---

## 📝 填充报告数据

### 1. 更新消融实验数据

编辑 `generate_figures.py` 的第 147 行，填入实际的 mIoU 数据：

```python
# 第 147 行
miou = [0.50, 0.58, 0.62, 0.65, 0.67, 0.69, 0.71]  # 替换为实际数据
```

### 2. 更新 Per-Class IoU 数据

编辑 `generate_figures.py` 的第 180-181 行：

```python
# 第 180-181 行
baseline_iou = [0.65, 0.72, 0.23]  # 替换为实际数据
optimized_iou = [0.71, 0.78, 0.42]  # 替换为实际数据
```

### 3. 填充 LaTeX 表格数据

在 `Report/body/` 的各章节 `.tex` 文件中，搜索 `0.XX` 并替换为实际数值：

```bash
cd Report/body
grep -n "0.XX" *.tex
```

**需要填充的表格**：
- `03_baseline.tex` - 表 3.1（Baseline 性能）
- `04_optimization.tex` - 表 4.1-4.5（各优化项效果）
- `05_results.tex` - 表 5.1-5.7（最终对比、消融实验、计算效率）

---

## 🎨 图表质量检查清单

生成图表后，检查以下内容：

- [ ] 所有图表分辨率 ≥ 300 DPI
- [ ] 中文字体正常显示（无方框乱码）
- [ ] 坐标轴标签清晰可读
- [ ] 图例位置合理，不遮挡数据
- [ ] 颜色对比度足够（打印后仍清晰）
- [ ] 数值标注准确（小数点后 2-3 位）

---

## 🔧 常见问题

### Q1: 中文字体显示为方框？

**解决方案**：
```python
# 在 generate_figures.py 开头添加
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
# 或
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows/Linux
```

### Q2: 模型加载失败？

**检查**：
- checkpoint 路径是否正确
- 模型架构是否匹配（Baseline 用 UNet，优化版用 smp.Unet）
- `weights_only=False` 参数是否设置

### Q3: 显存不足？

**解决方案**：
```bash
# 减小 batch size 或使用 CPU
python generate_predictions.py --device cpu --num-samples 4
```

### Q4: 数据集未下载？

**解决方案**：
```bash
# 手动触发下载
python -c "from torchvision.datasets import OxfordIIITPet; OxfordIIITPet(root='./data', download=True)"
```

---

## 📂 最终文件结构

```
Report/figures/
├── 02_class_distribution.pdf      ← 类别分布
├── 02_dataset_samples.pdf         ← 数据集展示（可选）
├── 03_baseline_curves.pdf         ← Baseline 训练曲线
├── 03_baseline_predictions.pdf    ← Baseline 预测问题
├── 05_training_curves.pdf         ← 训练曲线对比
├── 05_ablation.pdf                ← 消融实验
├── 05_per_class_iou.pdf           ← Per-Class IoU
├── 05_comparison.pdf              ← 综合对比
├── 05_success_cases.pdf           ← 成功案例
└── 05_failure_cases.pdf           ← 失败案例
```

---

## ✅ 完成后

1. 编译 LaTeX 报告：
   ```bash
   cd Report
   xelatex tjumain.tex
   bibtex tjumain
   xelatex tjumain.tex
   xelatex tjumain.tex
   ```

2. 检查生成的 PDF：
   ```bash
   open tjumain.pdf  # macOS
   ```

3. 确认所有图表正确嵌入，表格数据完整。
