# LaTeX 报告章节结构

## 章节安排

基于 Oxford-IIIT Pet 图像分割项目，报告共分为 6 章：

### 第 1 章：引言 (Introduction)
- **文件**: `body/01_introduction.tex`
- **内容**:
  - 研究背景与动机（图像分割的重要性、数据集挑战）
  - 研究目标（baseline → 优化，mIoU 提升）
  - 报告结构概述

### 第 2 章：数据集与评估指标 (Dataset & Metrics)
- **文件**: `body/02_dataset.tex`
- **内容**:
  - Oxford-IIIT Pet Dataset 介绍（规模、标注格式）
  - 数据集特点与挑战（类别不平衡、姿态多样性）
  - 评估指标定义（Dice、IoU、mIoU、Pixel Accuracy）

### 第 3 章：Baseline 模型构建 (Baseline Model)
- **文件**: `body/03_baseline.tex`
- **内容**:
  - U-Net 架构介绍（编码器-解码器结构）
  - 训练配置（超参数、损失函数、优化器）
  - Baseline 性能与问题分析（轮廓类 IoU 低、边缘模糊）

### 第 4 章：模型优化实验 (Optimization)
- **文件**: `body/04_optimization.tex`
- **内容**:
  - **数据侧优化**: Albumentations 高级增强、类别权重
  - **模型侧优化**: 预训练 Encoder（ResNet34 + ImageNet）
  - **训练策略优化**: Focal Tversky Loss、AdamW、Cosine LR、Early Stopping
  - **推理侧优化**: TTA、形态学后处理
  - 优化总结表（累积效果）

### 第 5 章：实验结果与分析 (Results & Analysis)
- **文件**: `body/05_results.tex`
- **内容**:
  - 定量结果对比（Baseline vs 优化模型）
  - Per-Class 性能分析（前景、背景、轮廓）
  - 定性结果展示（成功案例、失败案例）
  - 训练过程分析（loss 曲线、mIoU 曲线）
  - 消融实验（各优化项独立贡献）
  - 计算效率分析（训练时间、推理速度、显存占用）

### 第 6 章：总结与展望 (Conclusion)
- **文件**: `body/06_conclusion.tex`
- **内容**:
  - 工作总结（Baseline 构建、多层次优化、核心贡献）
  - 关键发现（迁移学习、类别不平衡、数据增强）
  - 局限性与未来工作（强遮挡、极端光照、实时性）
  - 潜在应用（医学图像、自动驾驶、遥感）
  - 个人收获与致谢

## 编译方式

```bash
cd Report
xelatex tjumain.tex
bibtex tjumain
xelatex tjumain.tex
xelatex tjumain.tex
```

或使用 latexmk：

```bash
latexmk -xelatex -synctex=1 -interaction=nonstopmode tjumain.tex
```

## 待填充内容（TODO）

在实际训练完成后，需要填充以下内容：

1. **第 2 章**：类别分布柱状图 (`figures/class_distribution.pdf`)
2. **第 3 章**：
   - Baseline 性能数值（表 3.1）
   - 训练曲线图 (`figures/baseline_curves.pdf`)
3. **第 4 章**：
   - 各优化项的性能数值（表 4.1-4.5）
   - 学习率曲线图 (`figures/lr_schedule.pdf`)
4. **第 5 章**：
   - 最终性能对比数值（表 5.1-5.4）
   - 成功/失败案例图 (`figures/success_cases.pdf`, `figures/failure_cases.pdf`)
   - 训练曲线对比图 (`figures/training_curves.pdf`)
   - 消融实验数值（表 5.5-5.6）
   - 计算效率数值（表 5.7）

## 图表命名规范

- **图片**: `figures/<chapter>_<name>.pdf` (例如 `figures/03_baseline_curves.pdf`)
- **表格**: 直接在 `.tex` 文件中用 `tabular` 环境编写
- **代码**: 使用 `lstlisting` 环境（已在 `setup/package.tex` 中配置）

## 参考文献

需要在 `references.bib` 中添加以下文献：

- U-Net 原论文 (Ronneberger et al., 2015)
- Oxford Pet Dataset (Parkhi et al., 2012)
- Tversky Loss (Salehi et al., 2017)
- Kaiming 初始化 (He et al., 2015)
- 其他引用的方法论文

## 注意事项

1. 所有数值保留 2 位小数（例如 0.XX）
2. 表格使用 `\caption{}` 和 `\label{}` 便于交叉引用
3. 图片使用 `\includegraphics[width=0.8\textwidth]{...}` 控制大小
4. 公式使用 `equation` 环境并编号
5. 代码片段使用 `lstlisting` 或 `verbatim` 环境
