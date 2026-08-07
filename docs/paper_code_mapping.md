# SpectraGate 论文与代码映射

本文档将论文中的每个组件映射到具体代码位置，便于复现与二次开发。

## 对应关系总览

论文：SpectraGate: Learnable Band Decomposition and State-Regularized Fusion for Visible-Infrared Object Detection
代码：本仓库，Ultralytics YOLO11 多模态改版，核心配置 `yolo11n-mm-mbsg-cm-cascade-V5.yaml`

| 论文组件 | 论文章节/公式 | 代码实现 | 代码位置 |
|---|---|---|---|
| 双分支骨干 RGB/IR | Stage 1，Eq. (1) | YAML backbone 双路（RGB 层 0-10，X 层 11-21），MultiModalRouter 按 `'RGB'`/`'X'` 标签路由 | `ultralytics/cfg/models/mm/11/yolo11n-mm-mbsg-cm-cascade-V5.yaml`；`ultralytics/nn/mm/router.py` |
| LMSG 模态内多频带谱门控 | Section III-C，Eq. (3)-(4) | LFEM_D_V5 低频增强模块，内含 _MBSG_V2 多频带谱门控 | `ultralytics/nn/modules/fusion/fredft.py`（LFEM_D_V5、_MBSG_V2） |
| CPSE 共轭交叉功率谱增强 | Section III-D，Eq. (5)-(7) | NLFEM_D_V5 非局部频率增强模块，跨谱共轭注意力 | `ultralytics/nn/modules/fusion/fredft.py`（NLFEM_D_V5） |
| CCSF 跨模态共轭谱融合 | Section III-E，Eq. (8)-(10) | FDFTM_V6，含 _CMFCA_V4 共轭跨谱注意力 + _FDFFN + 拼接投影 | `ultralytics/nn/modules/fusion/fredft.py`（FDFTM_V6、_CMFCA_V4、_FDFFN） |
| 可学习频带分解 | Section III-B，Eq. (2) | _MBSG_V2 内可学习截止频率，sigmoid 标量；parse_model 支持注入 fixed_cutoff | `ultralytics/nn/modules/fusion/fredft.py`；`ultralytics/nn/tasks.py` |
| FSCL 状态约束损失 | Section III-F，Eq. (11)-(16) | v8DetectionLossFusionV2 包装检测损失，收集 spectral/gate/cutoff 三项正则化；_FusionRegularizationMixin 提供正则化标量 | `ultralytics/utils/loss.py`；`ultralytics/nn/modules/fusion/fredft.py` |
| FPN + 多尺度检测头 | Stage 3 | YAML head 层 30-42，PAN 风格 + Detect 头 | `yolo11n-mm-mbsg-cm-cascade-V5.yaml` head 段 |

## 层级部署映射

论文：P3/P4 用 LMSG → CPSE → CCSF 三级级联，P5 用 LMSG → CCSF 两级级联，省略 CPSE。

| 特征层级 | 论文部署 | YAML 层 | 代码级联 |
|---|---|---|---|
| P3 | LMSG → CPSE → CCSF | 22-24 | LFEM_D_V5 → NLFEM_D_V5 → FDFTM_V6 |
| P4 | LMSG → CPSE → CCSF | 25-27 | LFEM_D_V5 → NLFEM_D_V5 → FDFTM_V6 |
| P5 | LMSG → CCSF | 28-29 | LFEM_D_V5 → FDFTM_V6 |

## 损失与训练超参

| 项目 | 论文 | 代码入口 |
|---|---|---|
| λ_s 谱平衡权重 | 0.03 | `trainMM.py --fusion-spectral-weight`（默认 0.03） |
| λ_g 门控激活权重 | 0.005 | 默认 0.005 |
| λ_c 截止频率权重 | 0.001 | 默认 0.001 |
| 优化器 | SGD，lr=1e-2 | `--optimizer SGD` |
| 轮数 | 200 | `--epochs 200` |
| 输入尺寸 | 640×640 | `--imgsz 640` |
| 模型规模 | m（10.14M 参数） | `--scale m` |

FSCL 三项正则化与论文公式对应：spectral 对应 R_spec 与 R_out，gate 对应 R_gate，cutoff 对应 R_cut。训练时仅在训练损失中累加，推理无额外开销。

## 数据加载

- 多模态数据由 `YOLOMultiModalImageDataset` 加载（`ultralytics/data/dataset.py`）
- 数据 YAML 用 `modality_used` 与 `modality` 字段配置 rgb/ir 双目录
- 推理时缺省模态自动零填充（`predictMM.py` 不传 `--ir` 时）

## 命名说明

论文模块名（LMSG/CPSE/CCSF）与代码类名（LFEM_D_V5/NLFEM_D_V5/FDFTM_V6）对应关系如上表。LFEM 为 Low-Frequency Enhancement Module，NLFEM 为 Non-Local Frequency Enhancement Module，命名延续频域融合文献的 FDFT 系列。
