# SpectraGate

**Learnable Band Decomposition and State-Regularized Fusion for Visible-Infrared Object Detection**

SpectraGate 是一个面向可见光-红外（RGB-IR）目标检测的频带感知渐进融合框架。核心思想是：把可学习的频带分解直接嵌入跨模态融合流程，让低频结构保持与高频细节甄别分别由不同的融合策略处理。

| | |
|---|---|
| 论文 | *SpectraGate: Learnable Band Decomposition and State-Regularized Fusion for Visible-Infrared Object Detection*（投稿中） |
| 任务 | RGB-Infrared Object Detection（DroneVehicle / LLVIP） |
| 基底框架 | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)（AGPL-3.0） |
| 许可证 | [AGPL-3.0](LICENSE) |

## 方法概览

三个级联融合模块构成渐进式融合管线，在 P3/P4 层级使用完整三级，P5 使用轻量两级：

```
                 LMSG             CPSE              CCSF
RGB ────┬───► ┌──────────┐ ┌──────────────┐ ┌──────────────┐
        │     │ 模态内多频 │ │ 跨模态共轭功率 │ │ 跨模态共轭谱  │
IR ─────┴───► │ 带谱门控  │→│ 谱增强(双向)  │→│ 融合→单流 F  │──► 检测头
              └──────────┘ └──────────────┘ └──────────────┘
```

- **LMSG（LFEM_D_V5）**：模态内多频带谱门控。基于跨模态 log 幅值差为低频/高频带生成独立门控，使模态内特征在跨模态交互前携带频带判别信息。
- **CPSE（NLFEM_D_V5）**：跨模态共轭交叉功率谱增强。把跨谱相位一致性转化为带符号空间门控，放大一致响应、抑制冲突响应。
- **CCSF（FDFTM_V6）**：双向频域交叉注意力 + 分带门控 + 多分支 FFN，把频带差异化双流特征压缩为单流检测表示。
- **FSCL（Fusion State Constraint Loss）**：约束内部融合状态，仅在频谱表示偏离有效区间时激活，防止截止频率坍缩与门控失活，不干扰检测主目标。

## 仓库结构

```
SpectraGate/
├── trainMM.py / valMM.py / predictMM.py / visMM.py / main_profile.py   # 训练/验证/推理/可视化入口
├── data_drone_mm.yaml / data_llvip_mm.yaml                             # 数据集配置
├── ultralytics/                                                        # 基于 8.3.163 的改版框架
│   └── nn/modules/fusion/fredft.py                                     # 核心：LMSG / CPSE / CCSF / FSCL
├── docs/paper_code_mapping.md                                          # 论文公式 ↔ 代码位置映射
└── scripts/download_datasets.md                                        # 数据集下载指引
```

## 环境安装

```bash
# Python 3.9+，CUDA 12.x
pip install -r requirements.txt
```

论文实验环境：PyTorch 2.8.0、单卡 RTX 5090。低显存设备可减小 batch 或使用 `--scale n`。

## 数据准备

DroneVehicle 与 LLVIP 均为公开数据集，需自行下载并按 YOLO 格式整理（见 `scripts/download_datasets.md`）。数据目录结构示例：

```
DroneVehicle_split/
├── images/          # RGB
│   ├── train/ val/ test/
├── images_ir/       # IR（与 RGB 同名文件）
│   ├── train/ val/ test/
└── labels/          # YOLO txt 标注
    ├── train/ val/ test/
```

## 快速开始

### 训练

```bash
# DroneVehicle，m 规模（论文配置），含 FSCL 辅助损失
python trainMM.py --data data_drone_mm.yaml --epochs 200 --batch 8 --name spectragate-drone

# 断点续训
python trainMM.py --resume runs/mm_train/spectragate-drone/weights/last.pt

# 消融：固定截止频率
python trainMM.py --fixed-cutoff 0.7 --name ablation-cutoff-0.7
```

### 验证

```bash
python valMM.py --weights runs/mm_train/spectragate-drone/weights/best.pt --data data_drone_mm.yaml
```

### 推理

```bash
# 双模态推理
python predictMM.py --weights best.pt --rgb img_rgb.png --ir img_ir.png --save

# 单模态消融（缺省模态自动零填充）
python predictMM.py --weights best.pt --rgb img_rgb.png
```

### 特征可视化

```bash
python visMM.py --weights best.pt --rgb img_rgb.png --ir img_ir.png --method feature --layers 22 23 24
```

## 论文与代码映射

| 论文组件 | 论文公式 | 代码类 | 代码位置 |
|---|---|---|---|
| 可学习频带分解 | Eq. (2)，`c_f = σ(θ_c)` | `_MBSG_V2._get_band_masks` / `_CMFCA_V4` | `ultralytics/nn/modules/fusion/fredft.py` |
| LMSG 模态内多频带谱门控 | Eq. (3)-(4) | `LFEM_D_V5`（内含 `_MBSG_V2`） | `fredft.py` |
| CPSE 共轭交叉功率谱增强 | Eq. (5)-(7) | `NLFEM_D_V5` | `fredft.py` |
| CCSF 跨模态共轭谱融合 | Eq. (8)-(10) | `FDFTM_V6`（内含 `_CMFCA_V4`、`_FDFFN`） | `fredft.py` |
| FSCL 状态约束损失 | Eq. (11)-(16) | `v8DetectionLossFusionV2` + `_FusionRegularizationMixin` | `ultralytics/utils/loss.py`；`fredft.py` |
| 双分支骨干 + 渐进级联部署 | Stage 1/2，P3/P4/P5 | `yolo11n-mm-mbsg-cm-cascade-V5.yaml` | `ultralytics/cfg/models/mm/11/` |
| 多模态数据路由 | Stage 1 | `MultiModalRouter` / `YOLOMultiModalImageDataset` | `ultralytics/nn/mm/router.py`；`ultralytics/data/dataset.py` |

详细映射见 `docs/paper_code_mapping.md`。

## 引用

```bibtex
@article{spectragate,
  title={SpectraGate: Learnable Band Decomposition and State-Regularized Fusion for Visible-Infrared Object Detection},
  author={Xu, Feng and Halimu, Yeerjiang and Mo, Zirui and Zhao, Yunjie and Lu, Yi and Dong, Luan},
  journal={IEEE Transactions on ...},
  year={2026}
}
```

（论文发表后补充正式引用信息。）

## 许可证

本项目基于 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)（AGPL-3.0）构建，采用 [AGPL-3.0](LICENSE) 协议发布。LMSG、CPSE、CCSF、FSCL 等融合模块为本项目原创代码。使用时请保留原始版权声明；商用前请确认 AGPL-3.0 对你的使用场景的适用性。

## 联系

Feng Xu：XuFeng967@outlook.com
