# 数据集下载指引

SpectraGate 论文的实验在 DroneVehicle 与 LLVIP 两个公开数据集上进行。两者均为第三方数据集，需自行下载并整理为 YOLO 格式。本仓库不随附任何数据。

## DroneVehicle

- 官方仓库：`tztztztztz/DroneVehicle`（GitHub，含数据集下载链接与官方划分）
- 内容：RGB + 红外 无人机视角车辆检测，5 类（car, truck, bus, van, freight_car），约 28k 帧
- 整理步骤：
  1. 从官方链接下载原始数据与划分文件
  2. 按官方划分生成 train/val/test 三个子集
  3. 将 XML 标注转换为 YOLO txt 格式（每行：`class x_center y_center w h`）
  4. RGB 图像放 `images/`，红外图像放 `images_ir/`（与 RGB 同名），标注放 `labels/`
  5. 修改 `data_drone_mm.yaml` 中的 `path` 指向数据集根目录

## LLVIP

- 官方仓库：`bupt-ai-cz/LLVIP`（GitHub，含数据集下载链接）
- 内容：RGB + 红外 夜间监控视角，15 万对配准图像，标注以 person 为主
- 整理步骤同上，修改 `data_llvip_mm.yaml` 中的 `path`

## 目录结构示例

```
DroneVehicle_split/
├── images/           # RGB
│   ├── train/ val/ test/
├── images_ir/        # IR（与 RGB 同名文件）
│   ├── train/ val/ test/
└── labels/           # YOLO txt 标注
    ├── train/ val/ test/
```

## 说明

- 两数据集的官方下载链接可能随维护状态变化，以官方仓库最新信息为准
- 训练脚本通过 `modality_used: ["rgb", "ir"]` 指定模态组合，数据加载由 `YOLOMultiModalImageDataset` 完成
