# Ultralytics YOLO, AGPL-3.0 license

"""
多模态分类验证器

本模块提供多模态分类任务的验证器，支持 RGB+X 模态的图像分类验证。
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch

from ultralytics.data import YOLOMultiModalClassifyDataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images
from ultralytics.nn.mm.utils import normalize_modality_token


class MultiModalClassificationValidator(BaseValidator):
    """
    多模态分类验证器

    继承 BaseValidator，提供多模态分类任务的验证流程，
    包括指标计算、混淆矩阵生成和结果可视化。

    Attributes:
        targets: 真实标签列表
        pred: 预测结果列表
        metrics: 分类指标对象
        names: 类别名称映射
        nc: 类别数量
        confusion_matrix: 混淆矩阵

    Methods:
        get_desc: 返回指标描述字符串
        init_metrics: 初始化指标
        preprocess: 预处理批次数据
        update_metrics: 更新指标
        build_dataset: 构建多模态分类数据集

    Examples:
        >>> from ultralytics.models.yolo.multimodal.classify import MultiModalClassificationValidator
        >>> validator = MultiModalClassificationValidator(args={"data": "mm_cls.yaml"})
        >>> validator()
    """

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None
    ) -> None:
        """
        初始化多模态分类验证器

        Args:
            dataloader: 数据加载器
            save_dir: 保存目录
            args: 配置参数
            _callbacks: 回调函数列表
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.targets = None
        self.pred = None

        # 仅对 rgb/x token 做归一化：rgb/RGB→RGB、x/X→X（其它模态名保持原样）
        self.modality = normalize_modality_token(getattr(self.args, "modality", None))
        if isinstance(self.args, dict):
            self.args["modality"] = self.modality
        else:
            setattr(self.args, "modality", self.modality)

        self.args.task = "classify"
        self.metrics = ClassifyMetrics()

    def get_desc(self) -> str:
        """返回指标描述字符串"""
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """初始化指标"""
        self.names = model.names
        self.nc = len(model.names)
        self.pred = []
        self.targets = []
        self.confusion_matrix = ConfusionMatrix(names=list(model.names.values()))

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """预处理批次数据"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds: torch.Tensor, batch: Dict[str, Any]) -> None:
        """更新指标"""
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(batch["cls"].type(torch.int32).cpu())

    def finalize_metrics(self) -> None:
        """完成指标计算"""
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,
                    normalize=normalize,
                    on_plot=self.on_plot
                )
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
        self.metrics.confusion_matrix = self.confusion_matrix

    def postprocess(
        self,
        preds: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
    ) -> torch.Tensor:
        """后处理预测结果"""
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def get_stats(self) -> Dict[str, float]:
        """计算并返回指标"""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def get_dataloader(self, dataset_path, batch_size):
        """构建并返回数据加载器"""
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def build_dataset(self, img_path: str) -> YOLOMultiModalClassifyDataset:
        """
        构建多模态分类数据集

        Args:
            img_path: 图像路径

        Returns:
            YOLOMultiModalClassifyDataset: 多模态分类数据集
        """
        # 获取数据配置
        if hasattr(self, 'data') and self.data:
            data = self.data
        else:
            data = check_det_dataset(self.args.data)

        return YOLOMultiModalClassifyDataset(
            img_path=img_path,
            data=data,
            args=self.args,
            augment=False,
            prefix=self.args.split
        )

    def print_results(self) -> None:
        """打印评估结果"""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """绘制验证样本"""
        batch["batch_idx"] = torch.arange(len(batch["img"]))
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self,
        batch: Dict[str, Any],
        preds: torch.Tensor,
        ni: int
    ) -> None:
        """绘制预测结果"""
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=torch.argmax(preds, dim=1),
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
