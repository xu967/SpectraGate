# SpectraGate YOLO 🚀, AGPL-3.0 license

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from spectragate.utils import LOGGER
from .self_modal_generator import SelfModalGenerator


class ModalityFiller:
    """
    多模态填充策略，用于单模态推理。
    
    为在双模态数据训练的模型上执行单模态推理时，
    提供各种智能填充策略来生成缺失的模态数据。
    """
    
    # 填充策略权重配置
    DEFAULT_STRATEGY_WEIGHTS = {
        'copy': 0.3,        # 直接复制原图像
        'noise': 0.25,      # 添加高斯噪声
        'channel_repeat': 0.2,  # 通道重复
        'edge_blur': 0.15,  # 边缘检测+模糊
        'mixed': 0.1        # 混合策略
    }
    
    def __init__(self, strategy_weights: Optional[Dict[str, float]] = None, 
                 noise_std: float = 0.1, blur_kernel_size: int = 5):
        """
        使用可配置策略初始化ModalityFiller。
        
        Args:
            strategy_weights (dict, optional): 不同填充策略的自定义权重
            noise_std (float): 高斯噪声的标准差
            blur_kernel_size (int): 模糊操作的核大小
        """
        self.strategy_weights = strategy_weights or self.DEFAULT_STRATEGY_WEIGHTS
        self.noise_std = noise_std
        self.blur_kernel_size = blur_kernel_size
        
        # 验证权重配置
        if abs(sum(self.strategy_weights.values()) - 1.0) > 1e-6:
            LOGGER.warning(f"策略权重总和不为1.0: {sum(self.strategy_weights.values())}")
    
    def generate_filling(self, source_tensor: torch.Tensor, 
                        source_modality: str, 
                        target_modality: str,
                        strategy: Optional[str] = None) -> torch.Tensor:
        """
        为缺失模态生成填充数据。
        
        Args:
            source_tensor (torch.Tensor): 源模态tensor [B, C, H, W]
            source_modality (str): 源模态类型 ('rgb', 'depth', 'thermal', etc.)
            target_modality (str): 要生成的目标模态类型
            strategy (str, optional): 要使用的特定策略，None时随机选择
            
        Returns:
            torch.Tensor: 生成的填充tensor，与源tensor形状相同
        """
        if strategy is None:
            strategy = self._select_random_strategy()
        
        if strategy == 'copy':
            return self._create_copy_fill(source_tensor)
        elif strategy == 'noise':
            return self._create_noise_fill(source_tensor)
        elif strategy == 'channel_repeat':
            return self._create_channel_repeat_fill(source_tensor)
        elif strategy == 'edge_blur':
            return self._create_edge_blur_fill(source_tensor)
        elif strategy == 'mixed':
            return self._create_mixed_fill(source_tensor)
        else:
            LOGGER.warning(f"未知填充策略: {strategy}, 使用复制策略")
            return self._create_copy_fill(source_tensor)
    
    def _select_random_strategy(self) -> str:
        """根据权重随机选择填充策略"""
        strategies = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())
        return random.choices(strategies, weights=weights)[0]
    
    def _create_copy_fill(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        直接复制原图像作为填充数据
        
        Args:
            tensor (torch.Tensor): 源模态tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: 复制的tensor
        """
        return tensor.clone()
    
    def _create_noise_fill(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        基于原图像添加高斯噪声作为填充数据
        
        Args:
            tensor (torch.Tensor): 源模态tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: 添加噪声的tensor
        """
        noise = torch.randn_like(tensor) * self.noise_std
        noisy_tensor = tensor + noise
        # 确保值在合理范围内 [0, 1]
        return torch.clamp(noisy_tensor, 0.0, 1.0)
    
    def _create_channel_repeat_fill(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        通道重复填充：将单通道重复到多通道
        
        Args:
            tensor (torch.Tensor): 源模态tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: 通道重复的tensor
        """
        # 转换为灰度（取平均）然后重复3通道
        if tensor.shape[1] == 3:
            grayscale = tensor.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            repeated = grayscale.repeat(1, 3, 1, 1)       # [B, 3, H, W]
            return repeated
        else:
            # 如果已经是单通道，直接重复
            return tensor.repeat(1, 3, 1, 1)
    
    def _create_edge_blur_fill(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        边缘检测+模糊处理作为填充数据
        
        Args:
            tensor (torch.Tensor): 源模态tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: 边缘模糊处理的tensor
        """
        # Sobel边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=tensor.dtype, device=tensor.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=tensor.dtype, device=tensor.device).unsqueeze(0).unsqueeze(0)
        
        # 对每个通道进行边缘检测
        edges = []
        for c in range(tensor.shape[1]):
            channel = tensor[:, c:c+1, :, :]
            edge_x = F.conv2d(channel, sobel_x, padding=1)
            edge_y = F.conv2d(channel, sobel_y, padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            edges.append(edge_magnitude)
        
        edge_tensor = torch.cat(edges, dim=1)
        
        # 应用高斯模糊
        return self._apply_gaussian_blur(edge_tensor)
    
    def _create_mixed_fill(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        混合多种策略的填充数据
        
        Args:
            tensor (torch.Tensor): 源模态tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: 混合策略处理的tensor
        """
        # 随机选择2-3种策略进行组合
        available_strategies = ['copy', 'noise', 'channel_repeat', 'edge_blur']
        selected_strategies = random.sample(available_strategies, 
                                          random.randint(2, min(3, len(available_strategies))))
        
        results = []
        for strategy in selected_strategies:
            if strategy == 'copy':
                results.append(self._create_copy_fill(tensor))
            elif strategy == 'noise':
                results.append(self._create_noise_fill(tensor))
            elif strategy == 'channel_repeat':
                results.append(self._create_channel_repeat_fill(tensor))
            elif strategy == 'edge_blur':
                results.append(self._create_edge_blur_fill(tensor))
        
        # 加权平均组合
        weights = torch.softmax(torch.rand(len(results)), dim=0)
        mixed_result = torch.zeros_like(tensor)
        for i, result in enumerate(results):
            mixed_result += weights[i] * result
        
        return mixed_result
    
    def _apply_gaussian_blur(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        应用高斯模糊
        
        Args:
            tensor (torch.Tensor): 输入tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: 模糊处理后的tensor
        """
        # 创建高斯核
        kernel_size = self.blur_kernel_size
        sigma = kernel_size / 3.0
        
        # 生成1D高斯核
        x = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device) - kernel_size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # 创建2D高斯核
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        
        # 对每个通道分别应用模糊
        blurred_channels = []
        for c in range(tensor.shape[1]):
            channel = tensor[:, c:c+1, :, :]
            blurred = F.conv2d(channel, gaussian_2d, padding=kernel_size//2)
            blurred_channels.append(blurred)
        
        return torch.cat(blurred_channels, dim=1)
    
    def get_statistics(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        计算tensor的统计信息，用于验证填充质量
        
        Args:
            tensor (torch.Tensor): 输入tensor
            
        Returns:
            dict: 包含均值、标准差、最大值、最小值的统计信息
        """
        return {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'max': tensor.max().item(),
            'min': tensor.min().item(),
            'shape': list(tensor.shape)
        }


# 全局实例，供其他模块使用
default_modality_filler = ModalityFiller()
default_self_modal_generator = SelfModalGenerator()


def generate_modality_filling(source_tensor: torch.Tensor,
                            source_modality: str,
                            target_modality: str,
                            strategy: Optional[str] = None,
                            filler: Optional[ModalityFiller] = None) -> torch.Tensor:
    """
    便捷函数：生成模态填充数据

    Args:
        source_tensor (torch.Tensor): 源模态tensor
        source_modality (str): 源模态类型
        target_modality (str): 目标模态类型
        strategy (str, optional): 填充策略
        filler (ModalityFiller, optional): 自定义填充器

    Returns:
        torch.Tensor: 生成的填充tensor
    """
    if filler is None:
        filler = default_modality_filler

    return filler.generate_filling(source_tensor, source_modality, target_modality, strategy)


def generate_self_modality(rgb_tensor: torch.Tensor, modal_type: str = 'edge',
                          algorithm: str = 'auto',
                          generator: Optional[SelfModalGenerator] = None) -> torch.Tensor:
    """
    便捷函数：生成自体模态数据

    Args:
        rgb_tensor (torch.Tensor): RGB输入tensor [B, 3, H, W]
        modal_type (str): 模态类型 ('edge', 'texture', 'gradient')
        algorithm (str): 具体算法
        generator (SelfModalGenerator, optional): 自定义生成器

    Returns:
        torch.Tensor: 生成的自体模态tensor [B, 3, H, W]
    """
    if generator is None:
        generator = default_self_modal_generator

    return generator.generate_self_modality(rgb_tensor, modal_type, algorithm)
