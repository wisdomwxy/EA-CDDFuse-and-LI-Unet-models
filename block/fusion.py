import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块
    
    原理：通过多个分支以不同的感受野捕获特征，然后进行自适应加权融合
    1. 空间金字塔池化分支捕获多尺度上下文信息
    2. 通道注意力机制学习特征重要性
    3. 残差连接保持梯度传播
    """
    def __init__(self, in_channels, reduction=16):
        super(MultiScaleFeatureFusion, self).__init__()
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=rate, dilation=rate),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for rate in [1, 2, 3]  # 使用不同膨胀率的卷积
        ])
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels * 3 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 3 // reduction, in_channels * 3),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        
    def forward(self, x):
        # 多分支特征提取
        branch_outs = []
        for branch in self.branches:
            branch_outs.append(branch(x))
        
        # 拼接特征
        concat_feat = torch.cat(branch_outs, dim=1)
        
        # 通道注意力
        b, c, _, _ = concat_feat.size()
        y = self.avg_pool(concat_feat).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        concat_feat = concat_feat * y
        
        # 特征融合
        out = self.fusion(concat_feat)
        
        # 残差连接
        return out + x

class CrossLayerFeatureEnhancement(nn.Module):
    """跨层特征增强模块
    
    原理：通过高低层特征的交互增强特征表达
    1. 高层特征指导低层特征提取
    2. 低层特征补充高层特征细节
    3. 自适应权重平衡两种特征
    """
    def __init__(self, low_channels, high_channels):
        super(CrossLayerFeatureEnhancement, self).__init__()
        
        self.high_conv = nn.Conv2d(high_channels, low_channels, kernel_size=1)
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, low_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True)
        )
        
        # 自适应权重
        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(low_channels, 2, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, low_feat, high_feat):
        h, w = low_feat.size(2), low_feat.size(3)
        
        # 处理高层特征
        high_feat = F.interpolate(high_feat, size=(h, w), mode='bilinear', align_corners=True)
        high_feat = self.high_conv(high_feat)
        
        # 增强低层特征
        low_feat = self.low_conv(low_feat)
        
        # 自适应融合
        weights = self.weight(low_feat)
        out = weights[:, 0:1, :, :] * low_feat + weights[:, 1:, :, :] * high_feat
        
        return out

class FeatureRefinementBlock(nn.Module):
    """特征精炼模块
    
    原理：通过空间和通道维度的特征重建提升特征质量
    1. 空间注意力捕获关键区域
    2. 通道重建优化特征表达
    3. 残差学习保持特征稳定性
    """
    def __init__(self, channels):
        super(FeatureRefinementBlock, self).__init__()
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道重建
        self.channel_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
    def forward(self, x):
        # 空间注意力
        spatial_weight = self.spatial_attention(x)
        x = x * spatial_weight
        
        # 通道重建
        identity = x
        x = self.channel_transform(x)
        
        # 残差连接
        return x + identity 