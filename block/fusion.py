import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MultiScaleFeatureFusion, self).__init__()
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=rate, dilation=rate),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for rate in [1, 2, 3] 
        ])
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels * 3 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 3 // reduction, in_channels * 3),
            nn.Sigmoid()
        )
        

        self.fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        
    def forward(self, x):
        branch_outs = []
        for branch in self.branches:
            branch_outs.append(branch(x))
        
        concat_feat = torch.cat(branch_outs, dim=1)
        
        b, c, _, _ = concat_feat.size()
        y = self.avg_pool(concat_feat).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        concat_feat = concat_feat * y
        
        # 特征融合
        out = self.fusion(concat_feat)
        
        return out + x

class CrossLayerFeatureEnhancement(nn.Module):
    def __init__(self, low_channels, high_channels):
        super(CrossLayerFeatureEnhancement, self).__init__()
        
        self.high_conv = nn.Conv2d(high_channels, low_channels, kernel_size=1)
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, low_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True)
        )
        
        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(low_channels, 2, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, low_feat, high_feat):
        h, w = low_feat.size(2), low_feat.size(3)
        
        high_feat = F.interpolate(high_feat, size=(h, w), mode='bilinear', align_corners=True)
        high_feat = self.high_conv(high_feat)
        
        low_feat = self.low_conv(low_feat)
        
        weights = self.weight(low_feat)
        out = weights[:, 0:1, :, :] * low_feat + weights[:, 1:, :, :] * high_feat
        
        return out

class FeatureRefinementBlock(nn.Module):
    def __init__(self, channels):
        super(FeatureRefinementBlock, self).__init__()
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.channel_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
    def forward(self, x):
        spatial_weight = self.spatial_attention(x)
        x = x * spatial_weight
        identity = x
        x = self.channel_transform(x)

        return x + identity 
