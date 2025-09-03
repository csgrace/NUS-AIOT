import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. 类别映射定义
FALL_CLASSES = {
    0: "正常",
    1: "摔倒"
}

CLASS_NAMES = ["正常", "摔倒"]

def label_to_class_id(label):
    """将标签转换为类别ID"""
    if isinstance(label, str):
        if label.lower() in ['fall', '摔倒', '1']:
            return 1
        else:
            return 0
    return int(label)

# 2. 摔倒检测特征提取器
class FallFeatureExtractor:
    """针对摔倒检测的特征提取"""
    
    @staticmethod
    def extract_fall_features(acc_data):
        """提取摔倒检测特征"""
        # 确保输入是3D数组 [batch, seq_len, features]
        if acc_data.ndim == 2:
            # 如果是2D数组 [seq_len, features]，转换为3D
            acc_data = np.expand_dims(acc_data, 0)
        
        batch_size, seq_len, n_features = acc_data.shape
        features_list = []
        
        # 1. 原始加速度 (基础信息)
        features_list.append(acc_data)
        
        # 2. 加速度向量模长 (摔倒时会有显著变化)
        magnitude = np.sqrt(np.sum(acc_data**2, axis=-1, keepdims=True))
        features_list.append(magnitude)
        
        # 3. 垂直加速度 (摔倒的关键指标)
        vertical_acc = acc_data[:, :, 2:3] if acc_data.shape[-1] > 2 else acc_data[:, :, 0:1]
        features_list.append(vertical_acc)
        
        # 4. 加速度变化率 (jerk) - 摔倒时急剧变化
        jerk = np.diff(acc_data, axis=1, prepend=acc_data[:, :1, :])
        features_list.append(jerk)
        
        # 5. 加速度变化率的模长
        jerk_magnitude = np.sqrt(np.sum(jerk**2, axis=-1, keepdims=True))
        features_list.append(jerk_magnitude)
        
        # 6. 姿态变化指标 (倾斜角度变化)
        # 计算与重力方向的夹角变化
        if acc_data.shape[-1] > 2:
            gravity_angle = np.arctan2(
                np.sqrt(acc_data[:, :, 0:1]**2 + acc_data[:, :, 1:2]**2),
                acc_data[:, :, 2:3]
            )
            features_list.append(gravity_angle)
        
        # 7. 滑动窗口统计特征 (捕获摔倒过程)
        for window_size in [5, 10]:
            # 滑动平均
            rolling_mean = np.zeros_like(acc_data)
            # 滑动标准差
            rolling_std = np.zeros_like(acc_data)
            
            for i in range(batch_size):
                for j in range(acc_data.shape[-1]):
                    signal = acc_data[i, :, j]
                    # 简单的滑动窗口实现
                    padded_signal = np.pad(signal, window_size//2, mode='edge')
                    for k in range(seq_len):
                        window_data = padded_signal[k:k+window_size]
                        rolling_mean[i, k, j] = np.mean(window_data)
                        rolling_std[i, k, j] = np.std(window_data)
        
        features_list.append(rolling_mean)
        features_list.append(rolling_std)
    
        # 合并所有特征
        enhanced_features = np.concatenate(features_list, axis=-1)
    
        # 如果是单个样本，去掉批次维度
        if enhanced_features.shape[0] == 1:
            enhanced_features = np.squeeze(enhanced_features, 0)
        
        return enhanced_features

# 3. 摔倒检测模型
class FallDetectionClassifier(nn.Module):
    """摔倒检测二分类模型"""
    
    def __init__(self, input_size=20, sequence_length=100, num_classes=2):
        super(FallDetectionClassifier, self).__init__()
        
        # 多尺度特征提取 - 捕获摔倒的不同阶段
        self.short_term_features = nn.Sequential(
            # 捕获瞬时变化 (摔倒瞬间)
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.medium_term_features = nn.Sequential(
            # 捕获中期变化 (摔倒过程)
            nn.Conv1d(input_size, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.long_term_features = nn.Sequential(
            # 捕获长期趋势 (摔倒前后状态)
            nn.Conv1d(input_size, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=1),  # 3*32=96
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 时序建模 - 捕获摔倒的时序特征
        self.lstm = nn.LSTM(64, 32, batch_first=True, bidirectional=True, num_layers=2)
        
        # 注意力机制 - 关注关键时刻
        self.attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        
        # 二分类器
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)
        
        # 多尺度特征提取
        short_feat = self.short_term_features(x)
        medium_feat = self.medium_term_features(x)
        long_feat = self.long_term_features(x)
        
        # 特征拼接和融合
        combined = torch.cat([short_feat, medium_feat, long_feat], dim=1)
        fused = self.feature_fusion(combined)
        
        # 转换为时序格式
        fused = fused.transpose(1, 2)  # (batch_size, sequence_length, 64)
        
        # LSTM处理
        lstm_out, _ = self.lstm(fused)
        
        # 注意力机制
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, 64)
        
        # 二分类
        logits = self.classifier(pooled)
        
        return logits

# 4. 轻量级摔倒检测模型
class LightweightFallDetector(nn.Module):
    """轻量级摔倒检测器"""
    
    def __init__(self, input_size=20, sequence_length=100, num_classes=2):
        super(LightweightFallDetector, self).__init__()
        
        # 简化但有效的特征提取
        self.features = nn.Sequential(
            # 第一层：基础特征
            nn.Conv1d(input_size, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 第二层：组合特征
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 第三层：高级特征
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 全局平均池化
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 二分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        features = self.features(x)
        logits = self.classifier(features)
        return logits

# 5. 实时摔倒检测器 (用于部署)
class RealTimeFallDetector(nn.Module):
    """实时摔倒检测器 - 优化推理速度"""
    
    def __init__(self, input_size=20, sequence_length=100):
        super(RealTimeFallDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 2)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.features(x)

