import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. 类别映射定义
# 您的具体类别映射
STEP_CLASSES = {
    0: 0,
    1: 1,
    2: 1.5,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6
}

CLASS_NAMES = ["0步", "1步", "1.5步", "2步", "3步", "4步", "5步", "6步"]

# 反向映射，用于从步数查找ID
VALUE_TO_ID = {v: k for k, v in STEP_CLASSES.items()}

def steps_to_class_id(steps):
    """将步数转换为类别ID，对意外输入有鲁棒性"""
    if steps in VALUE_TO_ID:
        return VALUE_TO_ID[steps]
    else:
        # Fallback：找到预定义步数中最接近的一个
        available_steps = list(VALUE_TO_ID.keys())
        closest_step = min(available_steps, key=lambda x: abs(x - steps))
        return VALUE_TO_ID[closest_step]

# 3. 特征工程优化
class StepFeatureExtractor:
    """针对细粒度步数分类的特征提取"""
    
    @staticmethod
    def extract_fine_grained_features(acc_data):
        """提取细粒度特征"""
        if acc_data.ndim == 2:
            acc_data = np.expand_dims(acc_data, 0)
            
        batch_size, seq_len, _ = acc_data.shape
        features_list = []
        
        # 1. 原始加速度 (保留原始信息)
        features_list.append(acc_data)
        
        # 2. 加速度向量模长
        magnitude = np.sqrt(np.sum(acc_data**2, axis=-1, keepdims=True))
        features_list.append(magnitude)
        
        # 3. 垂直加速度 (步数检测的关键)
        vertical_acc = acc_data[:, :, 2:3]  # Z轴
        features_list.append(vertical_acc)
        
        # 4. 加速度变化率 (jerk) - 对细微差别敏感
        jerk = np.diff(acc_data, axis=1, prepend=acc_data[:, :1, :])
        features_list.append(jerk)
        
        # 5. 二阶导数 (加速度的变化率)
        second_derivative = np.diff(jerk, axis=1, prepend=jerk[:, :1, :])
        features_list.append(second_derivative)
        
        # 6. 滑动窗口统计特征 (多个窗口大小)
        for window_size in [3, 5, 7]:
            rolling_features = np.zeros_like(acc_data)
            for i in range(batch_size):
                for j in range(3):
                    rolling_features[i, :, j] = np.convolve(
                        acc_data[i, :, j], 
                        np.ones(window_size)/window_size, 
                        mode='same'
                    )
            features_list.append(rolling_features)
        
        # 7. 峰值检测特征 (步数的直接指标)
        peak_features = np.zeros((batch_size, seq_len, 1))
        for i in range(batch_size):
            signal = magnitude[i, :, 0]
            # 动态阈值
            threshold = np.mean(signal) + 0.3 * np.std(signal)
            peaks = (signal > threshold).astype(float)
            peak_features[i, :, 0] = peaks
        features_list.append(peak_features)
        
        # 8. 频域特征 (步频信息)
        fft_features = np.zeros((batch_size, seq_len, 3))
        for i in range(batch_size):
            for j in range(3):
                fft_signal = np.fft.fft(acc_data[i, :, j])
                fft_magnitude = np.abs(fft_signal)
                # 取前一半频率分量并插值到原长度
                half_len = len(fft_magnitude) // 2
                fft_features[i, :, j] = np.interp(
                    np.linspace(0, half_len-1, seq_len),
                    np.arange(half_len),
                    fft_magnitude[:half_len]
                )
        features_list.append(fft_features)
        
        # 合并所有特征
        enhanced_features = np.concatenate(features_list, axis=-1)
        if enhanced_features.shape[0] == 1:
            enhanced_features = np.squeeze(enhanced_features, 0)
        return enhanced_features

# 2. 针对细粒度分类的模型架构
class FineGrainedStepClassifier(nn.Module):
    """细粒度步数分类模型"""
    
    def __init__(self, input_size=24, sequence_length=100, num_classes=8):
        super(FineGrainedStepClassifier, self).__init__()
        
        # 多分辨率特征提取 - 对细微差别敏感
        self.micro_features = nn.Sequential(
            # 捕获微小变化 (1.5步 vs 2步的细微差别)
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.local_features = nn.Sequential(
            # 捕获局部模式
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.global_features = nn.Sequential(
            # 捕获整体趋势
            nn.Conv1d(input_size, 32, kernel_size=7, padding=6, dilation=2),
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
        
        # 时序建模 - 双向LSTM捕获前后依赖
        self.lstm = nn.LSTM(64, 32, batch_first=True, bidirectional=True, num_layers=2)
        
        # 自注意力机制 - 关注关键时刻
        self.self_attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        
        # 分类器 - 针对细粒度分类
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)
        
        # 多分辨率特征提取
        micro_feat = self.micro_features(x)
        local_feat = self.local_features(x)
        global_feat = self.global_features(x)
        
        # 特征拼接和融合
        combined = torch.cat([micro_feat, local_feat, global_feat], dim=1)
        fused = self.feature_fusion(combined)
        
        # 转换为时序格式
        fused = fused.transpose(1, 2)  # (batch_size, sequence_length, 64)
        
        # LSTM处理
        lstm_out, _ = self.lstm(fused)
        
        # 自注意力
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, 64)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits

# 轻量级版本
class LightweightFineGrainedClassifier(nn.Module):
    """轻量级细粒度分类器"""
    
    def __init__(self, input_size=24, sequence_length=100, num_classes=8):
        super(LightweightFineGrainedClassifier, self).__init__()
        
        # 简化但有效的特征提取
        self.features = nn.Sequential(
            # 第一层：基础特征
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 第二层：组合特征
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 第三层：高级特征
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # 全局平均池化
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
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
