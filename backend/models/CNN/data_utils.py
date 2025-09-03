import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 设置全局字体
import seaborn as sns
from scipy import signal
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

class AccelerometerDataset(Dataset):
    """加速度计数据集"""
    
    def __init__(self, data, labels, sequence_length=100, transform=None):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform
        
        # 对于短序列，我们不需要创建滑动窗口，直接使用原始数据
        if data.shape[1] <= sequence_length:
            self.sequences = data  # 这里data的形状应该是[n_samples, seq_length, features]
            self.sequence_labels = labels
        else:
            # 创建滑动窗口
            self.sequences, self.sequence_labels = self._create_sequences()
    
    def _create_sequences(self):
        """创建滑动窗口序列"""
        sequences = []
        sequence_labels = []
        
        for i in range(len(self.data) - self.sequence_length + 1):
            seq = self.data[i:i + self.sequence_length]
            label = self.labels[i + self.sequence_length - 1]  # 使用窗口末尾的标签
            
            sequences.append(seq)
            sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # 对于单样本，形状应该是[seq_length, features]
        sequence = self.sequences[idx].astype(np.float32)
        label = self.sequence_labels[idx].astype(np.float32)
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return torch.tensor(sequence), torch.tensor(label)

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        """初始化"""
        # 初始化标准化器
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data, normalize=True, filter_data=True, short_sequence=False):
        """预处理数据"""
        # 复制数据避免修改原始数据
        processed_data = data.copy()
        
        if filter_data and not short_sequence:
            # 应用低通滤波器
            processed_data = self._apply_lowpass_filter(processed_data)
        
        if normalize:
            # 标准化数据
            processed_data = self._normalize_data(processed_data)
        
        return processed_data
    
    def _normalize_data(self, data):
        """标准化数据"""
        # 对每个特征单独标准化
        data_reshaped = data.reshape(-1, data.shape[-1])
        normalized_data = self.scaler.fit_transform(data_reshaped)
        
        return normalized_data.reshape(data.shape)
    
    def _apply_lowpass_filter(self, data, cutoff=10.0, fs=100.0):
        """应用低通滤波器"""
        # 设计巴特沃斯低通滤波器
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(4, normal_cutoff, btype='low')
        
        # 对每个轴单独应用滤波器
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[-1]):
            filtered_data[..., i] = signal.filtfilt(b, a, data[..., i])
            
        return filtered_data
    
    def augment_data(self, data, labels, augment_factor=1):
        """数据增强"""
        if augment_factor <= 0:
            return data, labels
            
        original_data = data.copy()
        original_labels = labels.copy()
        
        augmented_data = []
        augmented_labels = []
        
        # 添加原始数据
        augmented_data.append(original_data)
        augmented_labels.append(original_labels)
        
        # 添加噪声
        if augment_factor > 0:
            noisy_data = self._add_noise(original_data, noise_level=0.05)
            augmented_data.append(noisy_data)
            augmented_labels.append(original_labels)
        
        # 时间扭曲
        if augment_factor > 1:
            warped_data = self._time_warp(original_data)
            augmented_data.append(warped_data)
            augmented_labels.append(original_labels)
        
        # 振幅缩放
        if augment_factor > 2:
            scaled_data = self._scale_amplitude(original_data)
            augmented_data.append(scaled_data)
            augmented_labels.append(original_labels)
        
        # 随机旋转（对于加速度数据有意义）
        if augment_factor > 3 and original_data.shape[-1] >= 3:
            rotated_data = self._random_rotation(original_data)
            augmented_data.append(rotated_data)
            augmented_labels.append(original_labels)
            
        # 随机偏移
        if augment_factor > 4:
            shifted_data = self._random_shift(original_data)
            augmented_data.append(shifted_data)
            augmented_labels.append(original_labels)
        
        # 随机擦除
        if augment_factor > 5:
            erased_data = self._random_erasing(original_data)
            augmented_data.append(erased_data)
            augmented_labels.append(original_labels)
            
        # 组合增强方法
        if augment_factor > 6:
            combined_data = self._add_noise(self._time_warp(original_data), noise_level=0.03)
            augmented_data.append(combined_data)
            augmented_labels.append(original_labels)
            
        # 混合数据增强
        if augment_factor > 7:
            for i in range(min(2, augment_factor - 7)):
                mixed_data = self._mix_samples(original_data)
                mixed_labels = original_labels
                augmented_data.append(mixed_data)
                augmented_labels.append(mixed_labels)
        
        # 连接所有增强的数据
        augmented_data = np.vstack(augmented_data)
        augmented_labels = np.concatenate(augmented_labels)
        
        return augmented_data, augmented_labels
    
    def _add_noise(self, data, noise_level=0.05):
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    
    def _time_warp(self, data, sigma=0.2):
        """时间扭曲"""
        # 对每个序列单独处理
        warped_data = np.zeros_like(data)
        
        for i in range(len(data)):
            # 获取当前序列
            seq = data[i]
            seq_len = len(seq)
            
            # 创建时间扭曲函数（累积正态分布）
            time_warp = np.random.normal(0, sigma, seq_len)
            time_warp = np.cumsum(time_warp)
            
            # 标准化到[0, seq_len-1]
            time_warp = (time_warp - time_warp.min()) / (time_warp.max() - time_warp.min()) * (seq_len - 1)
            
            # 对每个维度单独插值
            for j in range(seq.shape[1]):
                channel = seq[:, j]
                warped_channel = np.interp(np.arange(seq_len), time_warp, channel)
                warped_data[i, :, j] = warped_channel
        
        return warped_data
    
    def _scale_amplitude(self, data, scale_range=(0.8, 1.2)):
        """随机缩放振幅"""
        scales = np.random.uniform(scale_range[0], scale_range[1], size=(len(data), 1, data.shape[-1]))
        return data * scales
    
    def _random_rotation(self, data):
        """随机旋转（针对三轴加速度数据）"""
        rotated_data = np.zeros_like(data)
        
        for i in range(len(data)):
            # 生成随机旋转矩阵
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            psi = np.random.uniform(0, 2*np.pi)
            
            # 旋转矩阵 - X轴旋转
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
            
            # 旋转矩阵 - Y轴旋转
            R_y = np.array([
                [np.cos(phi), 0, np.sin(phi)],
                [0, 1, 0],
                [-np.sin(phi), 0, np.cos(phi)]
            ])
            
            # 旋转矩阵 - Z轴旋转
            R_z = np.array([
                [np.cos(psi), -np.sin(psi), 0],
                [np.sin(psi), np.cos(psi), 0],
                [0, 0, 1]
            ])
            
            # 组合旋转矩阵
            R = np.dot(np.dot(R_z, R_y), R_x)
            
            # 应用旋转
            xyz_data = data[i, :, :3]  # 假设前三列是x, y, z
            rotated_xyz = np.dot(xyz_data, R.T)
            
            # 保存旋转后的数据
            rotated_data[i, :, :3] = rotated_xyz
            
            # 如果有额外的特征，保持不变
            if data.shape[-1] > 3:
                rotated_data[i, :, 3:] = data[i, :, 3:]
        
        return rotated_data
    
    def _random_shift(self, data, shift_range=(-0.1, 0.1)):
        """随机偏移"""
        shifts = np.random.uniform(shift_range[0], shift_range[1], size=(len(data), 1, data.shape[-1]))
        return data + shifts
    
    def _mix_samples(self, data, alpha=0.2):
        """混合不同样本"""
        mixed_data = data.copy()
        
        # 随机打乱索引
        idx = np.random.permutation(len(data))
        
        # 混合样本
        lam = np.random.beta(alpha, alpha)
        mixed_data = lam * data + (1 - lam) * data[idx]
        
        return mixed_data
    
    def _random_erasing(self, data, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        """随机擦除增强
        
        参数:
            data: 输入数据，形状为[N, T, C]
            p: 每个样本应用擦除的概率
            scale: 擦除区域相对于整个序列的比例范围
            ratio: 擦除区域的宽高比范围
        """
        erased_data = data.copy()
        
        for i in range(len(data)):
            # 以概率p应用擦除
            if np.random.random() > p:
                continue
                
            # 获取序列长度和通道数
            seq_len, n_channels = data[i].shape
            
            # 确定擦除区域大小
            erase_area = np.random.uniform(*scale) * seq_len
            aspect_ratio = np.random.uniform(*ratio)
            
            erase_length = int(np.sqrt(erase_area * aspect_ratio))
            
            # 确保擦除长度在合理范围内
            erase_length = min(erase_length, seq_len // 2)
            if erase_length < 1:
                continue
                
            # 随机选择起始位置
            start_idx = np.random.randint(0, seq_len - erase_length + 1)
            
            # 生成随机噪声值
            noise_value = np.random.normal(0, 0.1, size=(erase_length, n_channels))
            
            # 应用擦除
            erased_data[i, start_idx:start_idx+erase_length, :] = noise_value
            
        return erased_data

def generate_sample_data(n_samples=10000, sequence_length=100, input_size=3, 
                        step_range=(0, 50), fs=50):
    """生成模拟的加速度计数据和步数标签"""
    np.random.seed(42)
    
    all_data = []
    all_labels = []
    
    for i in range(n_samples):
        # 生成基础信号
        t = np.linspace(0, sequence_length/fs, sequence_length)
        
        # 模拟步行信号 - 周期性模式
        step_freq = np.random.uniform(1.5, 2.5)  # 步频 Hz
        step_amplitude = np.random.uniform(0.5, 2.0)
        
        # 生成三轴加速度数据
        x_acc = step_amplitude * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0, 0.1, len(t))
        y_acc = step_amplitude * np.cos(2 * np.pi * step_freq * t) + np.random.normal(0, 0.1, len(t))
        z_acc = step_amplitude * np.sin(2 * np.pi * step_freq * t + np.pi/4) + np.random.normal(0, 0.1, len(t))
        
        # 添加重力分量
        z_acc += 9.8
        
        # 组合数据
        data = np.column_stack([x_acc, y_acc, z_acc])
        
        # 计算步数 - 基于峰值检测
        magnitude = np.sqrt(np.sum(data**2, axis=1))
        peaks, _ = signal.find_peaks(magnitude, height=np.mean(magnitude), distance=fs//4)
        step_count = len(peaks)
        
        # 添加一些随机性
        step_count += np.random.randint(-2, 3)
        step_count = max(0, min(step_count, step_range[1]))
        
        all_data.append(data)
        all_labels.append(step_count)
    
    return np.array(all_data), np.array(all_labels)

def create_data_loaders(data, labels, sequence_length=100, batch_size=32, 
                       test_size=0.2, val_size=0.1, augment=True, classification=False):
    """创建数据加载器
    
    参数:
    - data: 输入数据
    - labels: 标签数据
    - sequence_length: 序列长度
    - batch_size: 批次大小
    - test_size: 测试集比例
    - val_size: 验证集比例
    - augment: 是否使用数据增强
    - classification: 是否为分类任务（如果是，将标签转为长整型）
    """
    
    # 分割数据
    # 如果是分类任务，使用分层采样确保标签分布平衡
    if classification:
        X_temp, X_test, y_temp, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=42, stratify=None
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
        )
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    
    # 检查是否为短序列
    is_short_sequence = sequence_length <= 30
    
    # 处理训练数据
    X_train_processed = []
    y_train_processed = []
    
    print("训练数据原始形状:", X_train.shape)
    
    for i in range(len(X_train)):
        # 确保输入数据形状正确
        if len(X_train[i].shape) == 3 and X_train[i].shape[0] == sequence_length:
            # 如果数据形状已经是[sequence_length, features]，则需要展平
            current_data = X_train[i].reshape(X_train[i].shape[0], -1)
        else:
            current_data = X_train[i]
            
        processed = preprocessor.preprocess_data(
            current_data, 
            short_sequence=is_short_sequence
        )
        if len(processed) >= sequence_length:  # 确保序列长度足够
            X_train_processed.append(processed)
            y_train_processed.append(y_train[i])
    
    X_train_processed = np.array(X_train_processed)
    y_train_processed = np.array(y_train_processed)
    
    print("处理后的训练数据形状:", X_train_processed.shape)
    
    # 数据增强
    if augment:
        X_train_processed, y_train_processed = preprocessor.augment_data(
            X_train_processed, y_train_processed, augment_factor=1
        )
        print("增强后的训练数据形状:", X_train_processed.shape)
    
    # 处理验证和测试数据
    X_val_processed = []
    y_val_processed = []
    for i in range(len(X_val)):
        # 确保输入数据形状正确
        if len(X_val[i].shape) == 3 and X_val[i].shape[0] == sequence_length:
            # 如果数据形状已经是[sequence_length, features]，则需要展平
            current_data = X_val[i].reshape(X_val[i].shape[0], -1)
        else:
            current_data = X_val[i]
            
        processed = preprocessor.preprocess_data(
            current_data,
            short_sequence=is_short_sequence
        )
        if len(processed) >= sequence_length:
            X_val_processed.append(processed)
            y_val_processed.append(y_val[i])
    
    # 转换为numpy数组
    X_val_processed = np.array(X_val_processed)
    y_val_processed = np.array(y_val_processed)
    
    X_test_processed = []
    y_test_processed = []
    for i in range(len(X_test)):
        # 确保输入数据形状正确
        if len(X_test[i].shape) == 3 and X_test[i].shape[0] == sequence_length:
            # 如果数据形状已经是[sequence_length, features]，则需要展平
            current_data = X_test[i].reshape(X_test[i].shape[0], -1)
        else:
            current_data = X_test[i]
            
        processed = preprocessor.preprocess_data(
            current_data,
            short_sequence=is_short_sequence
        )
        if len(processed) >= sequence_length:
            X_test_processed.append(processed)
            y_test_processed.append(y_test[i])
    
    # 转换为numpy数组
    X_test_processed = np.array(X_test_processed)
    y_test_processed = np.array(y_test_processed)
    
    # 打印处理后的数据形状
    print("验证数据形状:", X_val_processed.shape)
    print("测试数据形状:", X_test_processed.shape)
    
    # 对于分类任务，确保标签是整数类型
    if classification:
        y_train_processed = y_train_processed.astype(np.int64)
        y_val_processed = y_val_processed.astype(np.int64)
        y_test_processed = y_test_processed.astype(np.int64)
        print("分类任务：标签已转换为整数类型")
        
        # 打印每个集合中不同类别的数量
        for name, y_data in [("训练集", y_train_processed), ("验证集", y_val_processed), ("测试集", y_test_processed)]:
            unique, counts = np.unique(y_data, return_counts=True)
            print(f"{name}中各类别分布:")
            for cls, count in zip(unique, counts):
                print(f"  类别 {cls}: {count}个样本 ({count/len(y_data)*100:.2f}%)")
    
    # 创建数据集
    train_dataset = AccelerometerDataset(X_train_processed, y_train_processed, sequence_length)
    val_dataset = AccelerometerDataset(X_val_processed, y_val_processed, sequence_length)
    test_dataset = AccelerometerDataset(X_test_processed, y_test_processed, sequence_length)
    
    # 创建数据加载器 - 禁用多进程
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    # 检查一个批次的数据形状
    for data, labels in train_loader:
        print("批次数据形状:", data.shape)
        print("批次标签形状:", labels.shape)
        break
    
    return train_loader, val_loader, test_loader, preprocessor

def visualize_data(data, labels, n_samples=5):
    """可视化数据样本"""
    # 设置字体，确保能显示
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 3*n_samples))
    
    for i in range(n_samples):
        idx = np.random.randint(0, len(data))
        sample_data = data[idx]
        sample_label = labels[idx]
        
        # 创建x轴刻度（从1开始）
        x_ticks = np.arange(1, len(sample_data) + 1)
        
        # 绘制原始信号
        axes[i, 0].plot(x_ticks, sample_data[:, 0], label='X-axis', alpha=0.7)
        axes[i, 0].plot(x_ticks, sample_data[:, 1], label='Y-axis', alpha=0.7)
        axes[i, 0].plot(x_ticks, sample_data[:, 2], label='Z-axis', alpha=0.7)
        axes[i, 0].set_title(f'Sample {i+1} - Acceleration (Steps: {sample_label})')
        axes[i, 0].set_xlabel('Data Point')
        axes[i, 0].set_ylabel('Acceleration')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # 设置整数刻度，确保1到20都显示
        axes[i, 0].set_xticks(range(1, len(sample_data)+1))
        axes[i, 0].set_xlim(0.5, len(sample_data) + 0.5)
        
        # 绘制幅值
        magnitude = np.sqrt(np.sum(sample_data**2, axis=1))
        axes[i, 1].plot(x_ticks, magnitude, 'r-', linewidth=2)
        axes[i, 1].set_title('Acceleration Magnitude')
        axes[i, 1].set_xlabel('Data Point')
        axes[i, 1].set_ylabel('Magnitude')
        axes[i, 1].grid(True)
        
        # 设置整数刻度，确保1到20都显示
        axes[i, 1].set_xticks(range(1, len(sample_data)+1))
        axes[i, 1].set_xlim(0.5, len(sample_data) + 0.5)
    
    plt.tight_layout()
    plt.show()

def analyze_data_distribution(labels):
    """分析数据分布"""
    # 设置字体，确保能显示
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    # 计算最小和最大步数，以创建整数间隔
    min_steps = int(np.min(labels))
    max_steps = int(np.max(labels))
    bins = np.arange(min_steps, max_steps + 2) - 0.5  # +2是为了包括最大值，-0.5是为了bins边界
    
    # 绘制直方图
    n, bins, patches = plt.hist(labels, bins=bins, alpha=0.7, edgecolor='black', 
                               color='skyblue', rwidth=0.8)
    
    # 为每个条添加数值标签
    unique_labels, counts = np.unique(labels, return_counts=True)
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        plt.text(label, count + max(counts)*0.02, str(count), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.title('Steps Distribution')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(min_steps, max_steps+1, 1))  # 设置整数刻度
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(labels)
    plt.title('Steps Boxplot')
    plt.ylabel('Steps')
    plt.grid(True, alpha=0.3)
    
    # 添加散点展示数据分布
    x = np.random.normal(1, 0.1, size=len(labels))
    plt.scatter(x, labels, alpha=0.4, color='green', s=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Data Statistics:")
    print(f"Total Samples: {len(labels)}")
    print(f"Steps Range: {labels.min()} - {labels.max()}")
    print(f"Average Steps: {labels.mean():.2f}")
    print(f"Steps Standard Deviation: {labels.std():.2f}")
