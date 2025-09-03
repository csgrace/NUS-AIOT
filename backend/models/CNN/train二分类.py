import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torch.nn.functional as F
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

# 也添加当前目录到路径，以便直接导入
sys.path.append(current_dir)

try:
    # 尝试相对导入
    from model_binary import (
        FineGrainedStepClassifier,
        LightweightFineGrainedClassifier,
        StepFeatureExtractor
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from backend.models.CNN.model_binary import (
        FineGrainedStepClassifier,
        LightweightFineGrainedClassifier,
        StepFeatureExtractor
    )

# === 数据集和数据加载功能 ===

class BinaryAccelerometerDataset(Dataset):
    """二分类加速度数据集"""

    def __init__(self, data, labels, sequence_length=100):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 确保数据长度为sequence_length
        sample = self.data[idx]
        if len(sample) > self.sequence_length:
            sample = sample[:self.sequence_length]
        elif len(sample) < self.sequence_length:
            # 如果数据不够长，进行填充
            padding = np.zeros((self.sequence_length - len(sample), sample.shape[1]))
            sample = np.vstack([sample, padding])

        return torch.FloatTensor(sample), torch.LongTensor([self.labels[idx]])

def load_binary_data(data_folder, sequence_length=100, use_feature_enhancement=True):
    """
    加载二分类数据（坐着/站着）

    参数:
    - data_folder: 数据文件夹路径，应包含sit和stand子文件夹
    - sequence_length: 序列长度
    - use_feature_enhancement: 是否使用特征增强

    返回:
    - features: 特征数据
    - labels: 标签数据 (0: sit, 1: stand)
    """

    sit_folder = os.path.join(data_folder, "sit")
    stand_folder = os.path.join(data_folder, "stand")

    if not os.path.exists(sit_folder) or not os.path.exists(stand_folder):
        raise ValueError(f"数据文件夹不存在: {sit_folder} 或 {stand_folder}")

    all_features = []
    all_labels = []

    # 初始化特征提取器
    if use_feature_enhancement:
        feature_extractor = StepFeatureExtractor()

    # 加载坐着的数据 (标签: 0)
    print("加载坐着数据...")
    sit_files = glob.glob(os.path.join(sit_folder, "*.csv"))
    for i, file_path in enumerate(sit_files):
        try:
            df = pd.read_csv(file_path)
            if len(df) < 50:  # 跳过太短的数据
                continue

            # 提取x, y, z列
            data = df[['x', 'y', 'z']].values

            # 确保数据长度
            if len(data) > sequence_length:
                data = data[:sequence_length]
            elif len(data) < sequence_length:
                # 填充数据
                padding = np.zeros((sequence_length - len(data), 3))
                data = np.vstack([data, padding])

            # 特征增强
            if use_feature_enhancement:
                enhanced_features = feature_extractor.extract_fine_grained_features(data)
                all_features.append(enhanced_features)
            else:
                all_features.append(data)

            all_labels.append(0)  # sit = 0

            if (i + 1) % 100 == 0:
                print(f"已加载 {i + 1} 个坐着数据文件")

        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            continue

    print(f"坐着数据加载完成，共 {len([l for l in all_labels if l == 0])} 个样本")

    # 加载站着的数据 (标签: 1)
    print("加载站着数据...")
    stand_files = glob.glob(os.path.join(stand_folder, "*.csv"))
    for i, file_path in enumerate(stand_files):
        try:
            df = pd.read_csv(file_path)
            if len(df) < 50:  # 跳过太短的数据
                continue

            # 提取x, y, z列
            data = df[['x', 'y', 'z']].values

            # 确保数据长度
            if len(data) > sequence_length:
                data = data[:sequence_length]
            elif len(data) < sequence_length:
                # 填充数据
                padding = np.zeros((sequence_length - len(data), 3))
                data = np.vstack([data, padding])

            # 特征增强
            if use_feature_enhancement:
                enhanced_features = feature_extractor.extract_fine_grained_features(data)
                all_features.append(enhanced_features)
            else:
                all_features.append(data)

            all_labels.append(1)  # stand = 1

            if (i + 1) % 100 == 0:
                print(f"已加载 {i + 1} 个站着数据文件")

        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            continue

    print(f"站着数据加载完成，共 {len([l for l in all_labels if l == 1])} 个样本")

    if len(all_features) == 0:
        print("错误: 未能加载任何数据")
        return None, None

    # 转换为numpy数组
    features = np.array(all_features)
    labels = np.array(all_labels)

    print(f"数据加载完成:")
    print(f"  特征形状: {features.shape}")
    print(f"  标签形状: {labels.shape}")
    print(f"  坐着样本数: {np.sum(labels == 0)}")
    print(f"  站着样本数: {np.sum(labels == 1)}")

    return features, labels

def create_binary_data_loaders(features, labels, batch_size=32, test_size=0.2, val_size=0.1):
    """
    创建二分类数据加载器

    参数:
    - features: 特征数据
    - labels: 标签数据
    - batch_size: 批次大小
    - test_size: 测试集比例
    - val_size: 验证集比例

    返回:
    - train_loader, val_loader, test_loader
    """

    # 分层采样确保标签分布平衡
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )

    print(f"数据集划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    # 创建数据集
    train_dataset = BinaryAccelerometerDataset(X_train, y_train)
    val_dataset = BinaryAccelerometerDataset(X_val, y_val)
    test_dataset = BinaryAccelerometerDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

# === 损失函数优化 ===

class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡和困难样本"""
    def __init__(self, alpha=1, gamma=2, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    """标签平滑 - 防止过拟合"""
    def __init__(self, num_classes=2, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()
        return loss

def train_binary_classifier_advanced(model, train_loader, val_loader, epochs=100, device=None,
                                   model_save_path='best_binary_model.pth', config=None, label2idx=None, idx2label=None):
    """高级二分类器训练 - 使用组合损失函数"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 获取类别数
    num_classes = config.get('num_classes', 2) if config else 2

    # 组合损失函数
    focal_loss = FocalLoss(alpha=1, gamma=2, num_classes=num_classes).to(device)
    smooth_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_val_acc = 0
    patience_counter = 0
    max_patience = 25

    print(f"\n🚀 开始高级训练... (设备: {device})")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze().long()
            optimizer.zero_grad()

            outputs = model(batch_x)

            loss1 = focal_loss(outputs, batch_y)
            loss2 = smooth_loss(outputs, batch_y)
            loss = 0.7 * loss1 + 0.3 * loss2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze().long()
                outputs = model(batch_x)
                loss = focal_loss(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # 保存完整的模型信息
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'label2idx': label2idx,
                'idx2label': idx2label,
                'best_val_acc': best_val_acc,
                'epoch': epoch
            }
            torch.save(checkpoint, model_save_path)
            print(f"  ✅ 模型已保存到 {model_save_path} (验证集准确率: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  ⏹️ 提前停止训练 (连续{max_patience}轮无改善)")
                break

    print(f"\n🎉 训练完成! 最佳验证集准确率: {best_val_acc:.2f}%")
    return best_val_acc

def train_binary_classifier_simple(model, train_loader, val_loader, epochs=50, device=None,
                                 model_save_path='best_binary_model.pth', config=None, label2idx=None, idx2label=None):
    """简单二分类器训练 - 使用标准交叉熵损失"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用简单的交叉熵损失
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_acc = 0
    patience_counter = 0
    max_patience = 15

    print(f"\n🚀 开始简单训练... (设备: {device})")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze().long()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze().long()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        # 计算准确率
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        scheduler.step()

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # 保存完整的模型信息
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config or {
                    'model_type': 'lightweight',
                    'num_features': model.features[0].in_channels if hasattr(model, 'features') else 24,
                    'num_classes': 2,
                    'sequence_length': 100,
                    'use_feature_enhancement': True
                },
                'label2idx': label2idx or {"sit": 0, "stand": 1},
                'idx2label': idx2label or {0: "sit", 1: "stand"},
                'best_val_acc': best_val_acc,
                'epoch': epoch
            }
            torch.save(checkpoint, model_save_path)
            print(f"  ✅ 模型已保存到 {model_save_path} (验证集准确率: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  ⏹️ 提前停止训练 (连续{max_patience}轮无改善)")
                break

    print(f"\n🎉 训练完成! 最佳验证集准确率: {best_val_acc:.2f}%")
    return best_val_acc

def main():
    """主函数 - 支持简单和高级两种训练模式"""
    # --- 配置 ---
    DATA_FOLDER = "G:/乌七八糟的东西/NUS-Summer-Workshop-AIOT/backend/models/CNN/splited"
    MODEL_TYPE = 'lightweight'  # 'lightweight' 或 'full'
    USE_FEATURE_ENHANCEMENT = True
    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 512  # 批次大小
    EPOCHS = 100
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    MODEL_SAVE_PATH = 'best_binary_model.pth'
    TRAINING_MODE = 'simple'  # 'simple' 或 'advanced'

    print("=== 二分类模型训练 (合并版) ===")
    print(f"数据文件夹: {DATA_FOLDER}")
    print(f"模型类型: {MODEL_TYPE}")
    print(f"训练模式: {TRAINING_MODE}")
    print(f"使用特征增强: {USE_FEATURE_ENHANCEMENT}")
    print(f"序列长度: {SEQUENCE_LENGTH}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")

    # 检查数据文件夹
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ 错误: 数据文件夹不存在: {DATA_FOLDER}")
        return

    sit_folder = os.path.join(DATA_FOLDER, "sit")
    stand_folder = os.path.join(DATA_FOLDER, "stand")

    if not os.path.exists(sit_folder) or not os.path.exists(stand_folder):
        print(f"❌ 错误: sit或stand文件夹不存在")
        print(f"sit文件夹: {sit_folder} (存在: {os.path.exists(sit_folder)})")
        print(f"stand文件夹: {stand_folder} (存在: {os.path.exists(stand_folder)})")
        return

    # --- 数据加载 ---
    print("\n📂 开始加载二分类数据...")
    try:
        features, labels = load_binary_data(DATA_FOLDER, SEQUENCE_LENGTH, USE_FEATURE_ENHANCEMENT)

        if features is None:
            print("❌ 错误: 未能加载任何数据")
            return

        print(f"✅ 数据加载完成: {features.shape}, 标签: {labels.shape}")
        print(f"坐着样本: {sum(labels == 0)}, 站着样本: {sum(labels == 1)}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # --- 创建数据加载器 ---
    print("\n🔄 创建数据加载器...")
    try:
        train_loader, val_loader, _ = create_binary_data_loaders(
            features, labels,
            batch_size=BATCH_SIZE,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE
        )
        print("✅ 数据加载器创建成功")

    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return

    # --- 模型选择和训练 ---
    print(f"\n🤖 创建{MODEL_TYPE}模型...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_features = features.shape[-1]
        num_classes = 2  # 二分类

        print(f"使用设备: {device}")
        print(f"特征维度: {num_features}")
        print(f"类别数: {num_classes}")

        if MODEL_TYPE == 'lightweight':
            model = LightweightFineGrainedClassifier(input_size=num_features, num_classes=num_classes)
            print("✅ 使用 LightweightFineGrainedClassifier 模型")
        else:
            model = FineGrainedStepClassifier(input_size=num_features, num_classes=num_classes)
            print("✅ 使用 FineGrainedStepClassifier 模型")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数数量: {total_params:,}")

    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return

    # 保存配置信息
    config = {
        'model_type': MODEL_TYPE,
        'num_features': num_features,
        'num_classes': num_classes,
        'sequence_length': SEQUENCE_LENGTH,
        'use_feature_enhancement': USE_FEATURE_ENHANCEMENT
    }

    # 标签映射
    label2idx = {"sit": 0, "stand": 1}
    idx2label = {0: "sit", 1: "stand"}

    # 选择训练函数
    print(f"\n🚀 开始{TRAINING_MODE}训练...")
    try:
        if TRAINING_MODE == 'advanced':
            best_acc = train_binary_classifier_advanced(
                model, train_loader, val_loader,
                epochs=EPOCHS, device=device,
                model_save_path=MODEL_SAVE_PATH,
                config=config,
                label2idx=label2idx,
                idx2label=idx2label
            )
        else:  # simple
            best_acc = train_binary_classifier_simple(
                model, train_loader, val_loader,
                epochs=EPOCHS, device=device,
                model_save_path=MODEL_SAVE_PATH,
                config=config,
                label2idx=label2idx,
                idx2label=idx2label
            )

        print(f"\n🎯 训练完成! 最佳准确率: {best_acc:.2f}%")
        print(f"📁 模型已保存到: {MODEL_SAVE_PATH}")

        # 提示下一步
        print(f"\n💡 下一步:")
        print(f"1. 运行测试脚本验证模型: python test_binary_prediction.py")
        print(f"2. 使用预测函数进行推理")

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

