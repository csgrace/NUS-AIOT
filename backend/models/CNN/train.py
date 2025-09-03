import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import sys
import torch.nn.functional as F

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

from backend.models.CNN.model import (
    FineGrainedStepClassifier, 
    LightweightFineGrainedClassifier, 
    StepFeatureExtractor,
    steps_to_class_id
)

# 4. 损失函数优化
class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡和困难样本"""
    def __init__(self, alpha=1, gamma=2, num_classes=8):
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
    def __init__(self, num_classes=8, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()
        return loss

def load_and_preprocess_data(data_folder, sequence_length=100, use_feature_enhancement=False):
    """加载和预处理数据"""
    all_sequences = []
    all_labels = []
    extractor = StepFeatureExtractor()

    print("开始加载和预处理数据...")
    for subfolder_path in sorted(glob.glob(os.path.join(data_folder, "*"))):
        if os.path.isdir(subfolder_path):
            try:
                folder_name = os.path.basename(subfolder_path)
                steps = float(folder_name) if '.' in folder_name else int(folder_name)
                label = steps_to_class_id(steps)
            except ValueError:
                print(f"跳过无法解析步数的文件夹: {folder_name}")
                continue

            for csv_file in glob.glob(os.path.join(subfolder_path, "*.csv")):
                df = pd.read_csv(csv_file)
                if {'x', 'y', 'z'}.issubset(df.columns):
                    sequence = df[['x', 'y', 'z']].head(sequence_length).values
                    if len(sequence) == sequence_length:
                        if use_feature_enhancement:
                            sequence = extractor.extract_fine_grained_features(sequence)
                        all_sequences.append(sequence)
                        all_labels.append(label)
    
    if not all_sequences:
        return None, None

    print(f"数据加载完成. 总样本数: {len(all_sequences)}")
    return np.array(all_sequences, dtype=np.float32), np.array(all_labels, dtype=np.int64)

# 5. 训练策略
def train_fine_grained_classifier(model, train_loader, val_loader, epochs=150, device=None, model_save_path='best_fine_grained_model.pth'):
    """细粒度分类器训练"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 组合损失函数
    focal_loss = FocalLoss(alpha=1, gamma=2, num_classes=8).to(device)
    smooth_loss = LabelSmoothingLoss(num_classes=8, smoothing=0.1).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_acc = 0
    patience_counter = 0
    
    print("\n开始训练...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
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
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
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
            torch.save(model.state_dict(), model_save_path)
            print(f"  模型已保存到 {model_save_path} (验证集准确率: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= 25:
                print("提前停止训练 (Early stopping triggered)")
                break
    
    print(f"\n训练完成. 最佳验证集准确率: {best_val_acc:.2f}%")
    return best_val_acc

def main():
    # --- 配置 ---
    DATA_FOLDER = "G:/乌七八糟的东西/步频数据/splited"
    MODEL_TYPE = 'lightweight'  # 'lightweight' 或 'full'
    USE_FEATURE_ENHANCEMENT = True
    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 512
    EPOCHS = 150
    VAL_SPLIT = 0.2
    MODEL_SAVE_PATH = 'best_fine_grained_model.pth'
    
    # --- 数据加载 ---
    features, labels = load_and_preprocess_data(DATA_FOLDER, SEQUENCE_LENGTH, USE_FEATURE_ENHANCEMENT)
    
    if features is None:
        print("错误: 未能加载任何数据，请检查数据文件夹和内容。")
        return
        
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # --- 模型选择和训练 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    num_features = features.shape[-1]
    print(f"特征维度: {num_features}")

    if MODEL_TYPE == 'lightweight':
        model = LightweightFineGrainedClassifier(input_size=num_features)
        print("使用 LightweightFineGrainedClassifier 模型")
    else:
        model = FineGrainedStepClassifier(input_size=num_features)
        print("使用 FineGrainedStepClassifier 模型")
        
    train_fine_grained_classifier(model, train_loader, val_loader, epochs=EPOCHS, device=device, model_save_path=MODEL_SAVE_PATH)

if __name__ == '__main__':
    main()

