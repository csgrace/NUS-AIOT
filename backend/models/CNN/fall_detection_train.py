import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from fall_detection_model import (
    FallDetectionClassifier, 
    LightweightFallDetector, 
    RealTimeFallDetector,
    FallFeatureExtractor
)
from weda_fall_data_loader import WEDAFallDataLoader

def numpy_to_python_type(obj):
    """递归地将NumPy类型转换为Python原生类型，用于JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [numpy_to_python_type(item) for item in obj]
    else:
        return obj

class FallDetectionTrainer:
    """摔倒检测训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化特征提取器
        self.feature_extractor = FallFeatureExtractor()
        
    def load_weda_fall_data(self):
        """加载WEDA-FALL数据集"""
        # 检查是否已有处理好的数据
        if os.path.exists('weda_fall_data.npy') and os.path.exists('weda_fall_labels.npy'):
            print("发现已处理的数据文件，直接加载...")
            data = np.load('weda_fall_data.npy')
            labels = np.load('weda_fall_labels.npy')
        else:
            print("首次加载，处理原始数据...")
            loader = WEDAFallDataLoader(
                self.config['data_dir'],
                fall_dir=self.config.get('fall_dir'),
                no_fall_dir=self.config.get('no_fall_dir')
            )
            data, labels = loader.load_weda_fall_dataset()
            
            # 保存处理后的数据
            np.save('weda_fall_data.npy', data)
            np.save('weda_fall_labels.npy', labels)
            print("数据已保存为 weda_fall_data.npy 和 weda_fall_labels.npy")
        
        return data, labels
        
    def create_model(self, model_type='standard'):
        """创建模型"""
        input_size = self.config.get('input_size', 20)
        sequence_length = self.config.get('sequence_length', 100)
        
        print(f"创建{model_type}模型，输入维度: {input_size}")
        
        if model_type == 'standard':
            model = FallDetectionClassifier(input_size, sequence_length, num_classes=2)
        elif model_type == 'lightweight':
            model = LightweightFallDetector(input_size, sequence_length, num_classes=2)
        elif model_type == 'realtime':
            model = RealTimeFallDetector(input_size, sequence_length)
        else:
            raise ValueError(f"未知模型类型: {model_type}")
            
        return model.to(self.device)
    
    def prepare_data(self, data, labels):
        """准备训练数据"""
        print("提取摔倒检测特征...")
        
        # 特征提取
        enhanced_data = []
        for i in tqdm(range(len(data)), desc="特征提取"):
            features = self.feature_extractor.extract_fall_features(data[i])
            enhanced_data.append(features)
        
        enhanced_data = np.array(enhanced_data)
        print(f"增强后数据形状: {enhanced_data.shape}")
        
        # 更新配置中的input_size
        self.config['input_size'] = enhanced_data.shape[-1]
        print(f"特征维度: {self.config['input_size']}")
        
        # 数据标准化
        scaler = StandardScaler()
        original_shape = enhanced_data.shape
        enhanced_data_flat = enhanced_data.reshape(-1, enhanced_data.shape[-1])
        enhanced_data_scaled = scaler.fit_transform(enhanced_data_flat)
        enhanced_data = enhanced_data_scaled.reshape(original_shape)
        
        # 转换为张量
        X = torch.FloatTensor(enhanced_data)
        y = torch.LongTensor(labels.astype(np.int64))
        
        # 数据集分割
        dataset = TensorDataset(X, y)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        print(f"数据分割: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, scaler
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, model, val_loader, criterion):
        """验证模型"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        return total_loss / len(val_loader), accuracy, precision, recall, f1
    
    def train(self, train_loader, val_loader, model_type='standard'):
        """完整训练流程"""
        # 创建模型
        model = self.create_model(model_type)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 训练历史
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        best_f1 = 0
        patience_counter = 0
        
        print(f"开始训练 {model_type} 模型...")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate(
                model, val_loader, criterion
            )
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
            
            # 早停检查
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), f'best_fall_detection_{model_type}.pth')
                print(f"保存最佳模型 (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"早停触发 (patience: {self.config['patience']})")
                    break
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f'best_fall_detection_{model_type}.pth'))
        
        return model, history
    
    def evaluate(self, model, test_loader):
        """评估模型"""
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        
        print(f"\n测试结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"\n混淆矩阵:")
        print("       预测")
        print("实际   正常  摔倒")
        print(f"正常   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"摔倒   {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'probabilities': all_probs
        }

def main():
    # 配置参数
    config = {
        'data_dir': "G:\\乌七八糟的东西\\WEDA-FALL-main",
        'no_fall_dir': "G:\\乌七八糟的东西\\WEDA-FALL-main\\no",
        'fall_dir': "G:\\乌七八糟的东西\\WEDA-FALL-main\\slip",
        'sequence_length': 100,
        'input_size': 20,  # 将根据特征提取器自动调整
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'patience': 10
    }
    
    print("WEDA-FALL摔倒检测训练系统")
    print("=" * 50)
    
    # 创建训练器
    trainer = FallDetectionTrainer(config)
    
    try:
        # 加载WEDA-FALL数据
        data, labels = trainer.load_weda_fall_data()
        
        print(f"\n数据加载完成:")
        print(f"数据形状: {data.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签分布: 正常={np.sum(labels==0)}, 摔倒={np.sum(labels==1)}")
        
        # 准备数据
        train_loader, val_loader, test_loader, scaler = trainer.prepare_data(data, labels)
        
        # 训练不同类型的模型
        models = {}
        for model_type in ['lightweight', 'standard']:
            print(f"\n{'='*50}")
            print(f"训练 {model_type} 模型")
            print(f"{'='*50}")
            
            model, history = trainer.train(train_loader, val_loader, model_type)
            models[model_type] = model
            
            # 评估模型
            results = trainer.evaluate(model, test_loader)
            
            # 保存结果
            with open(f'weda_fall_{model_type}_results.json', 'w') as f:
                json.dump({
                    'config': numpy_to_python_type(config),
                    'results': {k: numpy_to_python_type(v) for k, v in results.items() if k != 'confusion_matrix'}
                }, f, indent=2)
            
            # 单独保存混淆矩阵
            np.save(f'weda_fall_{model_type}_confusion_matrix.npy', results['confusion_matrix'])
        
        print("\n训练完成！")
        print("模型文件已保存为: best_fall_detection_*.pth")
        print("结果文件已保存为: weda_fall_*_results.json")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()





