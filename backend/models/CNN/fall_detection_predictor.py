import torch
import numpy as np
import pandas as pd
import os
import json
from fall_detection_model import (
    FallDetectionClassifier, 
    LightweightFallDetector, 
    RealTimeFallDetector,
    FallFeatureExtractor
)

class FallDetectionPredictor:
    """摔倒检测预测器"""
    
    def __init__(self, model_path=None, model_type='lightweight', device=None):
        """
        初始化摔倒检测预测器
        
        参数:
            model_path: 模型文件路径，如果为None则使用默认路径
            model_type: 模型类型，可选 'lightweight', 'standard', 'realtime'
            device: 计算设备，如果为None则自动选择
        """
        self.model_type = model_type
        
        if model_path is None:
            self.model_path = f'best_fall_detection_{model_type}.pth'
        else:
            self.model_path = model_path
            
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        print(f"加载模型: {self.model_path}")
        
        # 加载特征提取器
        self.feature_extractor = FallFeatureExtractor()
        
        # 创建模型并加载权重
        self.model = self._create_model()
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        # 类别映射
        self.class_names = ["正常", "摔倒"]
    
    def _create_model(self):
        """创建模型并加载权重"""
        input_size = 16  # 特征提取后的维度
        sequence_length = 100
        
        if self.model_type == 'lightweight':
            model = LightweightFallDetector(input_size, sequence_length, num_classes=2)
        elif self.model_type == 'standard':
            model = FallDetectionClassifier(input_size, sequence_length, num_classes=2)
        elif self.model_type == 'realtime':
            model = RealTimeFallDetector(input_size, sequence_length)
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")
        
        # 加载模型权重
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        return model
    
    def load_csv_data(self, csv_file):
        """加载CSV文件数据"""
        try:
            df = pd.read_csv(csv_file)
            
            # 检查数据格式
            if df.shape[1] >= 3:
                # 假设前3列是加速度数据 (x, y, z)
                acc_data = df.iloc[:, :3].values
                
                # 确保数据长度足够
                if len(acc_data) < 100:
                    print(f"警告: 数据长度不足100点，将进行填充")
                    # 填充到100个点
                    padding = np.tile(acc_data[-1], (100 - len(acc_data), 1))
                    acc_data = np.vstack([acc_data, padding])
                
                # 如果数据过长，只取前100个点
                if len(acc_data) > 100:
                    acc_data = acc_data[:100]
                
                return acc_data
            else:
                raise ValueError(f"CSV文件列数不足3列: {csv_file}")
                
        except Exception as e:
            raise RuntimeError(f"读取CSV文件时出错: {e}")
    
    def preprocess_data(self, acc_data):
        """预处理数据"""
        # 确保数据是正确的形状 [seq_len, 3]
        if acc_data.shape != (100, 3):
            raise ValueError(f"数据形状不正确: {acc_data.shape}，期望形状: (100, 3)")
        
        # 提取特征
        features = self.feature_extractor.extract_fall_features(acc_data)
        
        # 简单标准化
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-6  # 添加小值避免除零
        features = (features - mean) / std
        
        # 转换为张量
        features_tensor = torch.FloatTensor(features).unsqueeze(0)  # 添加批次维度
        
        return features_tensor
    
    def predict_array(self, acc_data):
        """
        直接从NumPy数组预测摔倒
        
        参数:
            acc_data: 形状为(N, 3)的NumPy数组，包含加速度数据
            
        返回:
            预测结果字典
        """
        # 确保数据形状符合要求
        if len(acc_data.shape) != 2 or acc_data.shape[1] != 3:
            raise ValueError(f"输入数据形状错误，应为(N, 3)，实际为{acc_data.shape}")
            
        # 处理数据长度
        if len(acc_data) < 100:
            print(f"警告: 数据长度不足100点，将进行填充")
            # 填充到100个点
            padding = np.tile(acc_data[-1], (100 - len(acc_data), 1))
            acc_data = np.vstack([acc_data, padding])
        
        # 如果数据过长，只取前100个点
        if len(acc_data) > 100:
            acc_data = acc_data[:100]
            
        # 预处理数据
        features = self.preprocess_data(acc_data)
        
        # 推理
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        result = {
            'class_id': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }
        
        return result
    
    def predict(self, csv_file_or_array):
        """
        从CSV文件或NumPy数组预测摔倒
        
        参数:
            csv_file_or_array: CSV文件路径或形状为(N, 3)的NumPy数组
            
        返回:
            预测结果字典
        """
        # 根据输入类型选择不同的处理方法
        if isinstance(csv_file_or_array, str):
            # 输入是文件路径
            acc_data = self.load_csv_data(csv_file_or_array)
        elif isinstance(csv_file_or_array, np.ndarray):
            # 输入是NumPy数组
            acc_data = csv_file_or_array
        else:
            raise TypeError("输入类型必须是字符串(CSV文件路径)或NumPy数组")
            
        # 使用统一的预测函数
        return self.predict_array(acc_data)
    
    def predict_batch(self, csv_folder):
        """批量预测文件夹中的CSV文件"""
        results = {}
        
        for file_name in os.listdir(csv_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(csv_folder, file_name)
                try:
                    result = self.predict(file_path)
                    results[file_name] = result
                    print(f"{file_name}: {result['class_name']} (置信度: {result['confidence']:.4f})")
                except Exception as e:
                    print(f"处理文件 {file_name} 时出错: {e}")
        
        return results

# 创建一个便利函数，用于从外部直接调用
def load_fall_detection_model(model_path=None, model_type='lightweight'):
    """
    加载摔倒检测模型
    
    参数:
        model_path: 模型文件路径，如果为None则使用默认路径
        model_type: 模型类型，可选 'lightweight', 'standard', 'realtime'
        
    返回:
        加载好的FallDetectionPredictor实例
    """
    predictor = FallDetectionPredictor(model_path, model_type)
    return predictor

def predict_fall(acc_data, predictor=None):
    """
    使用摔倒检测模型预测
    
    参数:
        acc_data: 形状为(N, 3)的NumPy数组，包含加速度数据
        predictor: FallDetectionPredictor实例，如果为None则创建一个新实例
        
    返回:
        预测结果字典
    """
    if predictor is None:
        predictor = load_fall_detection_model()
        
    return predictor.predict(acc_data)

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="摔倒检测预测器")
    parser.add_argument('--model', default='lightweight', choices=['lightweight', 'standard', 'realtime'],
                        help="选择模型类型: lightweight, standard, realtime")
    parser.add_argument('--file', help="要预测的CSV文件路径")
    parser.add_argument('--folder', help="要批量预测的CSV文件夹路径")
    
    args = parser.parse_args()
    
    predictor = FallDetectionPredictor(model_type=args.model)
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"文件不存在: {args.file}")
            return
            
        try:
            result = predictor.predict(args.file)
            print("\n预测结果:")
            print(f"文件: {args.file}")
            print(f"类别: {result['class_name']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"各类别概率: 正常={result['probabilities'][0]:.4f}, 摔倒={result['probabilities'][1]:.4f}")
        except Exception as e:
            print(f"预测过程中出错: {e}")
            
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"文件夹不存在: {args.folder}")
            return
            
        try:
            results = predictor.predict_batch(args.folder)
            print(f"\n共处理 {len(results)} 个文件")
            
            # 统计结果
            fall_count = sum(1 for r in results.values() if r['class_id'] == 1)
            normal_count = len(results) - fall_count
            print(f"检测到 {normal_count} 个正常样本, {fall_count} 个摔倒样本")
            
        except Exception as e:
            print(f"批量预测过程中出错: {e}")
    else:
        # 如果没有提供参数，显示使用示例
        print("\n--- 使用示例 ---")
        # 生成随机测试数据
        test_data = np.random.randn(100, 3)
        print("使用随机数据进行测试...")
        try:
            result = predictor.predict(test_data)
            print(f"预测结果: {result['class_name']} (置信度: {result['confidence']:.4f})")
        except Exception as e:
            print(f"预测过程中出错: {e}")
            
        print("\n使用方法:")
        print("1. 从CSV文件预测:")
        print("   python fall_detection_predictor.py --file path/to/data.csv")
        print("\n2. 批量预测文件夹中的CSV文件:")
        print("   python fall_detection_predictor.py --folder path/to/csv_folder")
        print("\n3. 作为模块导入使用数组预测:")
        print("   from fall_detection_predictor import load_fall_detection_model, predict_fall")
        print("   predictor = load_fall_detection_model()")
        print("   result = predictor.predict(my_accelerometer_data)")  # 数组形状: (N, 3)

if __name__ == "__main__":
    main() 


"""
from backend.models.CNN.fall_detection_predictor import predict_fall

# 假设有一个形状为(N,3)的加速度数据
acc_data = np.array([...])  # 加速度数据，形状(N,3)
result = predict_fall(acc_data)
print(f"预测结果: {result['class_name']}, 置信度: {result['confidence']}")
"""