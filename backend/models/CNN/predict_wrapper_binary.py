"""
二分类预测封装器 (Binary Classification Prediction Wrapper)

这个模块提供了用于二分类预测的封装函数，支持：
1. 模型加载和管理
2. 单个样本预测
3. CSV文件批量预测
4. 概率分布输出
5. 命令行接口

主要功能：
- load_binary_prediction_model(): 加载二分类模型
- predict_binary_classification(): 核心预测函数
- predict_from_csv_file(): 从CSV文件预测的便捷函数

使用示例：
    python predict_wrapper二分类.py data.csv

或作为模块导入：
    from predict_wrapper二分类 import predict_from_csv_file
    result = predict_from_csv_file('data.csv', 'model.pth', return_probabilities=True)
"""

import torch
import numpy as np
import os
import sys
import pandas as pd

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

# 从二分类模型文件中导入所需模块
try:
    from model_binary import (
        FineGrainedStepClassifier,
        LightweightFineGrainedClassifier,
        StepFeatureExtractor
    )
except ImportError:
    from backend.models.CNN.model_binary import (
        FineGrainedStepClassifier,
        LightweightFineGrainedClassifier,
        StepFeatureExtractor
    )

# 二分类标签映射
BINARY_CLASSES = {
    0: "sit",    # 坐着
    1: "stand"   # 站着
}

# 反向映射
LABEL_TO_IDX = {"sit": 0, "stand": 1}
IDX_TO_LABEL = {0: "sit", 1: "stand"}

# 全局变量
MODEL_REGISTRY = {
    'lightweight': LightweightFineGrainedClassifier,
    'full': FineGrainedStepClassifier
}

def load_binary_prediction_model(model_path, model_type='lightweight'):
    """
    加载用于二分类预测的模型。

    参数:
    - model_path: 模型文件路径 (.pth)
    - model_type: 'lightweight' 或 'full'

    返回:
    - model: 加载好的PyTorch模型
    - extractor: 特征提取器实例
    - device: 运行设备
    - config: 模型配置信息
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"无效的模型类型: {model_type}. 可选项: {list(MODEL_REGISTRY.keys())}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=device)

        # 获取模型配置
        config = checkpoint.get('config', {})
        saved_model_type = config.get('model_type', model_type)
        num_classes = config.get('num_classes', 2)

        # 初始化特征提取器以确定输入维度
        extractor = StepFeatureExtractor()
        dummy_data = np.zeros((100, 3))
        num_features = extractor.extract_fine_grained_features(dummy_data).shape[1]

        # 创建模型实例
        model_class = MODEL_REGISTRY[saved_model_type]
        model = model_class(input_size=num_features, num_classes=num_classes)

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"二分类模型已加载: {model_path} (类型: {saved_model_type}, 设备: {device})")
        return model, extractor, device, config

    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

def predict_binary_classification(data, model, extractor, device, return_probabilities=False):
    """
    使用加载好的模型和提取器进行二分类预测（坐着/站着）。

    参数:
    - data: 形状为 (100, 3) 的Numpy数组。
    - model: 已加载的PyTorch模型。
    - extractor: StepFeatureExtractor实例。
    - device: PyTorch设备。
    - return_probabilities: 是否返回概率分布

    返回:
    - 如果 return_probabilities=False: 预测的标签 ("sit" 或 "stand")
    - 如果 return_probabilities=True: 包含预测结果、置信度和概率的字典
    """
    # 1. 验证数据形状
    if data.shape != (100, 3):
        raise ValueError(f"输入数据形状错误，应为 (100, 3)，实际为 {data.shape}")

    # 2. 特征工程
    enhanced_features = extractor.extract_fine_grained_features(data)

    # 3. 转换为Tensor并添加batch维度
    sequence_tensor = torch.FloatTensor(enhanced_features).unsqueeze(0).to(device)

    # 4. 模型预测
    with torch.no_grad():
        output = model(sequence_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class_id = torch.max(probabilities, 1)

        prediction_idx = predicted_class_id.item()
        confidence_score = confidence.item()
        prediction_label = IDX_TO_LABEL[prediction_idx]

        if return_probabilities:
            return {
                'prediction': prediction_label,
                'confidence': confidence_score,
                'probabilities': {
                    'sit': probabilities[0][0].item(),
                    'stand': probabilities[0][1].item()
                }
            }
        else:
            return prediction_label

def predict_from_csv_file(csv_path, model_path, model_type='lightweight', return_probabilities=False):
    """
    从CSV文件预测二分类结果的便捷函数

    参数:
    - csv_path: CSV文件路径
    - model_path: 模型文件路径
    - model_type: 模型类型
    - return_probabilities: 是否返回概率分布

    返回:
    - 预测结果
    """
    try:
        # 加载模型
        model, extractor, device, config = load_binary_prediction_model(model_path, model_type)

        # 读取CSV文件
        df = pd.read_csv(csv_path)
        if not {'x', 'y', 'z'}.issubset(df.columns):
            raise ValueError("CSV文件必须包含x, y, z列")

        # 准备数据
        data = df[['x', 'y', 'z']].values
        if len(data) > 100:
            data = data[:100]
        elif len(data) < 100:
            padding = np.zeros((100 - len(data), 3))
            data = np.vstack([data, padding])

        # 预测
        result = predict_binary_classification(data, model, extractor, device, return_probabilities)
        return result

    except Exception as e:
        print(f"从CSV文件预测失败: {e}")
        return None

# --- 使用示例 ---
if __name__ == "__main__":
    # 配置
    DEFAULT_MODEL_PATH = os.path.join(current_dir, 'best_binary_model.pth')
    DEFAULT_MODEL_TYPE = 'lightweight'  # 与训练脚本保持一致

    # 检查模型文件是否存在
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"错误: 找不到默认模型文件 '{DEFAULT_MODEL_PATH}'")
        print("请先运行 train二分类.py 训练模型，或通过命令行参数指定模型路径。")
        sys.exit(1)

    try:
        # 加载模型 (一次性)
        model, extractor, device, config = load_binary_prediction_model(DEFAULT_MODEL_PATH, DEFAULT_MODEL_TYPE)
        print(f"✅ 模型加载成功: {config}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)

    # 模式1: 从命令行读取CSV文件进行二分类预测
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"\n--- 从文件 {csv_file} 进行二分类预测 ---")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if all(col in df.columns for col in ['x', 'y', 'z']):
                    if len(df) >= 100:
                        data = df.head(100)[['x', 'y', 'z']].values

                        # 简单预测
                        result_simple = predict_binary_classification(data, model, extractor, device)
                        print(f"✅ 预测结果: {result_simple}")

                        # 详细预测（包含概率）
                        result_detailed = predict_binary_classification(data, model, extractor, device, return_probabilities=True)
                        print(f"✅ 详细结果: {result_detailed}")

                    else:
                        print(f"❌ 错误: 文件 '{csv_file}' 的数据点少于100, 无法预测。")
                else:
                    print("❌ 错误: CSV文件需要包含 'x', 'y', 'z' 列。")
            except Exception as e:
                print(f"处理CSV文件时出错: {e}")
        else:
            print(f"❌ 错误: 文件 '{csv_file}' 不存在。")

    # 模式2: 使用随机数据进行测试
    else:
        print("\n--- 使用随机数据进行二分类测试 ---")
        try:
            # 生成模拟坐着数据 (较小运动幅度)
            sit_data = np.random.normal(0, 0.1, (100, 3))
            sit_data[:, 2] -= 1.0  # z轴偏移模拟重力

            result_sit = predict_binary_classification(sit_data, model, extractor, device, return_probabilities=True)
            print(f"✅ 模拟坐着数据预测结果: {result_sit}")

            # 生成模拟站着数据 (较大运动幅度)
            stand_data = np.random.normal(0, 0.3, (100, 3))
            stand_data[:, 2] -= 0.5  # z轴偏移

            result_stand = predict_binary_classification(stand_data, model, extractor, device, return_probabilities=True)
            print(f"✅ 模拟站着数据预测结果: {result_stand}")

        except Exception as e:
            print(f"❌ 预测失败: {e}")

        print("\n" + "="*50)
        print("🎯 二分类预测封装器使用方法:")
        print("1. 作为模块导入:")
        print("   from backend.models.CNN.predict_wrapper二分类 import load_binary_prediction_model, predict_binary_classification")
        print("   model, extractor, device, config = load_binary_prediction_model('best_binary_model.pth')")
        print("   result = predict_binary_classification(data, model, extractor, device, return_probabilities=True)")
        print("\n2. 便捷函数:")
        print("   from backend.models.CNN.predict_wrapper二分类 import predict_from_csv_file")
        print("   result = predict_from_csv_file('data.csv', 'best_binary_model.pth', return_probabilities=True)")
        print("\n3. 作为脚本执行:")
        print(f"   python predict_wrapper二分类.py path/to/data.csv")
        print("\n4. 与合并的预测测试系统配合使用:")
        print("   python predict_and_test.py --mode predict --input data.csv")
        print("="*50)