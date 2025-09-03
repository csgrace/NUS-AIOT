import torch
import numpy as np
import os
import sys
import pandas as pd

# # 添加项目根目录到系统路径
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
# sys.path.append(project_root)

# 从新模型文件中导入所需模块
from models.CNN.model import (
    FineGrainedStepClassifier, 
    LightweightFineGrainedClassifier,
    StepFeatureExtractor,
    STEP_CLASSES
)

# 全局变量
MODEL_REGISTRY = {
    'lightweight': LightweightFineGrainedClassifier,
    'full': FineGrainedStepClassifier
}

def load_prediction_model(model_path, model_type='lightweight'):
    """
    加载用于预测的细粒度分类模型。

    参数:
    - model_path: 模型文件路径 (.pth)
    - model_type: 'lightweight' 或 'full'

    返回:
    - model: 加载好的PyTorch模型
    - extractor: 特征提取器实例
    - device: 运行设备
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"无效的模型类型: {model_type}. 可选项: {list(MODEL_REGISTRY.keys())}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化特征提取器以确定输入维度
    extractor = StepFeatureExtractor()
    dummy_data = np.zeros((100, 3))
    num_features = extractor.extract_fine_grained_features(dummy_data).shape[1]
    
    # 创建模型实例
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(input_size=num_features)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"模型已加载: {model_path} (类型: {model_type}, 设备: {device})")
    return model, extractor, device

def predict_steps(data, model, extractor, device):
    """
    使用加载好的模型和提取器预测步数。

    参数:
    - data: 形状为 (100, 3) 的Numpy数组。
    - model: 已加载的PyTorch模型。
    - extractor: StepFeatureExtractor实例。
    - device: PyTorch设备。

    返回:
    - 预测的步数值 (例如 0, 1, 1.5, ...)。
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
        _, predicted_class_id = torch.max(output, 1)
        predicted_steps = STEP_CLASSES[predicted_class_id.item()]
        
    return predicted_steps

# # --- 使用示例 ---
# if __name__ == "__main__":
#     # 配置
#     DEFAULT_MODEL_PATH = os.path.join(current_dir, 'best_fine_grained_model.pth')
#     DEFAULT_MODEL_TYPE = 'lightweight' # 与训练脚本保持一致

#     # 检查模型文件是否存在
#     if not os.path.exists(DEFAULT_MODEL_PATH):
#         print(f"错误: 找不到默认模型文件 '{DEFAULT_MODEL_PATH}'")
#         print("请先运行 train.py 训练模型，或通过命令行参数指定模型路径。")
#         sys.exit(1)
        
#     try:
#         # 加载模型 (一次性)
#         model, extractor, device = load_prediction_model(DEFAULT_MODEL_PATH, DEFAULT_MODEL_TYPE)
#     except Exception as e:
#         print(f"加载模型失败: {e}")
#         sys.exit(1)

#     # 模式1: 从命令行读取CSV文件进行预测
#     if len(sys.argv) > 1:
#         csv_file = sys.argv[1]
#         print(f"\n--- 从文件 {csv_file} 预测 ---")
#         if os.path.exists(csv_file):
#             try:
#                 df = pd.read_csv(csv_file)
#                 if all(col in df.columns for col in ['x', 'y', 'z']):
#                     if len(df) >= 100:
#                         data = df.head(100)[['x', 'y', 'z']].values
#                         result = predict_steps(data, model, extractor, device)
#                         print(f"✅ 预测步数结果: {result}")
#                     else:
#                         print(f"❌ 错误: 文件 '{csv_file}' 的数据点少于100, 无法预测。")
#                 else:
#                     print("❌ 错误: CSV文件需要包含 'x', 'y', 'z' 列。")
#             except Exception as e:
#                 print(f"处理CSV文件时出错: {e}")
#         else:
#             print(f"❌ 错误: 文件 '{csv_file}' 不存在。")
    
#     # 模式2: 使用随机数据进行测试
#     else:
#         print("\n--- 使用随机数据进行测试 ---")
#         try:
#             # 生成随机测试数据 (100, 3)
#             test_data = np.random.randn(100, 3)
#             result = predict_steps(test_data, model, extractor, device)
#             print(f"✅ 随机数据测试预测结果: {result}")
#         except Exception as e:
#             print(f"❌ 预测失败: {e}")
            
#         print("\n" + "="*30)
#         print("使用方法:")
#         print("1. 作为模块导入:")
#         print("   from backend.models.CNN.predict_wrapper import load_prediction_model, predict_steps")
#         print("   model, extractor, device = load_prediction_model('path/to/model.pth')")
#         print("   result = predict_steps(data, model, extractor, device)")
#         print("\n2. 作为脚本执行:")
#         print(f"   python predict_wrapper.py path/to/data.csv")
#         print("="*30) 