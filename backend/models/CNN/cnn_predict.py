import torch
import numpy as np
import os
import sys

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from models.CNN.model import create_model
from models.CNN.data_utils import DataPreprocessor

def predict(data, model_path):
    """
    预测函数：输入三轴加速度数据，输出步数预测结果
    
    参数:
    - data: 形状为(3, 100)的numpy数组，表示三轴加速度数据(x,y,z轴 x 100个时间点)
    - model_path: 模型文件路径
    
    返回:
    - 预测的步数或类别
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 调整输入数据形状为(100, 3)
    if data.shape[0] == 3 and data.shape[1] == 100:
        data = data.transpose()
    elif data.shape[0] != 100 or data.shape[1] != 3:
        raise ValueError(f"输入数据形状错误，应为(3, 100)或(100, 3)，实际为{data.shape}")
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_data(data, normalize=True)
    processed_data = np.expand_dims(processed_data, 0)  # 添加batch维度
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型参数
    config = checkpoint.get('config', {})
    label2idx = checkpoint.get('label2idx', None)
    idx2label = checkpoint.get('idx2label', None)
    num_classes = len(label2idx) if label2idx else 8
    model_type = config.get('model_type', 'efficient')
    
    # 创建并加载模型
    model = create_model(model_type=model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 预测
    with torch.no_grad():
        tensor_data = torch.FloatTensor(processed_data).to(device)
        output = model(tensor_data)
        
    # 处理输出
    if output.shape[-1] > 1:  # 分类模型
        _, predicted_class = torch.max(output, 1)
        prediction = predicted_class.item()
        # 如果有标签映射，转换为对应标签
        if idx2label is not None and prediction in idx2label:
            prediction = idx2label[prediction]
    else:  # 回归模型
        prediction = output.item()

    print(f"预测结果: {prediction}")
    
    return prediction 