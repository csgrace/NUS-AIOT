import os
import torch
import numpy as np
import pandas as pd
import glob
import sys
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

from backend.models.CNN.model import (
    FineGrainedStepClassifier, 
    LightweightFineGrainedClassifier,
    StepFeatureExtractor,
    steps_to_class_id,
    STEP_CLASSES,
    CLASS_NAMES
)

def load_model(model_path, model_type='lightweight'):
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 特征提取器决定了模型的输入维度
    extractor = StepFeatureExtractor()
    dummy_data = np.zeros((100, 3))
    num_features = extractor.extract_fine_grained_features(dummy_data).shape[1]

    if model_type == 'lightweight':
        model = LightweightFineGrainedClassifier(input_size=num_features)
    else: # full
        model = FineGrainedStepClassifier(input_size=num_features)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"模型已加载 (类型: {model_type}, 输入维度: {num_features}, 设备: {device})")
    return model, extractor

def predict_single_file(model, extractor, file_path, sequence_length=100, device=None):
    """预测单个文件"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        df = pd.read_csv(file_path)
        if {'x', 'y', 'z'}.issubset(df.columns):
            cols = ['x', 'y', 'z']
        else:
            return None, "缺少x, y, z列"
        
        if len(df) < sequence_length:
            return None, f"数据点不足 ({len(df)} < {sequence_length})"
            
        sequence = df[cols].head(sequence_length).values
        
        # 特征提取
        enhanced_sequence = extractor.extract_fine_grained_features(sequence)
        
        sequence_tensor = torch.FloatTensor(enhanced_sequence).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(sequence_tensor)
            _, predicted_class_id = torch.max(output, 1)
            predicted_steps = STEP_CLASSES[predicted_class_id.item()]
            
        return predicted_steps, None
    except Exception as e:
        return None, str(e)

def detailed_evaluation(model, extractor, test_folder, sequence_length=100, device=None):
    """详细评估细粒度分类器"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_predictions = []
    all_targets = []
    
    print(f"开始评估文件夹: {test_folder}")
    for subfolder_path in sorted(glob.glob(os.path.join(test_folder, "*"))):
        if os.path.isdir(subfolder_path):
            try:
                folder_name = os.path.basename(subfolder_path)
                steps = float(folder_name) if '.' in folder_name else int(folder_name)
                target_id = steps_to_class_id(steps)
            except ValueError:
                continue
            
            for csv_file in glob.glob(os.path.join(subfolder_path, "*.csv")):
                _, err, pred_id = predict_for_eval(model, extractor, csv_file, sequence_length, device)
                if err is None:
                    all_predictions.append(pred_id)
                    all_targets.append(target_id)
    
    if not all_targets:
        print("未找到有效数据进行评估")
        return

    # 分类报告
    print("\n详细分类报告:")
    print(classification_report(all_targets, all_predictions, 
                              target_names=CLASS_NAMES, digits=4))
    
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix_fine_grained.png')
    plt.show()
    
    # 计算邻近准确率 (±1类别的容错)
    adjacent_correct = sum(1 for true, pred in zip(all_targets, all_predictions) if abs(true - pred) <= 1)
    
    adjacent_accuracy = adjacent_correct / len(all_targets)
    exact_accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    
    print(f"\n精确准确率: {exact_accuracy:.4f}")
    print(f"邻近准确率 (±1 class): {adjacent_accuracy:.4f}")

def predict_for_eval(model, extractor, file_path, sequence_length, device):
    """专为评估使用的预测函数，返回类别ID"""
    try:
        df = pd.read_csv(file_path)
        sequence = df[['x', 'y', 'z']].head(sequence_length).values
        if len(sequence) < sequence_length: return None, "Data short", None
        
        enhanced_sequence = extractor.extract_fine_grained_features(sequence)
        sequence_tensor = torch.FloatTensor(enhanced_sequence).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(sequence_tensor)
            _, predicted_class_id = torch.max(output, 1)
        return STEP_CLASSES[predicted_class_id.item()], None, predicted_class_id.item()
    except Exception as e:
        return None, str(e), None

def main():
    # --- 配置 ---
    MODEL_PATH = 'best_fine_grained_model.pth'
    TEST_FOLDER = "G:/乌七八糟的东西/步频数据/splited"
    MODEL_TYPE = 'lightweight'  # 'lightweight' 或 'full'
    SEQUENCE_LENGTH = 100
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件 '{MODEL_PATH}' 不存在")
        return
        
    model, extractor = load_model(MODEL_PATH, model_type=MODEL_TYPE)
    
    if TEST_FOLDER and os.path.isdir(TEST_FOLDER):
        print(f"\n--- 正在评估文件夹: {TEST_FOLDER} ---")
        detailed_evaluation(model, extractor, TEST_FOLDER, sequence_length=SEQUENCE_LENGTH)
    else:
        print(f"测试文件夹 '{TEST_FOLDER}' 不存在或未设置。您可以设置单个文件路径进行预测。")
        # 示例: 单个文件预测
        # SINGLE_FILE = "path/to/your/single/file.csv"
        # if os.path.exists(SINGLE_FILE):
        #     predicted_steps, error = predict_single_file(model, extractor, SINGLE_FILE, SEQUENCE_LENGTH)
        #     if error:
        #         print(f"预测文件 {SINGLE_FILE} 出错: {error}")
        #     else:
        #         print(f"文件 {SINGLE_FILE} 的预测步数是: {predicted_steps}")

if __name__ == '__main__':
    main()
