# 心率预测器使用指南

## 概述

这是一个基于SVM的心率预测系统，提供了简单易用的接口来训练模型和预测心率。只需要提供CSV文件即可完成训练和预测。

## 文件结构

```
backend/models/SVM/
├── heart_rate_predictor.py    # 核心预测器类
├── hr_cli.py                  # 命令行工具
├── generate_training_data.py  # 训练数据生成器
├── SVM_HR_train.py           # 原始训练代码
└── README_HeartRatePredictor.md  # 本文档
```

## 快速开始

### 1. 生成训练数据

```bash
python generate_training_data.py
```

这将生成 `heart_rate_training_data.csv` 文件，包含1000+个训练样本。

### 2. 训练模型

#### 方法1: 使用命令行工具
```bash
python hr_cli.py train heart_rate_training_data.csv
```

#### 方法2: 使用Python代码
```python
from heart_rate_predictor import quick_train

# 一行代码完成训练
manager = quick_train("heart_rate_training_data.csv")
```

### 3. 预测心率

#### 方法1: 命令行单个预测
```bash
python hr_cli.py predict -c 160 -s 12
```

#### 方法2: 交互式预测
```bash
python hr_cli.py interactive
```

#### 方法3: Python代码预测
```python
from heart_rate_predictor import quick_predict

# 一行代码完成预测
heart_rate = quick_predict(160, 12)  # 步频160, 速度12km/h
print(f"预测心率: {heart_rate:.1f} bpm")
```

## 详细使用说明

### 数据格式要求

#### 训练数据CSV格式
```csv
cadence,speed,heart_rate
160,12,150
120,8,120
80,4,90
```

- `cadence`: 步频 (步/分钟)
- `speed`: 速度 (km/h)  
- `heart_rate`: 心率 (bpm)

#### 预测数据CSV格式
```csv
cadence,speed
160,12
120,8
80,4
```

### 命令行工具详细用法

#### 训练模型
```bash
# 基本训练
python hr_cli.py train data.csv

# 指定模型保存路径
python hr_cli.py train data.csv -m my_model.pkl

# 指定测试集比例
python hr_cli.py train data.csv -t 0.3
```

#### 单个预测
```bash
# 预测单个样本
python hr_cli.py predict -c 160 -s 12

# 使用指定模型
python hr_cli.py predict -c 160 -s 12 -m my_model.pkl
```

#### 批量预测
```bash
# 批量预测
python hr_cli.py predict -b input.csv

# 指定输出文件
python hr_cli.py predict -b input.csv -o output.csv
```

#### 查看模型信息
```bash
python hr_cli.py info
python hr_cli.py info -m my_model.pkl
```

### Python API详细用法

#### 使用HeartRateManager类

```python
from heart_rate_predictor import HeartRateManager

# 创建管理器
manager = HeartRateManager("my_model.pkl")

# 训练模型
result = manager.train_from_csv("training_data.csv")
print(f"训练结果: {result}")

# 单个预测
heart_rate = manager.predict_single(160, 12)
print(f"预测心率: {heart_rate:.1f} bpm")

# 批量预测
df = manager.predict_from_csv("input.csv", "output.csv")
print(df.head())

# 获取模型信息
info = manager.get_model_info()
print(f"模型信息: {info}")
```

#### 使用快速函数

```python
from heart_rate_predictor import quick_train, quick_predict

# 快速训练
manager = quick_train("training_data.csv", "my_model.pkl")

# 快速预测
heart_rate = quick_predict(160, 12, "my_model.pkl")
```

## 实际使用示例

### 示例1: 完整的训练和预测流程

```python
from heart_rate_predictor import HeartRateManager

# 1. 创建管理器
manager = HeartRateManager("sports_model.pkl")

# 2. 训练模型
print("开始训练...")
result = manager.train_from_csv("sports_data.csv")

if result['success']:
    print(f"训练成功! MAE: {result['metrics']['mae']:.2f} bpm")
    
    # 3. 预测不同运动强度的心率
    scenarios = [
        (0, 0, "静息"),
        (90, 4, "慢走"),
        (120, 7, "快走"),
        (150, 10, "慢跑"),
        (180, 15, "跑步")
    ]
    
    print("\n运动心率预测:")
    for cadence, speed, activity in scenarios:
        hr = manager.predict_single(cadence, speed)
        print(f"{activity}: 步频{cadence}, 速度{speed}km/h -> {hr:.1f} bpm")
```

### 示例2: 批量处理运动数据

```python
import pandas as pd
from heart_rate_predictor import HeartRateManager

# 创建测试数据
test_data = pd.DataFrame({
    'cadence': [80, 100, 120, 140, 160, 180],
    'speed': [3, 5, 7, 9, 12, 15]
})
test_data.to_csv("test_scenarios.csv", index=False)

# 加载模型并预测
manager = HeartRateManager()
manager.load_model("heart_rate_model.pkl")

# 批量预测
results = manager.predict_from_csv("test_scenarios.csv", "predictions.csv")
print(results)
```

## 性能指标说明

- **MAE (平均绝对误差)**: 预测值与真实值的平均绝对差值，单位bpm
- **RMSE (均方根误差)**: 预测误差的均方根，单位bpm
- **最大误差**: 单个样本的最大预测误差，单位bpm

一般来说：
- MAE < 10 bpm: 优秀
- MAE < 15 bpm: 良好  
- MAE < 20 bpm: 可接受
- MAE > 20 bpm: 需要改进

## 常见问题

### Q: 如何提高预测精度？
A: 
1. 增加训练数据量
2. 确保训练数据覆盖目标预测范围
3. 数据质量要高，避免异常值
4. 可以调整SVM参数(C, epsilon, gamma)

### Q: 预测结果不合理怎么办？
A:
1. 检查输入数据是否在训练范围内
2. 确认CSV文件格式正确
3. 重新训练模型
4. 检查训练数据质量

### Q: 如何处理超出训练范围的数据？
A: 系统会给出警告，但仍会预测。建议：
1. 扩展训练数据范围
2. 对超出范围的预测结果谨慎使用

## 技术细节

- **算法**: Support Vector Regression (SVR)
- **特征**: 步频 + 速度
- **目标**: 心率
- **数据预处理**: MinMax标准化
- **模型保存**: pickle格式

## 更新日志

- v1.0: 初始版本，基本训练和预测功能
- v1.1: 添加命令行工具
- v1.2: 修复列顺序问题，提高预测精度
- v1.3: 添加批量预测和交互式预测功能
