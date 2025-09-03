#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from SVM_HR_train import HeartRatePredictor

# 默认模型路径
DEFAULT_MODEL_PATH = "backend/models/SVM/SVM_models/newest.pkl"

def predict_heart_rate(speed, cadence, model_path=DEFAULT_MODEL_PATH):
    """
    预测心率
    
    Args:
        speed: 速度 (m/s)
        cadence: 步频 (步/分钟)
        model_path: 模型路径
        
    Returns:
        预测的心率值，如果出错则返回None
    """
    # 创建预测器
    predictor = HeartRatePredictor()
    
    # 加载模型
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        return None
    
    if not predictor.load_model(model_path):
        print(f"错误：无法加载模型: {model_path}")
        return None
    
    # 设置特征范围，由于训练时特征顺序颠倒，这里设置的范围也需要对应调整
    # 实际上模型把第一个输入当作速度，第二个输入当作步频
    predictor.feature_range = {
        'min_cadence': 0.0,     # 这实际上是速度的最小值
        'max_cadence': 10.0,    # 这实际上是速度的最大值
        'min_speed': 0.0,       # 这实际上是步频的最小值
        'max_speed': 200.0      # 这实际上是步频的最大值
    }
    
    predictor.target_range = {
        'min_hr': 60.0,
        'max_hr': 180.0
    }
    
    # 预测心率 - 注意：由于训练时特征顺序颠倒，这里需要交换参数位置
    try:
        # 交换参数顺序：把速度传给cadence参数，把步频传给speed参数
        heart_rate = predictor.predict(speed, cadence)  # 故意颠倒参数顺序以匹配训练时的错误
        
        # 确保心率在合理范围内
        if heart_rate < 40:
            print("警告：预测心率异常低，已调整到最小值60")
            heart_rate = 60
            
        if heart_rate > 200:
            print("警告：预测心率异常高，已调整到最大值180")
            heart_rate = 180
            
        return heart_rate
    except Exception as e:
        print(f"预测错误: {e}")
        return None

def main():
    """命令行界面"""
    # 显示欢迎信息
    print("\n===== 心率预测器 =====")
    print("输入速度和步频，预测对应的心率")
    
    # 加载模型
    model_path = DEFAULT_MODEL_PATH
    
    # 检查命令行参数中是否提供了模型路径
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        print(f"您可以指定模型路径作为第一个参数: python {sys.argv[0]} <model_path>")
        return
    
    # 创建预测器并加载模型
    predictor = HeartRatePredictor()
    if not predictor.load_model(model_path):
        print(f"错误：无法加载模型: {model_path}")
        return
    
    print(f"成功加载模型: {model_path}")
    
    # 交互式预测循环
    while True:
        try:
            # 获取用户输入
            print("\n输入 'q' 退出程序")
            
            speed_input = input("请输入速度 (m/s): ")
            if speed_input.lower() == 'q':
                break
            
            cadence_input = input("请输入步频 (步/分钟): ")
            if cadence_input.lower() == 'q':
                break
            
            # 转换输入
            try:
                speed = float(speed_input)
                cadence = float(cadence_input)
            except ValueError:
                print("错误：请输入有效的数字")
                continue
            
            # 预测 - 使用修正后的参数顺序，与函数签名一致
            heart_rate = predict_heart_rate(speed, cadence, model_path)
            
            if heart_rate is not None:
                print(f"\n预测结果:")
                print(f"速度: {speed:.1f} m/s")
                print(f"步频: {cadence:.1f} 步/分钟")
                print(f"预测心率: {heart_rate:.1f} bpm")
            else:
                print("预测失败")
                
        except KeyboardInterrupt:
            print("\n程序已退出")
            break
        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    main() 