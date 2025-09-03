#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
心率预测器 - 统一接口
提供简单的训练和预测接口，只需传入CSV文件即可使用
"""

import os
import sys
import pandas as pd
import numpy as np
from .SVM_HR_train import HeartRatePredictor

class HeartRateManager:
    """心率预测管理器 - 提供统一的训练和预测接口"""
    
    def __init__(self, model_path="heart_rate_model.pkl"):
        """
        初始化心率预测管理器
        
        Args:
            model_path: 模型保存/加载路径
        """
        self.model_path = model_path
        self.predictor = HeartRatePredictor(C=10.0, epsilon=0.05, gamma='auto')
        self.is_model_loaded = False
        
    def train_from_csv(self, csv_file, save_model=True, test_split=0.2):
        """
        从CSV文件训练模型
        
        Args:
            csv_file: CSV文件路径，格式：cadence,speed,heart_rate
            save_model: 是否保存训练好的模型
            test_split: 测试集比例
            
        Returns:
            dict: 训练结果，包含性能指标
        """
        print(f"🏃 开始从CSV文件训练模型: {csv_file}")
        
        # 检查文件是否存在
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV文件不存在: {csv_file}")
        
        # 加载数据
        cadence, speed, heart_rate = self.predictor.load_data(csv_file)
        
        if cadence is None:
            raise ValueError("数据加载失败，请检查CSV文件格式")
        
        if len(cadence) < 10:
            raise ValueError(f"数据量不足，至少需要10个样本，当前只有{len(cadence)}个")
        
        # 分割训练和测试数据
        train_size = int(len(cadence) * (1 - test_split))
        
        cadence_train = cadence[:train_size]
        speed_train = speed[:train_size]
        heart_rate_train = heart_rate[:train_size]
        
        cadence_test = cadence[train_size:]
        speed_test = speed[train_size:]
        heart_rate_test = heart_rate[train_size:]
        
        print(f"📊 数据分割: 训练集{len(cadence_train)}个样本, 测试集{len(cadence_test)}个样本")
        
        # 训练模型
        success = self.predictor.train(cadence_train, speed_train, heart_rate_train)
        
        if not success:
            raise RuntimeError("模型训练失败")
        
        print("✅ 模型训练完成")
        
        # 评估模型
        metrics = None
        if len(cadence_test) > 0:
            print("📈 开始模型评估...")
            metrics = self.predictor.evaluate(cadence_test, speed_test, heart_rate_test)
            
            if metrics:
                print(f"\n📊 模型性能指标:")
                print(f"  平均绝对误差 (MAE): {metrics['mae']:.2f} bpm")
                print(f"  均方根误差 (RMSE): {metrics['rmse']:.2f} bpm")
                print(f"  最大误差: {metrics['max_error']:.2f} bpm")
                print(f"  最小误差: {metrics['min_error']:.2f} bpm")
                print(f"  预测次数: {metrics['predictions_count']}")
        
        # 保存模型
        if save_model:
            self.predictor.save_model(self.model_path)
            print(f"💾 模型已保存到: {self.model_path}")
            self.is_model_loaded = True
        
        return {
            'success': True,
            'metrics': metrics,
            'train_samples': len(cadence_train),
            'test_samples': len(cadence_test),
            'model_path': self.model_path if save_model else None
        }
    
    def load_model(self, model_path=None):
        """
        加载已训练的模型
        
        Args:
            model_path: 模型文件路径，如果为None则使用默认路径
            
        Returns:
            bool: 加载是否成功
        """
        if model_path is None:
            model_path = self.model_path
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        success = self.predictor.load_model(model_path)
        if success:
            self.is_model_loaded = True
            print(f"✅ 模型加载成功: {model_path}")
        else:
            print(f"❌ 模型加载失败: {model_path}")
        
        return success
    
    def predict_single(self, cadence, speed):
        """
        预测单个样本的心率
        
        Args:
            cadence: 步频 (步/分钟)
            speed: 速度 (km/h)
            
        Returns:
            float: 预测的心率值，失败返回None
        """
        if not self.is_model_loaded:
            print("❌ 模型未加载，请先训练或加载模型")
            return None
        
        try:
            heart_rate = self.predictor.predict(cadence, speed)
            return heart_rate
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def predict_from_csv(self, csv_file, output_file=None):
        """
        从CSV文件批量预测心率
        
        Args:
            csv_file: 输入CSV文件路径，格式：cadence,speed
            output_file: 输出CSV文件路径，如果为None则不保存
            
        Returns:
            pandas.DataFrame: 包含预测结果的DataFrame
        """
        if not self.is_model_loaded:
            raise RuntimeError("模型未加载，请先训练或加载模型")
        
        print(f"📊 开始批量预测: {csv_file}")
        
        # 读取数据
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"读取CSV文件失败: {e}")
        
        # 检查列名
        required_cols = ['cadence', 'speed']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"CSV文件缺少必要的列: {missing_cols}")
        
        # 批量预测
        predictions = []
        for idx, row in df.iterrows():
            cadence = row['cadence']
            speed = row['speed']
            
            heart_rate = self.predict_single(cadence, speed)
            predictions.append(heart_rate)
        
        # 添加预测结果到DataFrame
        df['predicted_heart_rate'] = predictions
        
        # 保存结果
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"💾 预测结果已保存到: {output_file}")
        
        print(f"✅ 批量预测完成，共处理{len(df)}个样本")
        
        return df
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            dict: 模型信息
        """
        if not self.is_model_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_path": self.model_path,
            "feature_range": self.predictor.feature_range,
            "target_range": self.predictor.target_range,
            "is_trained": self.predictor.is_trained
        }

def quick_train(csv_file, model_path="heart_rate_model.pkl"):
    """
    快速训练函数 - 一行代码完成训练
    
    Args:
        csv_file: 训练数据CSV文件路径
        model_path: 模型保存路径
        
    Returns:
        HeartRateManager: 训练好的管理器实例
    """
    manager = HeartRateManager(model_path)
    result = manager.train_from_csv(csv_file)
    return manager

def quick_predict(cadence, speed, model_path="backend/models/SVM/SVM_models/newest.pkl"):
    """
    快速预测函数 - 一行代码完成预测
    
    Args:
        cadence: 步频 (步/分钟)
        speed: 速度 (km/h)
        model_path: 模型文件路径
        
    Returns:
        float: 预测的心率值
    """
    manager = HeartRateManager(model_path)
    if not manager.load_model():
        raise RuntimeError(f"无法加载模型: {model_path}")
    
    return manager.predict_single(cadence, speed)

# 使用示例
if __name__ == "__main__":
    # 示例1: 训练模型
    print("=== 示例1: 训练模型 ===")
    manager = HeartRateManager("my_heart_rate_model.pkl")
    
    # 检查是否有训练数据
    training_data = "heart_rate_training_data.csv"
    if os.path.exists(training_data):
        try:
            result = manager.train_from_csv(training_data)
            print(f"训练结果: {result}")
        except Exception as e:
            print(f"训练失败: {e}")
    else:
        print(f"训练数据文件不存在: {training_data}")
        print("请先运行 generate_training_data.py 生成训练数据")
    
    # 示例2: 预测
    print("\n=== 示例2: 预测 ===")
    if manager.is_model_loaded:
        # 单个预测
        hr = manager.predict_single(160, 12)
        print(f"步频160, 速度12km/h -> 预测心率: {hr:.1f} bpm")
        
        # 模型信息
        info = manager.get_model_info()
        print(f"模型信息: {info}")
    
    # 示例3: 快速使用
    print("\n=== 示例3: 快速使用 ===")
    if os.path.exists(training_data):
        try:
            # 快速训练
            quick_manager = quick_train(training_data, "quick_model.pkl")
            
            # 快速预测
            hr = quick_predict(180, 15, "quick_model.pkl")
            print(f"快速预测 - 步频180, 速度15km/h -> 心率: {hr:.1f} bpm")
        except Exception as e:
            print(f"快速使用失败: {e}")
