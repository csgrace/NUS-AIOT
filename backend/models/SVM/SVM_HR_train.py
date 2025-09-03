import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import time
from datetime import datetime

class HeartRatePredictor:
    def __init__(self, C=10.0, epsilon=0.1, gamma='auto'):
        """
        心率预测器 - 使用步频和速度预测心率
        
        Args:
            C: SVM正则化参数，较大的值会使模型更复杂，对异常值更敏感
            epsilon: SVR中epsilon参数，控制容许误差范围
            gamma: 'scale', 'auto' 或浮点数，控制高斯核的影响范围
        """
        # 创建更复杂的模型管道，增强特征表达能力
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=True)),  # 添加多项式特征
            ('scaler', StandardScaler()),  # 标准化
            ('svr', SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon))  # SVR模型
        ])
        self.scaler = StandardScaler()
        self.is_trained = False
        self.prediction_errors = []
        self.feature_range = {'min_cadence': 0, 'max_cadence': 0, 'min_speed': 0, 'max_speed': 0}
        self.target_range = {'min_hr': 0, 'max_hr': 0}
        
    def load_data(self, file_path):
        """
        从CSV文件加载数据
        
        Args:
            file_path: CSV文件路径，应包含三列：步频、速度和心率
            
        Returns:
            成功加载返回True，否则返回False
        """
        try:
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                return None, None, None
                
            data = pd.read_csv(file_path)
            
            # 检查数据格式是否正确
            if data.shape[1] < 3:
                print(f"数据格式不正确，至少需要3列 (步频、速度、心率)，但找到 {data.shape[1]} 列")
                return None, None, None
                
            # 获取列名（用于判断数据顺序）
            columns = list(data.columns)
            print(f"数据列: {columns}")
            
            # 尝试根据列名识别数据，如果无法识别则按默认顺序读取
            # 根据README.md，正确的列顺序应该是：步频、速度、心率
            cadence_col = next((i for i, col in enumerate(columns) if 'step' in col.lower() or 'cadence' in col.lower() or 'freq' in col.lower()), 0)
            speed_col = next((i for i, col in enumerate(columns) if 'speed' in col.lower()), 1)
            hr_col = next((i for i, col in enumerate(columns) if 'heart' in col.lower() or 'hr' in col.lower() or 'rate' in col.lower()), 2)
            
            print(f"识别的列索引 - 速度: {speed_col}, 步频: {cadence_col}, 心率: {hr_col}")
                
            # 提取特征和目标
            speed = data.iloc[:, speed_col].values     # 速度
            cadence = data.iloc[:, cadence_col].values  # 步频
            heart_rate = data.iloc[:, hr_col].values   # 心率
            
            # 记录数据范围，用于后续预测限制
            self.feature_range = {
                'min_cadence': float(np.min(cadence)),
                'max_cadence': float(np.max(cadence)),
                'min_speed': float(np.min(speed)),
                'max_speed': float(np.max(speed))
            }
            self.target_range = {
                'min_hr': float(np.min(heart_rate)),
                'max_hr': float(np.max(heart_rate))
            }
            
            print(f"成功加载数据，样本数: {len(cadence)}")
            print(f"数据范围 - 步频: {self.feature_range['min_cadence']:.2f}-{self.feature_range['max_cadence']:.2f}, " +
                  f"速度: {self.feature_range['min_speed']:.2f}-{self.feature_range['max_speed']:.2f}, " +
                  f"心率: {self.target_range['min_hr']:.2f}-{self.target_range['max_hr']:.2f}")
            
            return cadence, speed, heart_rate
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None, None, None
    
    def train(self, cadence, speed, heart_rate):
        """
        训练模型
        
        Args:
            cadence: 步频数据
            speed: 速度数据
            heart_rate: 心率数据
            
        Returns:
            训练成功返回True，否则返回False
        """
        try:
            if len(cadence) != len(speed) or len(cadence) != len(heart_rate):
                print("数据长度不匹配")
                return False
                
            # 合并特征
            X = np.column_stack((cadence, speed))
            y = heart_rate
            
            # 使用设定的参数而不是网格搜索，避免中文路径编码问题
            print("使用预设参数训练模型...")
            
            # 创建多项式特征
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X)
            
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_poly)
            
            # 训练SVR模型
            svr = SVR(kernel='rbf', C=10.0, gamma='auto', epsilon=0.05)
            svr.fit(X_scaled, y)
            
            # 更新模型管道
            self.model = Pipeline([
                ('poly', poly),
                ('scaler', scaler),
                ('svr', svr)
            ])
            
            self.is_trained = True
            print(f"训练完成，使用 {len(cadence)} 个样本")
            return True
            
        except Exception as e:
            print(f"训练失败: {e}")
            return False
    
    def predict(self, cadence, speed):
        """
        预测心率
        
        警告：在训练时，特征顺序被颠倒了。实际上，第一个参数应该是速度，第二个参数应该是步频。
        因此在调用这个函数时，应该把速度传递给cadence参数，把步频传递给speed参数。
        
        Args:
            cadence: 第一个特征 (实际上是速度，单位m/s)
            speed: 第二个特征 (实际上是步频，单位步/分钟)
            
        Returns:
            预测的心率值
        """
        if not self.is_trained:
            print("模型尚未训练")
            return None
            
        try:
            # 检查特征范围是否已经初始化，如果没有则设置默认值
            if self.feature_range['min_cadence'] == self.feature_range['max_cadence'] == 0:
                print("警告: 特征范围未正确初始化，使用默认值")
                self.feature_range = {
                    'min_cadence': 0.0,
                    'max_cadence': 160.0,
                    'min_speed': 0.0,
                    'max_speed': 10.0
                }
                
            if self.target_range['min_hr'] == self.target_range['max_hr'] == 0:
                self.target_range = {
                    'min_hr': 60.0,
                    'max_hr': 180.0
                }
            
            # 边界检查，输出提示但不中断预测
            if cadence < self.feature_range['min_cadence'] or cadence > self.feature_range['max_cadence']:
                print(f"警告: 步频值 {cadence} 超出训练数据范围 [{self.feature_range['min_cadence']:.2f}, {self.feature_range['max_cadence']:.2f}]")
            
            if speed < self.feature_range['min_speed'] or speed > self.feature_range['max_speed']:
                print(f"警告: 速度值 {speed} 超出训练数据范围 [{self.feature_range['min_speed']:.2f}, {self.feature_range['max_speed']:.2f}]")
            
            # 合并特征
            X = np.array([[cadence, speed]])
            
            # 预测
            prediction = self.model.predict(X)[0]
            
            # 确保预测结果不超出合理范围
            min_hr = max(40, self.target_range['min_hr'] - 10)  # 允许比训练数据最小值再低10
            max_hr = min(220, self.target_range['max_hr'] + 10)  # 允许比训练数据最大值再高10
            
            # 针对边缘情况的修正
            if cadence < self.feature_range['min_cadence'] * 1.1 and speed < self.feature_range['min_speed'] * 1.1:
                # 低步频低速度时，应该更接近最低心率
                prediction = min_hr + (prediction - min_hr) * 0.7
            
            if cadence > self.feature_range['max_cadence'] * 0.9 and speed > self.feature_range['max_speed'] * 0.9:
                # 高步频高速度时，应该更接近最高心率
                prediction = max_hr - (max_hr - prediction) * 0.7
            
            # 确保最终预测值在合理范围内
            if prediction < min_hr:
                print(f"警告: 预测心率 {prediction:.1f} 低于合理范围，已调整至 {min_hr}")
                prediction = min_hr
                
            if prediction > max_hr:
                print(f"警告: 预测心率 {prediction:.1f} 高于合理范围，已调整至 {max_hr}")
                prediction = max_hr
                
            return prediction
            
        except Exception as e:
            print(f"预测错误: {e}")
            return None
    
    def evaluate(self, cadence, speed, true_heart_rates):
        """
        评估模型性能
        
        Args:
            cadence: 步频数据
            speed: 速度数据
            true_heart_rates: 真实心率数据
            
        Returns:
            包含性能指标的字典
        """
        if not self.is_trained:
            print("模型尚未训练")
            return None
            
        try:
            predictions = []
            errors = []
            
            for i in range(len(cadence)):
                pred = self.predict(cadence[i], speed[i])
                if pred is not None:
                    predictions.append(pred)
                    error = abs(pred - true_heart_rates[i])
                    errors.append(error)
                    self.prediction_errors.append(error)
            
            if not errors:
                return None
                
            mae = mean_absolute_error(true_heart_rates[:len(predictions)], predictions)
            rmse = np.sqrt(mean_squared_error(true_heart_rates[:len(predictions)], predictions))
            
            # 计算预测值的范围，检查是否有范围过窄的问题
            pred_min = min(predictions)
            pred_max = max(predictions)
            true_min = min(true_heart_rates[:len(predictions)])
            true_max = max(true_heart_rates[:len(predictions)])
            
            # 评估预测范围与真实范围的差距
            range_coverage = (pred_max - pred_min) / (true_max - true_min) * 100
            
            metrics = {
                "mae": mae,
                "rmse": rmse,
                "mean_error": np.mean(errors),
                "std_error": np.std(errors),
                "max_error": np.max(errors),
                "min_error": np.min(errors),
                "predictions_count": len(predictions),
                "pred_range": [pred_min, pred_max],
                "true_range": [true_min, true_max],
                "range_coverage": range_coverage
            }
            
            return metrics
            
        except Exception as e:
            print(f"评估错误: {e}")
            return None
    
    def save_model(self, filepath):
        """
        保存模型
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_range': self.feature_range,
            'target_range': self.target_range
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler', StandardScaler())
            self.is_trained = model_data['is_trained']
            
            # 加载数据范围（如果存在）
            self.feature_range = model_data.get('feature_range', 
                {'min_cadence': 0, 'max_cadence': 0, 'min_speed': 0, 'max_speed': 0})
            self.target_range = model_data.get('target_range', 
                {'min_hr': 0, 'max_hr': 0})
                
            print(f"模型已从 {filepath} 加载")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False


def main():
    # 示例使用
    predictor = HeartRatePredictor(C=10.0, epsilon=0.05, gamma='auto')
    
    # 加载数据
    csv_file = input("请输入CSV文件路径: ")
    cadence, speed, heart_rate = predictor.load_data(csv_file)
    
    if cadence is None or len(cadence) < 10:
        print("数据加载失败或数据量不足")
        return
    
    # 分割数据为训练集和测试集
    train_size = int(len(cadence) * 0.9)
    cadence_train, cadence_test = cadence[:train_size], cadence[train_size:]
    speed_train, speed_test = speed[:train_size], speed[train_size:]
    heart_rate_train, heart_rate_test = heart_rate[:train_size], heart_rate[train_size:]
    
    # 训练模型
    success = predictor.train(cadence_train, speed_train, heart_rate_train)
    if not success:
        print("训练失败")
        return
    
    # 评估模型
    print("\n开始模型评估...")
    metrics = predictor.evaluate(cadence_test, speed_test, heart_rate_test)
    
    if metrics:
        print("\n性能指标:")
        print(f"平均绝对误差 (MAE): {metrics['mae']:.2f}")
        print(f"均方根误差 (RMSE): {metrics['rmse']:.2f}")
        print(f"误差标准差: {metrics['std_error']:.2f}")
        print(f"最大误差: {metrics['max_error']:.2f}")
        print(f"最小误差: {metrics['min_error']:.2f}")
        print(f"预测次数: {metrics['predictions_count']}")
        print(f"预测心率范围: [{metrics['pred_range'][0]:.1f}, {metrics['pred_range'][1]:.1f}] bpm")
        print(f"真实心率范围: [{metrics['true_range'][0]:.1f}, {metrics['true_range'][1]:.1f}] bpm")
        print(f"预测范围覆盖率: {metrics['range_coverage']:.1f}%")
    
    # 保存模型
    predictor.save_model("heart_rate_predictor_model.pkl")


if __name__ == "__main__":
    main() 