#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合并GPS、步频和心率数据 - 重采样和插值版本
使用pandas的重采样和插值功能，生成均匀间隔的合并数据
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_data(file_path):
    """加载CSV数据并确保时间列被正确解析"""
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    # 将时间设为索引，便于重采样
    df = df.set_index('time')
    return df

def resample_and_interpolate(df, freq='1s', method='linear'):
    """
    对数据进行重采样和插值
    
    参数:
    - df: 输入的DataFrame，时间列应已设为索引
    - freq: 重采样频率，如'1s'表示1秒
    - method: 插值方法，如'linear', 'cubic', 'spline'等
    
    返回:
    - 重采样和插值后的DataFrame
    """
    # 重采样到指定频率
    resampled = df.resample(freq)
    
    # 使用指定方法进行插值
    interpolated = resampled.interpolate(method=method)
    
    return interpolated

def main():
    # 数据文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gps_file = os.path.join(current_dir, 'gps_data.csv')
    steps_file = os.path.join(current_dir, 'band_steps.csv')
    heart_rate_file = os.path.join(current_dir, 'band_heart_rate.csv')
    
    # 加载数据
    gps_df = load_data(gps_file)
    steps_df = load_data(steps_file)
    heart_rate_df = load_data(heart_rate_file)
    
    # 确定全局的时间范围
    min_time = min(gps_df.index.min(), steps_df.index.min(), heart_rate_df.index.min())
    max_time = max(gps_df.index.max(), steps_df.index.max(), heart_rate_df.index.max())
    
    print(f"数据时间范围: {min_time} 到 {max_time}")
    
    # 创建一个共同的时间范围DataFrame
    # 使用1秒的频率进行重采样 (可根据需要调整)
    freq = '1s'
    time_range = pd.date_range(start=min_time, end=max_time, freq=freq)
    
    # 对各个数据集进行重采样和插值
    gps_resampled = resample_and_interpolate(gps_df, freq=freq, method='linear')
    steps_resampled = resample_and_interpolate(steps_df, freq=freq, method='linear')
    heart_rate_resampled = resample_and_interpolate(heart_rate_df, freq=freq, method='linear')
    
    # 合并数据集
    merged_df = pd.DataFrame(index=time_range)
    merged_df['speed'] = gps_resampled['speed']
    merged_df['step_frequency'] = steps_resampled['step_frequency']
    merged_df['heart_rate'] = heart_rate_resampled['heart_rate']
    
    # 重置索引，将时间作为列
    merged_df = merged_df.reset_index()
    merged_df = merged_df.rename(columns={'index': 'time'})
    
    # 可选：去除任何仍然含有NaN的行
    merged_df_no_nan = merged_df.dropna()
    
    # 保存合并后的数据
    output_file = os.path.join(current_dir, 'merged_data_resampled.csv')
    merged_df.to_csv(output_file, index=False)
    print(f'合并完成，数据已保存至: {output_file}')
    print(f'合并后的数据总行数: {len(merged_df)}')
    
    # 保存去除NaN的数据
    no_nan_output = os.path.join(current_dir, 'merged_data_resampled_no_nan.csv')
    merged_df_no_nan.to_csv(no_nan_output, index=False)
    print(f'去除NaN后的数据行数: {len(merged_df_no_nan)}')
    print(f'去除NaN后的数据已保存至: {no_nan_output}')
    
    # 高级：尝试不同的插值方法
    methods = ['linear', 'cubic', 'spline']
    for method in methods:
        try:
            # 对GPS数据使用不同的插值方法
            gps_resampled_method = resample_and_interpolate(gps_df, freq=freq, method=method)
            
            # 合并数据
            merged_method = pd.DataFrame(index=time_range)
            merged_method['speed'] = gps_resampled_method['speed']
            merged_method['step_frequency'] = steps_resampled['step_frequency']
            merged_method['heart_rate'] = heart_rate_resampled['heart_rate']
            
            # 重置索引，将时间作为列
            merged_method = merged_method.reset_index()
            merged_method = merged_method.rename(columns={'index': 'time'})
            
            # 去除任何仍然含有NaN的行
            merged_method_no_nan = merged_method.dropna()
            
            # 保存数据
            method_output = os.path.join(current_dir, f'merged_data_resampled_{method}.csv')
            merged_method_no_nan.to_csv(method_output, index=False)
            print(f'使用{method}插值方法的数据行数: {len(merged_method_no_nan)}')
            print(f'使用{method}插值方法的数据已保存至: {method_output}')
        except Exception as e:
            print(f'使用{method}插值方法时出错: {e}')

if __name__ == "__main__":
    main()
