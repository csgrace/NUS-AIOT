#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合并GPS、步频和心率数据
使用最近时间点匹配的方式将三个表合并
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_data(file_path):
    """加载CSV数据并确保时间列被正确解析"""
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    return df

def find_closest_time(target_time, time_series):
    """查找时间序列中与目标时间最接近的时间点的索引"""
    # 使用numpy的绝对差值计算最接近的时间点
    time_diff = np.abs(time_series - target_time)
    return time_diff.argmin()

def merge_data_by_closest_time(base_df, other_dfs, other_cols):
    """基于最接近的时间点合并数据集"""
    # 创建结果DataFrame，初始只包含基础数据
    result = base_df.copy()
    
    # 为每个其他DataFrame添加相应的列
    for df, col in zip(other_dfs, other_cols):
        # 创建新列，初始化为NaN
        result[col] = np.nan
        
        # 对基准DataFrame中的每个时间点
        for idx, row in result.iterrows():
            target_time = row['time']
            
            # 找到其他DataFrame中最接近的时间点
            closest_idx = find_closest_time(target_time, df['time'])
            
            # 获取对应的值并添加到结果中
            result.loc[idx, col] = df.iloc[closest_idx][col]
    
    return result

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
    
    # 以GPS数据为基准合并数据
    other_dfs = [steps_df, heart_rate_df]
    other_cols = ['step_frequency', 'heart_rate']
    
    merged_df = merge_data_by_closest_time(gps_df, other_dfs, other_cols)
    
    # 可选：为结果增加一个时间差列，表示每个匹配的最大时间差（单位：秒）
    time_diff_steps = []
    time_diff_heart = []
    
    for idx, row in merged_df.iterrows():
        target_time = row['time']
        
        # 找到步频数据中最接近的时间点
        closest_idx_steps = find_closest_time(target_time, steps_df['time'])
        steps_time = steps_df.iloc[closest_idx_steps]['time']
        time_diff_steps.append(abs((target_time - steps_time).total_seconds()))
        
        # 找到心率数据中最接近的时间点
        closest_idx_heart = find_closest_time(target_time, heart_rate_df['time'])
        heart_time = heart_rate_df.iloc[closest_idx_heart]['time']
        time_diff_heart.append(abs((target_time - heart_time).total_seconds()))
    
    merged_df['steps_time_diff_seconds'] = time_diff_steps
    merged_df['heart_rate_time_diff_seconds'] = time_diff_heart
    
    # 保存合并后的数据
    output_file = os.path.join(current_dir, 'merged_data.csv')
    merged_df.to_csv(output_file, index=False)
    print(f'合并完成，数据已保存至: {output_file}')
    
    # 输出一些统计信息
    print(f'合并后的数据总行数: {len(merged_df)}')
    print(f'步频数据时间差的平均值: {merged_df["steps_time_diff_seconds"].mean():.2f} 秒')
    print(f'心率数据时间差的平均值: {merged_df["heart_rate_time_diff_seconds"].mean():.2f} 秒')
    
    # 可选：过滤时间差过大的数据
    max_allowed_diff = 5.0  # 最大允许时间差（秒）
    filtered_df = merged_df[
        (merged_df['steps_time_diff_seconds'] <= max_allowed_diff) & 
        (merged_df['heart_rate_time_diff_seconds'] <= max_allowed_diff)
    ]
    
    filtered_output = os.path.join(current_dir, 'merged_data_filtered.csv')
    filtered_df.to_csv(filtered_output, index=False)
    print(f'过滤后的数据行数: {len(filtered_df)}')
    print(f'过滤后的数据已保存至: {filtered_output}')

if __name__ == "__main__":
    main()
