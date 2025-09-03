#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合并GPS、步频和心率数据 - 高效版本
使用pandas的merge_asof函数实现最近时间点匹配
"""

import pandas as pd
import os
from datetime import timedelta

def load_and_prepare_data(file_path):
    """加载CSV数据并进行预处理"""
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')  # 确保数据按时间排序
    return df

def merge_with_time_window(base_df, other_dfs, other_names, tolerance=None):
    """
    使用pandas的merge_asof函数合并数据
    
    参数:
    - base_df: 作为基准的DataFrame
    - other_dfs: 其他要合并的DataFrame列表
    - other_names: 其他DataFrame的名称列表，用于生成列名
    - tolerance: 匹配的最大时间差，None表示不限制
    
    返回:
    - 合并后的DataFrame
    """
    result = base_df.copy()
    
    # 依次合并每个DataFrame
    for df, name in zip(other_dfs, other_names):
        # 使用merge_asof函数，以最近的时间点进行匹配
        result = pd.merge_asof(
            result, 
            df,
            on='time',  # 按时间列合并
            direction='nearest',  # 使用最近的时间点
            tolerance=tolerance  # 可选的最大时间差
        )
        
        # 如果存在重名列（通常是time列），重命名右侧的列
        duplicate_cols = [col for col in result.columns if col.endswith('_y')]
        for col in duplicate_cols:
            original_col = col[:-2]  # 去掉_y后缀
            result = result.rename(columns={col: f'{name}_{original_col}'})
            # 删除_x后缀的列
            if f'{original_col}_x' in result.columns:
                result = result.rename(columns={f'{original_col}_x': original_col})
    
    return result

def calculate_time_diffs(merged_df, original_dfs, df_names):
    """计算每个匹配的时间差"""
    for df, name in zip(original_dfs, df_names):
        # 为每个原始数据源创建一个时间差列
        diff_col_name = f'{name}_time_diff_seconds'
        merged_df[diff_col_name] = None
        
        # 计算每行的时间差
        for idx, row in merged_df.iterrows():
            base_time = row['time']
            
            # 找到原始DataFrame中最接近的时间
            closest_time = df['time'].iloc[(df['time'] - base_time).abs().argmin()]
            
            # 计算时间差（秒）
            time_diff = abs((base_time - closest_time).total_seconds())
            merged_df.loc[idx, diff_col_name] = time_diff
    
    return merged_df

def main():
    # 数据文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gps_file = os.path.join(current_dir, 'gps_data.csv')
    steps_file = os.path.join(current_dir, 'band_steps.csv')
    heart_rate_file = os.path.join(current_dir, 'band_heart_rate.csv')
    
    # 加载并预处理数据
    gps_df = load_and_prepare_data(gps_file)
    steps_df = load_and_prepare_data(steps_file)
    heart_rate_df = load_and_prepare_data(heart_rate_file)
    
    # 合并数据（不设置时间限制）
    other_dfs = [steps_df, heart_rate_df]
    other_names = ['steps', 'heart']
    
    merged_df = merge_with_time_window(gps_df, other_dfs, other_names)
    
    # 计算并添加时间差列
    merged_df = calculate_time_diffs(merged_df, other_dfs, other_names)
    
    # 保存合并后的数据
    output_file = os.path.join(current_dir, 'merged_data_efficient.csv')
    merged_df.to_csv(output_file, index=False)
    print(f'合并完成，数据已保存至: {output_file}')
    
    # 输出一些统计信息
    print(f'合并后的数据总行数: {len(merged_df)}')
    print(f'步频数据时间差的平均值: {merged_df["steps_time_diff_seconds"].mean():.2f} 秒')
    print(f'心率数据时间差的平均值: {merged_df["heart_time_diff_seconds"].mean():.2f} 秒')
    
    # 过滤时间差过大的数据
    max_allowed_diff = 5.0  # 最大允许时间差（秒）
    filtered_df = merged_df[
        (merged_df['steps_time_diff_seconds'] <= max_allowed_diff) & 
        (merged_df['heart_time_diff_seconds'] <= max_allowed_diff)
    ]
    
    filtered_output = os.path.join(current_dir, 'merged_data_efficient_filtered.csv')
    filtered_df.to_csv(filtered_output, index=False)
    print(f'过滤后的数据行数: {len(filtered_df)}')
    print(f'过滤后的数据已保存至: {filtered_output}')
    
    # 另一种方法：使用时间窗口直接限制合并
    # 设置最大允许时间差为5秒
    tolerance = pd.Timedelta(seconds=5)
    
    windowed_df = merge_with_time_window(gps_df, other_dfs, other_names, tolerance)
    windowed_output = os.path.join(current_dir, 'merged_data_windowed.csv')
    windowed_df.to_csv(windowed_output, index=False)
    print(f'使用时间窗口方法合并的数据行数: {len(windowed_df)}')
    print(f'时间窗口方法的数据已保存至: {windowed_output}')

if __name__ == "__main__":
    main()
