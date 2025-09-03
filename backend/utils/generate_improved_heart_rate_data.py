import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs('data', exist_ok=True)

def generate_improved_heart_rate_data(n_samples=2000):
    print("生成改进的心率数据...")
    
    # 初始化数据容器
    speeds = []
    step_frequencies = []
    heart_rates = []
    
    # 1. 生成均匀分布的数据点，确保覆盖整个范围
    # 将速度范围(0-10)和步频范围(0-200)划分为网格
    speed_grid = np.linspace(0, 15, 20)
    stepfreq_grid = np.linspace(0, 300, 40)
    
    # 为网格中的每个组合生成样本
    grid_samples = int(n_samples * 0.7)  # 70%的样本来自均匀网格
    grid_points = []
    
    for speed in speed_grid:
        for stepfreq in stepfreq_grid:
            grid_points.append((speed, stepfreq))
    
    # 从网格点中随机选择
    selected_indices = np.random.choice(len(grid_points), grid_samples, replace=True)
    for idx in selected_indices:
        speed, stepfreq = grid_points[idx]
        # 添加少量噪声使数据不完全落在网格线上
        speed += np.random.uniform(-0.25, 0.25)
        stepfreq += np.random.uniform(-2.5, 2.5)
        speeds.append(max(0, speed))
        step_frequencies.append(max(0, stepfreq))
    
    # 2. 增加一些符合真实场景的数据点 (30%的样本)
    scene_samples = n_samples - grid_samples
    
    # 静止场景 (约10%)
    still_count = int(scene_samples * 0.1)
    speeds.extend(np.random.uniform(0, 0.5, still_count))
    step_frequencies.extend(np.random.uniform(0, 10, still_count))
    
    # 步行场景 (约10%)
    walk_count = int(scene_samples * 0.1)
    speeds.extend(np.random.uniform(0.5, 2.0, walk_count))
    step_frequencies.extend(np.random.uniform(20, 80, walk_count))
    
    # 慢跑场景 (约5%)
    jog_count = int(scene_samples * 0.05)
    speeds.extend(np.random.uniform(2.0, 4.0, jog_count))
    step_frequencies.extend(np.random.uniform(80, 140, jog_count))
    
    # 跑步场景 (约5%)
    run_count = int(scene_samples * 0.05)
    speeds.extend(np.random.uniform(4.0, 10.0, run_count))
    step_frequencies.extend(np.random.uniform(140, 200, run_count))
    
    # 3. 生成心率值
    # 基于速度和步频的心率计算模型:
    # 心率 = 基础心率 + 速度影响 + 步频影响 + 非线性交互影响 + 随机变化
    
    for i in range(len(speeds)):
        speed = speeds[i]
        step_freq = step_frequencies[i]
        
        # 基础心率 (静息)
        base_hr = 60
        
        # 速度影响 (非线性)
        speed_effect = 8 * np.sqrt(speed) + 2 * speed
        
        # 步频影响 (非线性)
        step_effect = 0.2 * step_freq + 0.01 * step_freq**1.3
        
        # 交互影响 (速度与步频的协同效应)
        interaction_effect = 0.02 * speed * step_freq
        
        # 生理噪声
        noise = np.random.normal(0, 4)
        
        # 计算心率
        heart_rate = base_hr + speed_effect + step_effect + interaction_effect + noise
        
        # 处理边缘情况
        if speed < 0.2 and step_freq < 5:
            # 静止状态
            heart_rate = np.random.uniform(60, 75)
        elif speed > 9 and step_freq > 180:
            # 极限运动
            heart_rate = np.random.uniform(160, 180)
            
        # 确保心率在合理范围内
        heart_rate = max(60, min(180, heart_rate))
        heart_rates.append(heart_rate)
    
    # 4. 创建DataFrame并确保数据点数量正确
    df = pd.DataFrame({
        'speed': np.round(speeds, 4),
        'step_frequency': np.round(step_frequencies, 2),
        'heart_rate': np.round(heart_rates, 0)
    })
    
    # 如果有多余的数据点，随机去除
    if len(df) > n_samples:
        df = df.sample(n_samples)
    
    # 如果数据点不足，通过复制并添加噪声补充
    if len(df) < n_samples:
        samples_needed = n_samples - len(df)
        extra_samples = df.sample(samples_needed, replace=True)
        extra_samples['heart_rate'] = extra_samples['heart_rate'] + np.random.uniform(-5, 5, len(extra_samples))
        df = pd.concat([df, extra_samples])
    
    # 确保心率在合理范围内
    df['heart_rate'] = df['heart_rate'].clip(60, 180).round(0)
    
    # 5. 验证数据的相关性和分布
    corr = df.corr()
    print(f"速度与步频相关系数: {corr.loc['speed', 'step_frequency']:.4f}")
    print(f"速度与心率相关系数: {corr.loc['speed', 'heart_rate']:.4f}")
    print(f"步频与心率相关系数: {corr.loc['step_frequency', 'heart_rate']:.4f}")
    
    # 6. 数据统计信息
    print("\n数据统计:")
    print(df.describe())
    
    # 绘制数据分布散点图
    plt.figure(figsize=(15, 10))
    
    # 速度vs心率
    plt.subplot(2, 2, 1)
    plt.scatter(df['speed'], df['heart_rate'], alpha=0.3)
    plt.xlabel('速度 (m/s)')
    plt.ylabel('心率 (bpm)')
    plt.title('速度与心率关系')
    plt.grid(True)
    
    # 步频vs心率
    plt.subplot(2, 2, 2)
    plt.scatter(df['step_frequency'], df['heart_rate'], alpha=0.3)
    plt.xlabel('步频 (步/分钟)')
    plt.ylabel('心率 (bpm)')
    plt.title('步频与心率关系')
    plt.grid(True)
    
    # 速度vs步频 (颜色为心率)
    plt.subplot(2, 2, 3)
    plt.scatter(df['speed'], df['step_frequency'], c=df['heart_rate'], 
                alpha=0.5, cmap='viridis')
    plt.colorbar(label='心率 (bpm)')
    plt.xlabel('速度 (m/s)')
    plt.ylabel('步频 (步/分钟)')
    plt.title('速度、步频与心率三者关系')
    plt.grid(True)
    
    # 心率分布直方图
    plt.subplot(2, 2, 4)
    plt.hist(df['heart_rate'], bins=30)
    plt.xlabel('心率 (bpm)')
    plt.ylabel('频率')
    plt.title('心率分布')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/improved_heart_rate_data_distribution.png')
    print("数据分布图已保存到: data/improved_heart_rate_data_distribution.png")
    
    return df

# 生成2000条数据
df_new = generate_improved_heart_rate_data(5000)

# 保存到CSV文件
output_path = 'data/improved_heart_rate_data.csv'
df_new.to_csv(output_path, index=False)
print(f"\n数据已保存到: {output_path}")
