import pandas as pd

# 读取数据
df = pd.read_csv('backend/datasets/running/merged_data_efficient_no_time.csv')

# 将 speed 列从 km/h 转换为 km/s
df['speed'] = df['speed'] / 3600

# 保存为新的 CSV 文件
df.to_csv('merged_data_efficient_no_time_km_s.csv', index=False)