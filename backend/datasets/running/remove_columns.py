import pandas as pd

# 读取原始CSV文件
file_path = '/Users/izumedonabe/CS/NUS-Summer-Workshop-AIOT/backend/datasets/running/merged_data_efficient_filtered.csv'
output_path = '/Users/izumedonabe/CS/NUS-Summer-Workshop-AIOT/backend/datasets/running/merged_data_efficient_filtered_new.csv'

# 读取CSV文件
df = pd.read_csv(file_path)

# 删除指定的两列
columns_to_drop = ['steps_time_diff_seconds', 'heart_time_diff_seconds']
df = df.drop(columns=columns_to_drop)

# 保存到新文件
df.to_csv(output_path, index=False)

print(f"已成功删除列并保存到 {output_path}")
