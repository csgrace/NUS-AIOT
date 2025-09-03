import pandas as pd

# 读取CSV文件
file_path = '/Users/izumedonabe/CS/NUS-Summer-Workshop-AIOT/backend/datasets/running/merged_data_efficient_no_diff.csv'
output_path = '/Users/izumedonabe/CS/NUS-Summer-Workshop-AIOT/backend/datasets/running/merged_data_efficient_no_time.csv'

# 读取数据
df = pd.read_csv(file_path)

# 删除time列
df = df.drop(columns=['time'])

# 保存结果
df.to_csv(output_path, index=False)

print(f"已成功从文件中移除time列并保存到 {output_path}")
