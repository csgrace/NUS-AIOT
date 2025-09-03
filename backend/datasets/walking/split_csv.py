import os
import pandas as pd

# 输入和输出目录
input_dir = os.path.join(os.path.dirname(__file__), 'original_csv')
output_dir = os.path.join(os.path.dirname(__file__), 'splited')

# 切片大小
chunk_size = 100

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历所有csv文件
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        # 读取csv并移除time列
        df = pd.read_csv(file_path)
        if 'time' in df.columns:
            df = df.drop(columns=['time'])
        # 创建该文件的输出子目录
        name_wo_ext = os.path.splitext(filename)[0]
        sub_output_dir = os.path.join(output_dir, name_wo_ext)
        os.makedirs(sub_output_dir, exist_ok=True)
        # 切片并保存
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            if not chunk.empty:
                chunk_file = os.path.join(sub_output_dir, f'part_{i//chunk_size}.csv')
                chunk.to_csv(chunk_file, index=False)
print('批量切片完成！')
