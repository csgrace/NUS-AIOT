import pandas as pd
import psycopg2
from datetime import date, timedelta

def get_training_data():
    """
    从PostgreSQL数据库中提取、合并和处理昨天的跑步训练数据。

    该函数会执行以下操作：
    1.  从 'workout_records' 表中获取所有类型为 'running' 的运动记录。
    2.  对于每一段跑步记录，根据其开始和结束时间：
        - 从 'gps_data' 表提取 'time' 和 'speed'。
        - 从 'band_steps' 表提取 'time' 和 'step_frenquency'。
        - 从 'band_heart_rate' 表提取 'time' 和 'heart_rate'。
    3.  由于三个数据源的采样时间点不同，我们以GPS数据的时间戳为基准，
        使用 `pd.merge_asof` 方法找到另外两个数据源在最邻近时间点的数据进行合并。
        这是一种高效的按时间对齐不完全匹配时间序列数据的方法。
    4.  合并所有跑步时段的数据。
    5.  删除时间列，并按照 ['step_frequency', 'speed', 'heart_rate'] 的顺序重新排列列。
    6.  返回处理完成的 DataFrame。

    Returns:
        pandas.DataFrame: 包含所有合并和处理后的训练数据。
                          列顺序为 ['step_frenquency', 'speed', 'heart_rate']。
    """
    # 1. 设置数据库连接参数
    # !!! 请根据你的实际情况修改这里的数据库连接信息 !!!
    db_params = {
        'dbname': 'aiot',
        'user': 'group4',
        'password': 'groupgroup4',
        'host': '127.0.0.1',
        'port': '5432'
    }

    # 初始化一个空列表，用于存放每个运动分段处理后的数据
    all_workouts_data = []

    try:
        # 建立数据库连接
        conn = psycopg2.connect(**db_params)
        
        # 2. 从 workout_records 表中提取跑步记录
        query_workouts = f"""
        SELECT start_time, end_time 
        FROM workout_records 
        WHERE exercise_type = 'running';
        """
        workout_sessions = pd.read_sql_query(query_workouts, conn)

        print(f"查询到昨天共有 {len(workout_sessions)} 段跑步记录。")

        # 3. 循环处理每一段跑步记录
        for index, session in workout_sessions.iterrows():
            start_time = session['start_time']
            end_time = session['end_time']
            
            print(f"\n正在处理分段 {index + 1}: {start_time} -> {end_time}")

            # 4. 在指定时间范围内，分别从三个表中提取数据
            query_gps = f"SELECT time, speed FROM gps_data WHERE time BETWEEN '{start_time}' AND '{end_time}';"
            query_steps = f"SELECT time, step_frequency FROM band_steps WHERE time BETWEEN '{start_time}' AND '{end_time}';"
            query_hr = f"SELECT time, heart_rate FROM band_heart_rate WHERE time BETWEEN '{start_time}' AND '{end_time}';"

            df_gps = pd.read_sql_query(query_gps, conn)
            df_steps = pd.read_sql_query(query_steps, conn)
            df_hr = pd.read_sql_query(query_hr, conn)

            #将速度从 m/s 转换为 km/h
            if not df_gps.empty:
                df_gps['speed'] = df_gps['speed'] * 3.6

            # 检查是否提取到了数据
            if df_gps.empty:
                print(f"警告：分段 {index + 1} 没有提取到GPS数据，将跳过此分段。")
                continue

            # 5. 数据预处理：将 'time' 列转换为 datetime 对象并排序，这是 merge_asof 的前提
            for df in [df_gps, df_steps, df_hr]:
                # 如果 'time' 列不是 datetime 类型，则转换
                if df['time'].dtype != 'datetime64[ns]':
                    df['time'] = pd.to_datetime(df['time'])
                df.sort_values('time', inplace=True)

            # 6. 使用 merge_asof 进行近邻合并
            # 以 df_gps 为基准，因为它通常是训练数据中最重要的时间序列
            # 'direction=nearest' 会寻找最近的时间点
            merged_df = pd.merge_asof(
                df_gps, 
                df_steps, 
                on='time', 
                direction='nearest'
            )
            merged_df = pd.merge_asof(
                merged_df, 
                df_hr, 
                on='time', 
                direction='nearest'
            )

            # 删除可能存在的空值行（如果某个时间点附近完全没有其他数据）
            merged_df.dropna(inplace=True)

            all_workouts_data.append(merged_df)

    except (Exception, psycopg2.Error) as error:
        print(f"数据库连接或查询时出错: {error}")
        return None
    finally:
        # 关闭数据库连接
        if 'conn' in locals() and conn is not None:
            conn.close()
            print("\n数据库连接已关闭。")

    # 7. 合并所有分段的数据
    if not all_workouts_data:
        print("未能处理任何数据，返回空结果。")
        return pd.DataFrame()
        
    final_df = pd.concat(all_workouts_data, ignore_index=True)

    # 8. 删去time列，并按照指定顺序重排
    final_df.drop('time', axis=1, inplace=True)
    final_df = final_df[['step_frequency', 'speed', 'heart_rate']]

    print("\n所有数据处理完成！")
    return final_df

# # --- 执行函数并打印结果 ---
# if __name__ == '__main__':
#     training_data = get_training_data()
#     if training_data is not None and not training_data.empty:
#         print("\n最终提取和处理的数据预览 (前10行):")
#         print(training_data.head(10))
#         print(f"\n总共合并了 {len(training_data)} 条数据记录。")
#         # 自动保存为csv
#         csv_path = 'training_data_output.csv'
#         training_data.to_csv(csv_path, index=False)
#         print(f"\n数据已保存为 {csv_path}")