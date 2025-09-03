import pandas as pd

# 1. 加载所有CSV文件
try:
    food_df = pd.read_csv('food.csv')
    nutrient_df = pd.read_csv('nutrient.csv')
    food_nutrient_df = pd.read_csv('food_nutrient.csv')
    food_nutrient_conversion_factor_df = pd.read_csv('food_nutrient_conversion_factor.csv')
    food_calorie_conversion_factor_df = pd.read_csv('food_calorie_conversion_factor.csv')
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure all CSV files are in the same directory.")
    exit()

# 2. 识别蛋白质、脂肪和碳水化合物的nutrient_id
# 根据nutrient.csv的截图，我们可以大致判断出：
# 蛋白质 (Protein): id = 1003
# 总脂肪 (Total lipid (fat)): id = 1004
# 碳水化合物 (Carbohydrate, by difference): id = 1005

# 查找这些营养素的ID，以防ID在不同数据集中有所变动
protein_id = nutrient_df[nutrient_df['name'].str.contains('Protein', case=False)]['id'].iloc[0]
fat_id = nutrient_df[nutrient_df['name'].str.contains('Total lipid', case=False)]['id'].iloc[0]
carbohydrate_id = nutrient_df[nutrient_df['name'].str.contains('Carbohydrate, by difference', case=False)]['id'].iloc[0]

print(f"Protein ID: {protein_id}")
print(f"Fat ID: {fat_id}")
print(f"Carbohydrate ID: {carbohydrate_id}")

# 3. 合并数据以获取每种食物的蛋白质、脂肪和碳水化合物含量
# 筛选出我们关心的营养素
relevant_nutrients_df = food_nutrient_df[
    food_nutrient_df['nutrient_id'].isin([protein_id, fat_id, carbohydrate_id])
].copy() # Using .copy() to avoid SettingWithCopyWarning


print("\n--- macro_nutrients_df (filtered food_nutrient data) ---")
print(relevant_nutrients_df.head())
print(relevant_nutrients_df.columns)

# 将 'amount' 列转换为数值类型，如果其中有非数值，将它们设为 NaN
relevant_nutrients_df['amount'] = pd.to_numeric(relevant_nutrients_df['amount'], errors='coerce')
relevant_nutrients_df.dropna(subset=['amount'], inplace=True) # 删除NaN值

print("\n--- 'amount' 列转换为数值类型，如果其中有非数值，将它们设为 NaN ---")
print(relevant_nutrients_df.head())
print(relevant_nutrients_df.columns)

# 将营养素数据透视，使每行是一个fdc_id，列是蛋白质、脂肪和碳水化合物的含量
pivot_nutrients_df = relevant_nutrients_df.pivot_table(
    index='fdc_id',
    columns='nutrient_id',
    values='amount',
    fill_value=0 # 如果某种营养素不存在，则填充0
).reset_index()

# 重命名列以便于识别
pivot_nutrients_df.rename(columns={
    protein_id: 'protein_g',
    fat_id: 'fat_g',
    carbohydrate_id: 'carbohydrate_g'
}, inplace=True)

print("\n--- pivot_nutrients_df (protein, fat, carb amounts per food) ---")
print(pivot_nutrients_df.head())
print(f"Max protein_g: {pivot_nutrients_df['protein_g'].max()}")
print(f"Max fat_g: {pivot_nutrients_df['fat_g'].max()}")
print(f"Max carbohydrate_g: {pivot_nutrients_df['carbohydrate_g'].max()}")

# 4. 合并食物描述
food_calories_df = pd.merge(food_df[['fdc_id', 'description']],
                            pivot_nutrients_df,
                            on='fdc_id',
                            how='left')

print("\n--- 合并食物描述 ---")
print(food_calories_df.head())

# 填充可能由于没有对应营养素而产生的NaN为0
food_calories_df[['protein_g', 'fat_g', 'carbohydrate_g']] = food_calories_df[['protein_g', 'fat_g', 'carbohydrate_g']].fillna(0)

print("\n--- 填充可能由于没有对应营养素而产生的NaN为0 ---")
print(food_calories_df.head())


# 5. 合并卡路里转换系数
# food_nutrient_conversion_factor_df 包含了 fdc_id 和 food_nutrient_conversion_factor_id
# food_calorie_conversion_factor_df 包含了 food_nutrient_conversion_factor_id 和卡路里系数
# 首先合并这两个表
conversion_factors_merged_df = pd.merge(food_nutrient_conversion_factor_df,
                                        food_calorie_conversion_factor_df,
                                        left_on='id', # 这里的'id'是food_nutrient_conversion_factor_id
                                        right_on='food_nutrient_conversion_factor_id',
                                        how='left')

# 转换转换系数列为数值类型
conversion_factors_merged_df['protein_value'] = pd.to_numeric(conversion_factors_merged_df['protein_value'], errors='coerce')
conversion_factors_merged_df['fat_value'] = pd.to_numeric(conversion_factors_merged_df['fat_value'], errors='coerce')
conversion_factors_merged_df['carbohydrate_value'] = pd.to_numeric(conversion_factors_merged_df['carbohydrate_value'], errors='coerce')


# 合并食物数据和转换系数
food_calories_df = pd.merge(food_calories_df,
                            conversion_factors_merged_df[['fdc_id', 'protein_value', 'fat_value', 'carbohydrate_value']],
                            on='fdc_id',
                            how='left')

# 填充可能由于没有对应转换系数而产生的NaN。如果缺失，我们将使用Atwater通用系数作为默认值
# 蛋白质: 4 kcal/g, 脂肪: 9 kcal/g, 碳水化合物: 4 kcal/g
food_calories_df['protein_value'].fillna(4, inplace=True)
food_calories_df['fat_value'].fillna(9, inplace=True)
food_calories_df['carbohydrate_value'].fillna(4, inplace=True)

# 6. 计算每100g食物的卡路里
# 假设 food_nutrient.csv 中的 'amount' 是指每100克食物中的含量，或者我们需要将其标准化到100g。
# 根据通常的食物营养数据，'amount'列通常代表每100g食物中的含量，或者可以根据'unit_name'推断。
# 在这里，我们假设 'amount' 已经是每100g的含量。如果不是，需要额外的数据或逻辑进行转换。

food_calories_df['calories_per_100g'] = (
    (food_calories_df['protein_g'] * food_calories_df['protein_value']) +
    (food_calories_df['fat_g'] * food_calories_df['fat_value']) +
    (food_calories_df['carbohydrate_g'] * food_calories_df['carbohydrate_value'])
)

# 7. 选择并排序最终结果
final_calories_report = food_calories_df[['fdc_id', 'description', 'protein_g', 'fat_g', 'carbohydrate_g', 'calories_per_100g']].copy()
final_calories_report.sort_values(by='description', inplace=True)

# 8. 打印或保存结果
print("\n每100g食物的卡路里报告:")
print(final_calories_report.head(20)) # 打印前20行作为示例

# 保存结果到新的CSV文件
output_filename = 'food_calories_report.csv'
final_calories_report.to_csv(output_filename, index=False)
print(f"\n报告已保存至 '{output_filename}'")