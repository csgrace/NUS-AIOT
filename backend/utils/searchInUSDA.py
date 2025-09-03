import requests
import json

# --- 配置您的 API 密钥 ---
# 请务必替换为您的实际 API 密钥。您可以从 data.gov 注册获取。
# 对于测试，您也可以使用 DEMO_KEY，但其有严格的速率限制。
API_KEY = "mY212lBizpYLmnRd8FEbSj9t5LTIH05syovByKUO" # <--- 在这里替换成您的 data.gov API 密钥

BASE_URL = "https://api.nal.usda.gov/fdc/v1"

def search_food(query):
    """
    通过食物名称搜索食物并返回匹配的食物列表 (包含 FDC ID)。
    """
    search_url = f"{BASE_URL}/foods/search"
    params = {
        "api_key": API_KEY,
        "query": query
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("foods", [])
    except requests.exceptions.RequestException as e:
        print(f"搜索食物时发生错误: {e}")
        return []  # 修改为返回空列表

def get_food_details(fdc_id):
    """
    通过 FDC ID 获取食物的详细营养信息。
    """
    details_url = f"{BASE_URL}/food/{fdc_id}"
    params = {
        "api_key": API_KEY
    }

    try:
        response = requests.get(details_url, params=params)
        response.raise_for_status() # 如果请求失败，抛出 HTTPError
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"获取食物详情时发生错误: {e}")
        return None

def calculate_calories(food_details):
    """
    从食物详情中提取并计算 100 克食物的卡路里。
    优先 Energy/KCAL，其次 Energy/kcal，再次 Energy/kJ（换算为 kcal）。
    """
    if not food_details or "foodNutrients" not in food_details:
        return None

    # 优先查找 KCAL 或 kcal
    for nutrient in food_details["foodNutrients"]:
        nutrient_info = nutrient.get("nutrient", {})
        name = nutrient_info.get("name")
        unit = nutrient_info.get("unitName")
        amount = nutrient.get("amount")
        if name == "Energy" and unit in ["KCAL", "kcal"]:
            return amount
    # 若无，再查找 kJ 并换算
    for nutrient in food_details["foodNutrients"]:
        nutrient_info = nutrient.get("nutrient", {})
        name = nutrient_info.get("name")
        unit = nutrient_info.get("unitName")
        amount = nutrient.get("amount")
        if name == "Energy" and unit == "kJ":
            return amount / 4.184 if amount is not None else None
    return None

def get_calories_by_food_name(food_name):
    """
    传入食物名称，优先选择第一个 'Foundation' 数据类型的食物，
    若不存在则返回结果中第一个食物的主要营养素（每 100g）。
    返回 dict: { "Protein": float, "Fat": float, "Carbohydrate": float, "Fiber": float, "Energy": float }
    """
    foods = search_food(food_name)
    if not foods:
        return None

    selected_food = None
    for food in foods:
        if food.get('dataType') == 'Foundation':
            selected_food = food
            break
    if not selected_food and len(foods) > 0:
        selected_food = foods[0]
    if not selected_food:
        return None

    fdc_id = selected_food.get("fdcId")
    food_details = get_food_details(fdc_id)
    if not food_details or "foodNutrients" not in food_details:
        return None

    result = {
        "Protein": None,
        "Fat": None,
        "Carbohydrate": None,
        "Fiber": None,
        "Energy": None
    }
    for nutrient in food_details["foodNutrients"]:
        name = nutrient.get("nutrient", {}).get("name")
        unit = nutrient.get("nutrient", {}).get("unitName")
        amount = nutrient.get("amount")
        if name == "Protein":
            result["Protein"] = amount
        elif name == "Total lipid (fat)":
            result["Fat"] = amount
        elif name == "Carbohydrate, by difference":
            result["Carbohydrate"] = amount
        elif name == "Fiber, total dietary":
            result["Fiber"] = amount
        elif name == "Energy":
            if unit in ["KCAL", "kcal"]:
                result["Energy"] = amount
            elif unit == "kJ" and amount is not None:
                result["Energy"] = amount / 4.184
    return result

def main():
    print("--- FoodData Central 食物营养查询 ---")
    print("请注意：对于频繁的请求，请确保使用您的 data.gov API 密钥，而不是 DEMO_KEY。")

    while True:
        food_name = input("\n请输入您想查询的食物名称 (输入 'exit' 退出): ").strip()
        if food_name.lower() == 'exit':
            break

        print(f"正在搜索 '{food_name}'...")
        foods = search_food(food_name)

        if foods:
            # 找到第一个 dataType 为 "Foundation" 的食物
            foundation_food = None
            for food in foods:
                if food.get('dataType') == 'Foundation':
                    foundation_food = food
                    break
            
            if foundation_food:
                fdc_id = foundation_food.get("fdcId")
                description = foundation_food.get("description")
                print(f"\n已自动选择第一个 'Foundation' 数据类型的食物:")
                print(f"FDC ID: {fdc_id}, 描述: {description}")

                print(f"\n正在获取 FDC ID {fdc_id} 的详细信息...")
                food_details = get_food_details(fdc_id)
                
                if food_details:
                    print("\n部分主要营养素 (每 100 克):")
                    for nutrient in food_details.get("foodNutrients", []):
                        name = nutrient.get("nutrient", {}).get("name")
                        amount = nutrient.get("amount")
                        unit = nutrient.get("nutrient", {}).get("unitName")
                        if name in ["Protein", "Total lipid (fat)", "Carbohydrate, by difference", "Fiber, total dietary","Energy"]:
                            print(f"- {name}: {amount:.2f} {unit}")
                else:
                    print("未能获取食物详情。")
            else:
                print("未能找到 'Foundation' 数据类型的食物。")
    
            # 仍然保留了让用户手动选择的选项，以防没有 Foundation 类型的数据或者用户想看其他类型
            print("\n--- 其他搜索结果 (可选手动选择) ---")
            for i, food in enumerate(foods):
                print(f"{i + 1}. FDC ID: {food.get('fdcId')}, 描述: {food.get('description')}, 数据类型: {food.get('dataType')}")
            
            try:
                choice = int(input("请选择一个食物的序号以查看详情 (或输入 0 跳过并继续搜索): "))
                if choice == 0:
                    continue
                
                if 1 <= choice <= len(foods):
                    selected_food = foods[choice - 1]
                    fdc_id = selected_food.get("fdcId")
                    
                    print(f"\n正在获取 FDC ID {fdc_id} 的详细信息...")
                    food_details = get_food_details(fdc_id)
                    
                    if food_details:
                        print("\n部分主要营养素 (每 100 克):")
                        for nutrient in food_details.get("foodNutrients", []):
                            name = nutrient.get("nutrient", {}).get("name")
                            amount = nutrient.get("amount")
                            unit = nutrient.get("nutrient", {}).get("unitName")
                            if name in ["Protein", "Total lipid (fat)", "Carbohydrate, by difference", "Fiber, total dietary","Energy"]:
                                print(f"- {name}: {amount:.2f} {unit}")
                    else:
                        print("未能获取食物详情。")
                else:
                    print("无效的选择，请重试。")
            except ValueError:
                print("无效输入，请输入数字。")
        else:
            print(f"未能找到与 '{food_name}' 匹配的食物。")

if __name__ == "__main__":
    #main()
    while True:
        query = input("请输入要查询的食物名称 (输入 'exit' 退出): ").strip()
        if query.lower() == 'exit':
            break
        calories = get_calories_by_food_name(query)
        if calories is not None:
            print(f"{query} 每 100 克的卡路里含量为: {calories:.2f} kcal")
        else:
            print(f"未能找到 {query} 的卡路里信息。")