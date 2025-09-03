from utils import logger
from utils import searchInUSDA
logger = logger.get_logger()
import requests
import base64
import time
import asyncio
import psycopg2

FOOD_API_URL = "http://192.168.137.85:3000/api/v1/prediction/cf9ecc4a-a8e3-4b67-87be-6168b00d06cb"

def deep_find_key(obj, target_key):
    """
    深度递归查找目标 key 的值，支持 dict、list、字符串中的 dict。
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target_key:
                return v
            # 递归查找
            found = deep_find_key(v, target_key)
            if found is not None:
                return found
        return None
    elif isinstance(obj, list):
        for item in obj:
            found = deep_find_key(item, target_key)
            if found is not None:
                return found
        return None
    elif isinstance(obj, str):
        # 尝试将字符串解析为 dict
        try:
            import ast
            parsed = ast.literal_eval(obj)
            return deep_find_key(parsed, target_key)
        except Exception:
            return None
    else:
        return None

class FoodHandler:
    def __init__(self):
        #First load yolo model
        from models.yolov11n.infer import YOLOInfer
        self.yolo_infer = YOLOInfer(model_path='backend/models/yolov11n/best.pt')
        

    def recognize_food(self, image_path):
        """
        Recognizes food items in the given image.
        
        :param image_path: Path to the image file.
        :return: List of recognized food items.
        """
        # First use yolo to detect food items
        yolo_results = self.yolo_infer.infer(image_path, save=False)
        food_items = []

        if yolo_results is None or len(yolo_results) == 0:
            logger.warning("No food items detected by YOLO, trying GPT for recognition.")
            try:
                food_detail = self.ask_GPT(image_path)
                if food_detail:
                    food_items.append(food_detail)
                    logger.info(f"GPT provided food details: {food_detail}")
                else:
                    logger.error("GPT returned no valid food details.")
            except Exception as e:
                logger.error(f"Error while asking GPT for food details: {str(e)}")
                food_items.append("Unknown food item")
            return food_items[0] if food_items else "Unknown food item"

        for result in yolo_results:
            if result['confidence'] > 0.6:
                food_items.append(result['class_name'])
                logger.info(f"Detected food item: {result['class_name']} with confidence {result['confidence']:.2f}")
            else:
                logger.warning(f"Low confidence for {result['class_name']}: {result['confidence']:.2f}, trying GPT for more details.")
                  # Call GPT to get more details if confidence is low
                try:
                    food_detail = self.ask_GPT(image_path)
                    if food_detail:
                        food_items.append(food_detail)
                        logger.info(f"GPT provided additional details: {food_detail}")
                except Exception as e:
                    logger.error(f"Error while asking GPT for food details: {str(e)}")
                    food_items.append(f"Unknown food item")
        return food_items[0]
    
    def ask_GPT(self, image_path, max_total_requests=5, retry_interval=1):
        """
        默认询问两次，若结果一致直接输出，否则询问第三次，取相同的两次输出。
        若有失败请求则重试，总请求次数不超过五次。
        """
        def single_request():
            # 读取图片并编码为 base64
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            API_URL = FOOD_API_URL
            payload = {
                "uploads": [
                    {
                        "type": "file",
                        "name": "food.jpg",
                        "data": f"data:image/jpeg;base64,{img_base64}",
                        "mime": "image/jpeg"
                    }
                ]
            }
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code != 200:
                    logger.error(f"Error in API request: {response.status_code} - {response.text}")
                    return None
                logger.info("Received response from food recognition API")
                resp_json = response.json()
                food_value = deep_find_key(resp_json, 'food')
                if food_value is not None:
                    return food_value
                logger.warning("No 'food' field found, retrying...")
                return None
            except Exception as e:
                logger.error(f"GPT request error: {str(e)}")
                return None

        results = []
        attempts = 0
        while len(results) < 2 and attempts < max_total_requests:
            result = single_request()
            if result is not None:
                results.append(result)
            else:
                time.sleep(retry_interval)
            attempts += 1

        if len(results) < 2:
            # 如果两次都失败，继续重试直到有两个结果或达到最大次数
            while len(results) < 2 and attempts < max_total_requests:
                result = single_request()
                if result is not None:
                    results.append(result)
                else:
                    time.sleep(retry_interval)
                attempts += 1

        if len(results) == 0:
            raise Exception("Failed to get valid response from the food recognition API after retries")

        if results[0] == results[1]:
            return results[0]

        # 如果前两次结果不同，第三次请求
        third_result = None
        while attempts < max_total_requests:
            third_result = single_request()
            attempts += 1
            if third_result is not None:
                break
            else:
                time.sleep(retry_interval)

        if third_result is None:
            # 如果第三次也失败，返回前两次中任意一个
            logger.warning("Third GPT request failed, returning the first valid result")
            return results[0]

        # 统计出现次数最多的结果
        all_results = results + [third_result]
        from collections import Counter
        most_common = Counter(all_results).most_common(1)[0][0]
        return most_common

async def async_ask(path, photo_id):
    handler = FoodHandler()
    food_items = handler.recognize_food(path)
    print("Recognized food items:", food_items)
    #query the calories of the recognized food item
    query = searchInUSDA.get_calories_by_food_name(food_items)
    # 按照指定格式解析并输出
    if query and isinstance(query, dict):
        result = {
            'Protein': query.get('Protein'),
            'Fat': query.get('Fat'),
            'Carbohydrate': query.get('Carbohydrate'),
            'Fiber': query.get('Fiber'),
            'Energy': query.get('Energy')
        }
        print(result)
        # 写入数据库
        db_params = {
            'dbname': 'aiot',
            'user': 'group4',
            'password': 'groupgroup4',
            'host': 'localhost',
            'port': '5432'
        }
        try:
            conn = psycopg2.connect(**db_params)
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE food_photos
                    SET 
                        food_name = %s,
                        calorie = %s,
                        protein = %s,
                        fat = %s,
                        carbohydrate = %s,
                        fiber = %s
                    WHERE id = %s;
                """, (
                    food_items,
                    result['Energy'],
                    result['Protein'],
                    result['Fat'],
                    result['Carbohydrate'],
                    result['Fiber'],
                    photo_id
                ))
                conn.commit()
            conn.close()
            print("数据库已更新")
        except Exception as e:
            print(f"数据库写入失败: {e}")
    else:
        print("未能获取营养信息")
    

if __name__ == "__main__":
    asyncio.run(async_ask("/Users/izumedonabe/CS/NUS-Summer-Workshop-AIOT/test_photo/微信图片_20250714110819_159.jpg", 1))







