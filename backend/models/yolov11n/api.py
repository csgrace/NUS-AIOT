import os
import logging
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 全局变量存储模型
model = None

# 食物类别映射
FOOD_CLASSES = {
    1: 'rice',
    2: 'eels on rice',
    3: 'pilaf',
    4: 'chicken-\'n\'-egg on rice',
    5: 'pork cutlet on rice',
    6: 'beef curry',
    7: 'sushi',
    8: 'chicken rice',
    9: 'fried rice',
    10: 'tempura bowl',
    11: 'bibimbap',
    12: 'toast',
    13: 'croissant',
    14: 'roll bread',
    15: 'raisin bread',
    16: 'chip butty',
    17: 'hamburger',
    18: 'pizza',
    19: 'sandwiches',
    20: 'udon noodle',
    21: 'tempura udon',
    22: 'soba noodle',
    23: 'ramen noodle',
    24: 'beef noodle',
    25: 'tensin noodle',
    26: 'fried noodle',
    27: 'spaghetti',
    28: 'Japanese-style pancake',
    29: 'takoyaki',
    30: 'gratin',
    31: 'sauteed vegetables',
    32: 'croquette',
    33: 'grilled eggplant',
    34: 'sauteed spinach',
    35: 'vegetable tempura',
    36: 'miso soup',
    37: 'potage',
    38: 'sausage',
    39: 'oden',
    40: 'omelet',
    41: 'ganmodoki',
    42: 'jiaozi',
    43: 'stew',
    44: 'teriyaki grilled fish',
    45: 'fried fish',
    46: 'grilled salmon',
    47: 'salmon meuniere',
    48: 'sashimi',
    49: 'grilled pacific saury',
    50: 'sukiyaki',
    51: 'sweet and sour pork',
    52: 'lightly roasted fish',
    53: 'steamed egg hotchpotch',
    54: 'tempura',
    55: 'fried chicken',
    56: 'sirloin cutlet',
    57: 'nanbanzuke',
    58: 'boiled fish',
    59: 'seasoned beef with potatoes',
    60: 'hambarg steak',
    61: 'beef steak',
    62: 'dried fish',
    63: 'ginger pork saute',
    64: 'spicy chili-flavored tofu',
    65: 'yakitori',
    66: 'cabbage roll',
    67: 'rolled omelet',
    68: 'egg sunny-side up',
    69: 'fermented soybeans',
    70: 'cold tofu',
    71: 'egg roll',
    72: 'chilled noodle',
    73: 'stir-fried beef and peppers',
    74: 'simmered pork',
    75: 'boiled chicken and vegetables',
    76: 'sashimi bowl',
    77: 'sushi bowl',
    78: 'fish-shaped pancake with bean jam',
    79: 'shrimp with chill source',
    80: 'roast chicken',
    81: 'steamed meat dumpling',
    82: 'omelet with fried rice',
    83: 'cutlet curry',
    84: 'spaghetti meat sauce',
    85: 'fried shrimp',
    86: 'potato salad',
    87: 'green salad',
    88: 'macaroni salad',
    89: 'Japanese tofu and vegetable chowder',
    90: 'pork miso soup',
    91: 'chinese soup',
    92: 'beef bowl',
    93: 'kinpira-style sauteed burdock',
    94: 'rice ball',
    95: 'pizza toast',
    96: 'dipping noodles',
    97: 'hot dog',
    98: 'french fries',
    99: 'mixed rice',
    100: 'goya chanpuru'
}

def load_model():
    """加载YOLO模型"""
    global model
    try:
        logger.info("正在加载YOLO模型...")
        model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        model = YOLO(model_path)
        logger.info("YOLO模型加载成功!")
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return False

def decode_image(image_data):
    """解码base64图像数据"""
    try:
        # 移除数据URL前缀
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # 添加padding
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += '=' * (4 - missing_padding)
        
        # 解码并创建图像对象
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        return image
        
    except Exception as e:
        logger.error(f"图像解码失败: {str(e)}")
        raise e

def detect_food_in_image(image):
    """在图像中检测食物"""
    try:
        if model is None:
            raise Exception("Model not loaded")
        
        # 转换为numpy数组并检测
        image_array = np.array(image)
        results = model(image_array)
        
        # 解析检测结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # 只保留食物类别
                    if class_id in FOOD_CLASSES:
                        detection = {
                            'class_name': FOOD_CLASSES[class_id],
                            'confidence': confidence,
                            'bbox': {
                                'x1': float(x1),
                                'y1': float(y1),
                                'x2': float(x2),
                                'y2': float(y2)
                            }
                        }
                        detections.append(detection)
        
        return detections
        
    except Exception as e:
        logger.error(f"食物检测失败: {str(e)}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

@app.route('/detect_food', methods=['POST'])
def detect_food():
    """食物检测端点"""
    try:
        # 检查模型状态
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
        
        # 获取图像数据
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No image data provided'
            }), 400
        
        # 解码图像
        image = decode_image(data['image'])
        
        # 检测食物
        detections = detect_food_in_image(image)
        
        # 返回结果
        return jsonify({
            'status': 'success',
            'detections_count': len(detections),
            'detections': detections
        })
        
    except Exception as e:
        logger.error(f"检测失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("启动Food Detection API...")
    
    # 加载模型
    if not load_model():
        logger.error("模型加载失败")
    
    # 启动服务
    app.run(host='127.0.0.1', port=2333, debug=False)
