from flask import Flask, jsonify, request, render_template
from flask import send_file, request
from flasgger import Swagger, swag_from
import psycopg2
import psycopg2.extras
import os
import base64
import re
import sys
import threading
from psycopg2.extras import RealDictCursor
import json
import time
from flask_cors import CORS, cross_origin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FoodHandler import FoodHandler
from FoodHandler import async_ask

# 修改导入和实例化方式，避免命名冲突
from SportDataHandler import SportDataHandler as SportDataHandlerClass

from utils import searchInUSDA

import asyncio
from datetime import datetime

FoodHandler = FoodHandler()  # 实例化 FoodHandler
sport_handler = SportDataHandlerClass()  # 使用不同的变量名实例化 SportDataHandler
# --- 配置 ---
app = Flask(__name__)
swagger = Swagger(app)  # 初始化 Flasgger
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'food_photos')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 从环境变量或直接配置数据库连接信息
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "aiot")  # 替换为你的数据库名
DB_USER = os.getenv("DB_USER", "group4")  # 替换为你的数据库用户名
DB_PASS = os.getenv("DB_PASS", "groupgroup4")  # 替换为你的数据库密码


def get_db_connection():
    """建立数据库连接"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            client_encoding='utf8'
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to database: {e}")
        return None


# --- 新增：主页路由，用于渲染 index.html ---
@app.route('/')  # <-- 2. 定义根路径路由
def index():
    """
    渲染主页 (index.html)。
    当用户访问应用的根 URL (例如 http://树莓派IP:5000/) 时，
    此函数会被调用，并返回 templates 文件夹中的 index.html 文件。
    """
    return render_template('index.html')  # <-- 3. 使用 render_template 返回 index.html


# --- API 端点 ---
@app.route('/getSteps', methods=['GET'])
@swag_from('openapi.yaml')
def get_steps():
    """
    Retrieves the latest step count.
    ---
    responses:
      200:
        description: A dictionary containing the latest step count.
        schema:
          type: object
          properties:
            steps:
              type: integer
              description: Current step count.
      500:
        description: Internal server error.
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Modified to use 'band_steps' table and 'step_count' column
        cur.execute("""
                    SELECT step_count, time
                    FROM band_steps
                    WHERE date (time) = CURRENT_DATE
                    ORDER BY time DESC
                        LIMIT 1
                    """)
        data = cur.fetchone()
        if data:
            return jsonify({
                "steps": data['step_count']
            })
        else:
            return jsonify({"steps": 0})
    except Exception as e:
        print(f"Error fetching steps data: {e}")
        return jsonify({"error": "Failed to retrieve steps data"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/setStepTarget', methods=['POST'])
@swag_from('openapi.yaml')
def set_step_target():
    """
    Sets the daily step target for the user.
    ---
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required:
            - target
          properties:
            target:
              type: integer
              description: The daily step target.
    responses:
      200:
        description: Step target set successfully.
      400:
        description: Invalid input.
      500:
        description: Internal server error.
    """
    data = request.get_json()
    target = data.get('target')

    if not isinstance(target, int) or target <= 0:
        return jsonify({"error": "Invalid step target. Must be a positive integer."}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        cur = conn.cursor()
        # Assuming a 'user_settings' table with 'setting_name', 'setting_value'
        cur.execute(
            "INSERT INTO user_settings (setting_name, setting_value, last_updated) VALUES (%s, %s, %s) "
            "ON CONFLICT (setting_name) DO UPDATE SET setting_value = EXCLUDED.setting_value, last_updated = EXCLUDED.last_updated",
            ('step_target', str(target), datetime.now())
        )
        conn.commit()
        return jsonify({"message": "Step target set successfully"}), 200
    except Exception as e:
        print(f"Error setting step target: {e}")
        return jsonify({"error": "Failed to set step target"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/getHeartRate', methods=['GET'])
@swag_from('openapi.yaml')
def get_heart_rate():
    """
    Retrieves the latest heart rate reading.
    ---
    parameters:
      - name: time
        in: query
        type: string
        format: date-time
        required: false
        description: Optional timestamp to query heart rate around that time.
    responses:
      200:
        description: A dictionary containing the heart rate value and abnormality status.
        schema:
          type: object
          properties:
            value:
              type: integer
              description: Current heart rate in beats per minute (bpm).
            abnormal:
              type: boolean
              description: True if heart rate is abnormal, False otherwise.
      500:
        description: Internal server error.
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Assuming a 'heart_rate_data' table with 'timestamp', 'value', 'abnormal'
        # For simplicity, just get the latest. In a real app, 'time' param would filter.
        cur.execute("""
                    SELECT heart_rate, time
                    FROM band_heart_rate
                    WHERE date (time) = CURRENT_DATE
                    ORDER BY time DESC
                        LIMIT 1
                    """)
        data = cur.fetchone()
        if data:
            return jsonify({
                "value": data['heart_rate']
            })
        else:
            return jsonify({"value": 0, "abnormal": False})  # Default if no data
    except Exception as e:
        print(f"Error fetching heart rate data: {e}")
        return jsonify({"error": "Failed to retrieve heart rate data"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/getHeartRateTimeSlot', methods=['GET'])
@swag_from('openapi.yaml')  # 引用 OpenAPI 规范文件
def get_heart_rate_time_slot():
    """
    Retrieves heart rate records within a specified time slot, with sampling for charting.
    ---
    parameters:
      - name: start
        in: query
        type: string
        format: date-time
        required: true
        description: Start timestamp of the time slot (ISO 8601 format).
      - name: end
        in: query
        type: string
        format: date-time
        required: true
        description: End timestamp of the time slot (ISO 8601 format).
    responses:
      200:
        description: A list of sampled heart rate records.
        schema:
          type: object
          properties:
            records:
              type: array
              items:
                type: object
                properties:
                  time:
                    type: string
                    format: date-time
                  value:
                    type: integer
                  abnormal:
                    type: boolean
      400:
        description: Invalid date format or missing parameters.
      500:
        description: Internal server error.
    """
    start_str = request.args.get('start')
    end_str = request.args.get('end')

    if not start_str or not end_str:
        return jsonify({"error": "Start and end timestamps are required"}), 400

    try:
        # 将ISO格式的字符串转换为Python的datetime对象
        # .replace('Z', '+00:00') 处理 Z 结尾的UTC时间，确保时区正确解析
        start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
    except ValueError:
        return jsonify({"error": "Invalid date format. Use ISO 8601."}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            "SELECT time, heart_rate, predicted_heart_rate FROM band_heart_rate WHERE time BETWEEN %s AND %s ORDER BY time ASC",
            (start_time, end_time)
        )
        all_records = cur.fetchall()

        sampled_records = []
        def is_abnormal(row):
            pred = row['predicted_heart_rate']
            val = row['heart_rate']
            if pred is not None:
                try:
                    diff = float(val)- float(pred)
                    return diff > 30
                except Exception:
                    return False
            return False

        for row in all_records:
            sampled_records.append({
                "time": row['time'].isoformat(),
                "value": row['heart_rate'],
                "predicted": row['predicted_heart_rate'],
                "abnormal": is_abnormal(row)
            })

        return jsonify({"records": sampled_records})
    except Exception as e:
        print(f"Error fetching heart rate time slot data: {e}")
        # 打印详细错误堆栈，便于调试
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to retrieve heart rate records"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/getWaterAmount', methods=['GET'])
def get_water_amount():
    """
    Retrieves the total water intake for a specific date.
    Query param: date=YYYY-MM-DD
    """
    date_str = request.args.get('date')
    if not date_str:
        today = datetime.now().date()
    else:
        try:
            today = datetime.fromisoformat(date_str).date()
        except Exception:
            return jsonify({"error": "Invalid date format"}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # 这里假设你的表叫 water_drink，字段是 timestamp 和 amount_ml
        cur.execute(
            "SELECT SUM(amount) as total_amount FROM water_drink WHERE CAST(time AS DATE) = %s",
            (today,)
        )
        data = cur.fetchone()
        total_amount = int(data['total_amount']) if data and data['total_amount'] else 0
        return jsonify({"amount": total_amount})
    except Exception as e:
        print(f"Error fetching water amount: {e}")
        return jsonify({"error": "Failed to retrieve water amount"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/setWaterAmountTarget', methods=['POST'])
@swag_from('openapi.yaml')
def set_water_amount_target():
    """
    Sets the daily water intake target.
    ---
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required:
            - target
          properties:
            target:
              type: integer
              description: The daily water intake target in ml.
    responses:
      200:
        description: Water target set successfully.
      400:
        description: Invalid input.
      500:
        description: Internal server error.
    """
    data = request.get_json()
    target = data.get('target')

    if not isinstance(target, int) or target <= 0:
        return jsonify({"error": "Invalid water target. Must be a positive integer."}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_settings (setting_name, setting_value, last_updated) VALUES (%s, %s, %s) "
            "ON CONFLICT (setting_name) DO UPDATE SET setting_value = EXCLUDED.setting_value, last_updated = EXCLUDED.last_updated",
            ('water_target', str(target), datetime.now())
        )
        conn.commit()
        return jsonify({"message": "Water target set successfully"}), 200
    except Exception as e:
        print(f"Error setting water target: {e}")
        return jsonify({"error": "Failed to set water target"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/updateFoodPhoto', methods=['POST'])
def update_food_photo():
    data = request.get_json()
    data_url = data.get('photo', '')
    if not data_url:
        return jsonify({'msg': 'No photo data!'}), 400
    data_url = data.get('photo', '')
    meal_type = data.get('mealType', '')  # 新增餐段参数
    photo_date = data.get('photoDate')  # 新增字段
    if not data_url:
        return jsonify({'msg': 'No photo data!'}), 400

    # 解析 DataURL 格式
    match = re.match(r'data:image/(?P<ext>\w+);base64,(?P<data>.+)', data_url)
    if not match:
        return jsonify({'msg': 'Invalid photo format!'}), 400

    ext = match.group('ext')
    img_data = base64.b64decode(match.group('data'))
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    filename = f'photo_{timestamp}.{ext}'
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, 'wb') as f:
        f.write(img_data)

    # 写入数据库
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO food_photos (file_path, description, photo_date) VALUES (%s, %s, %s) RETURNING id;",
        (file_path, meal_type, photo_date)
    )
    photo_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    # 自动调用识别方式进行识别（异步后台线程，不阻塞返回）
    threading.Thread(target=lambda: asyncio.run(async_ask(file_path, photo_id)), daemon=True).start()

    return jsonify({
        'msg': 'Upload success!',
        'file_path': file_path,
        'photo_id': photo_id
    }), 200

    # What is THIS UNDER ?
    # 解析 DataURL 格式
    match = re.match(r'data:image/(?P<ext>\w+);base64,(?P<data>.+)', data_url)
    if not match:
        return jsonify({'msg': 'Invalid photo format!'}), 400

    ext = match.group('ext')
    img_data = base64.b64decode(match.group('data'))
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    filename = f'photo_{timestamp}.{ext}'
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, 'wb') as f:
        f.write(img_data)

    # 写入数据库
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO food_photos (file_path) VALUES (%s) RETURNING id;",
        (file_path,)
    )
    photo_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        'msg': 'Upload success!',
        'file_path': file_path,
        'photo_id': photo_id,
        'filename': filename  # 新增返回文件名
    }), 200


@app.route('/getFoodPhotos', methods=['GET'])
def get_food_photos():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT id, file_path, description, photo_date, food_name FROM food_photos ORDER BY id DESC")
    photos = []
    for row in cur.fetchall():
        photos.append({
            'id': row['id'],
            'file_path': row['file_path'],
            'filename': os.path.basename(row['file_path']),
            'description': row['description'] if row['description'] else '',
            'photo_date': row['photo_date'].isoformat() if row['photo_date'] else '',
            'food_name': row['food_name'] if row['food_name'] else '',  # 补充这一行
        })
    cur.close()
    conn.close()
    return jsonify({'photos': photos})


@app.route('/getFoodPhotoDetail', methods=['GET'])
def get_food_photo_detail():
    photo_id = request.args.get('photo_id')
    if not photo_id:
        return jsonify({'error': 'photo_id required'}), 400
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
                SELECT file_path,
                       food_name,
                       calorie,
                       protein,
                       fat,
                       carbohydrate,
                       fiber,
                       weight
                FROM food_photos
                WHERE id = %s
                """, (photo_id,))
    data = cur.fetchone()
    cur.close()
    conn.close()
    if not data:
        return jsonify({'error': 'Not found'}), 404

    def clean(val):  # 处理None为"？？"
        return val if val is not None else "？？"

    return jsonify({
        'file_path': data['file_path'],
        'food_name': clean(data['food_name']),
        'Calorie': clean(data['calorie']),
        'Protein': clean(data['protein']),
        'Fat': clean(data['fat']),
        'Carbohydrate': clean(data['carbohydrate']),
        'Fiber': clean(data['fiber']),
        'Weight': clean(data['weight']),  # 新增返回weight
    })


@app.route('/updateFood', methods=['POST'])
def update_food():
    data = request.get_json()
    photo_id = data.get('photoId')
    calories = data.get('calories')
    desc = data.get('desc')  # 你可以用来填 food_name 或 description 字段
    sync = data.get('sync', False)
    new_name = data.get('newName')

    if not photo_id:
        return jsonify({"error": "photoId is required"}), 400

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        if new_name:
            # 只改名字
            cur.execute(
                "UPDATE food_photos SET food_name = %s WHERE id = %s",
                (new_name, photo_id)
            )
            conn.commit()
            return jsonify({"message": "Food name updated", "food_name": new_name}), 200

        if sync:
            # 只更新 food_photos 的 calorie / food_name
            if calories is None or desc is None:
                return jsonify({"error": "Calories and description are required for intake confirmation"}), 400
            cur.execute(
                "UPDATE food_photos SET calorie = %s, food_name = %s WHERE id = %s",
                (calories, desc, photo_id)
            )
            conn.commit()
            return jsonify({"message": "Food info updated in food_photos"}), 200
        else:
            # 识别阶段，可返回模拟数据或什么都不做
            simulated_calories = 350.5
            simulated_desc = "A bowl of noodles with vegetables"
            return jsonify({
                "calories": simulated_calories,
                "desc": simulated_desc,
                "message": "Food recognized (simulated)"
            }), 200
    except Exception as e:
        print(f"Error updating food information: {e}")
        return jsonify({"error": "Failed to update food information"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/getCaloriesUsedToday', methods=['GET'])
@swag_from('openapi.yaml')
def get_calories_used_today():
    """
    Retrieves the total calories burned for a specific date.
    ---
    parameters:
      - name: date
        in: query
        type: string
        format: date
        required: true
        description: Date for which to get calories burned (ISO 8601 format).
    responses:
      200:
        description: A dictionary containing total calories burned.
        schema:
          type: object
          properties:
            calories:
              type: number
              format: float
              description: Total calories burned.
      400:
        description: Invalid date format.
      500:
        description: Internal server error.
    """
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        query_date = datetime.fromisoformat(date_str.split('T')[0]).date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Assuming 'daily_activity_summary' table stores daily calories burned
        cur.execute(
            "SELECT SUM(calories_burned) as total_burned FROM daily_activity_summary WHERE activity_date = %s",
            (query_date,)
        )
        data = cur.fetchone()
        total_burned = float(data['total_burned']) if data and data['total_burned'] else 0.0
        return jsonify({"calories": total_burned})
    except Exception as e:
        print(f"Error fetching calories used today: {e}")
        return jsonify({"error": "Failed to retrieve calories burned"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/setCaloriesTarget', methods=['POST'])
@swag_from('openapi.yaml')
def set_calories_target():
    """
    Sets the daily calorie target.
    ---
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required:
            - target
          properties:
            target:
              type: integer
              description: The daily calorie target in kcal.
    responses:
      200:
        description: Calorie target set successfully.
      400:
        description: Invalid input.
      500:
        description: Internal server error.
    """
    data = request.get_json()
    target = data.get('target')

    if not isinstance(target, int) or target <= 0:
        return jsonify({"error": "Invalid calorie target. Must be a positive integer."}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_settings (setting_name, setting_value, last_updated) VALUES (%s, %s, %s) "
            "ON CONFLICT (setting_name) DO UPDATE SET setting_value = EXCLUDED.setting_value, last_updated = EXCLUDED.last_updated",
            ('calorie_target', str(target), datetime.now())
        )
        conn.commit()
        return jsonify({"message": "Calorie target set successfully"}), 200
    except Exception as e:
        print(f"Error setting calorie target: {e}")
        return jsonify({"error": "Failed to set calorie target"}), 500
    finally:
        if conn:
            conn.close()


from datetime import datetime

@app.route('/getRoutine', methods=['GET'])
@swag_from('openapi.yaml')
def get_routine():
    """
    Retrieves routine path (GPS coordinates) for the server's current day (ignores input time range, always fetches server localdate).
    """
    query_date = datetime.now().date()  # 服务器当前日期

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            "SELECT latitude, longitude, time, speed FROM gps_data WHERE DATE(time) = %s ORDER BY time ASC",
            (query_date,)
        )
        route_points = []
        for row in cur.fetchall():
            route_points.append({
                "lat": row['latitude'],
                "lng": row['longitude'],
                "time": row['time'].isoformat(),
                "speed": row['speed']
            })
        return jsonify({"route": route_points})
    except Exception as e:
        print(f"Error fetching routine data: {e}")
        return jsonify({"error": "Failed to retrieve routine data"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/photo_view_image')
def photo_view_image():
    filename = request.args.get('filename')
    if not filename:
        return "Filename required", 400
    # 统一用 UPLOAD_FOLDER
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    # 防止路径穿越攻击
    if not os.path.abspath(file_path).startswith(os.path.abspath(UPLOAD_FOLDER)):
        return "Invalid filename", 403
    if not os.path.exists(file_path):
        return "File not found", 404
    ext = os.path.splitext(file_path)[1].lower()
    mimetype = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return send_file(file_path, mimetype=mimetype)


@app.route('/food_photo_by_filename')
def food_photo_by_filename():
    filename = request.args.get('filename')
    raw = request.args.get('raw')  # 可选参数，如果有 raw 就只返回图片
    if not filename:
        return "Filename required", 400

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # 防止路径穿越攻击
    if not os.path.abspath(file_path).startswith(os.path.abspath(UPLOAD_FOLDER)):
        return "Invalid filename", 403

    # 查询数据库
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
                SELECT id,
                       file_path,
                       food_name,
                       calorie,
                       protein,
                       fat,
                       carbohydrate,
                       fiber
                FROM food_photos
                WHERE file_path LIKE %s
                ORDER BY id DESC LIMIT 1
                """, (f"%{filename}",))
    data = cur.fetchone()
    cur.close()
    conn.close()

    if not os.path.exists(file_path):
        return "File not found", 404

    # 只返回图片数据
    ext = os.path.splitext(file_path)[1].lower()
    mimetype = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    if raw == '1':
        return send_file(file_path, mimetype=mimetype)

    # 返回图片和营养信息
    def clean(val):  # 处理None为"？？"
        return val if val is not None else "??"

    return jsonify({
        'id': data['id'] if data else None,
        'file_path': data['file_path'] if data else file_path,
        'img_url': f"/food_photo_by_filename?filename={filename}&raw=1",
        'food_name': clean(data['food_name']) if data else "??",
        'calorie': clean(data['calorie']) if data else "??",
        'protein': clean(data['protein']) if data else "??",
        'fat': clean(data['fat']) if data else "??",
        'carbohydrate': clean(data['carbohydrate']) if data else "??",
        'fiber': clean(data['fiber']) if data else "??"
    })


@app.route('/deleteFoodPhoto', methods=['POST'])
def delete_food_photo():
    """
    删除 food_photos 表某条记录，并删除文件
    参数: photoId
    """
    data = request.get_json()
    photo_id = data.get('photoId')
    if not photo_id:
        return jsonify({'error': 'photoId required'}), 400
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT file_path FROM food_photos WHERE id = %s", (photo_id,))
        record = cur.fetchone()
        if not record:
            cur.close()
            conn.close()
            return jsonify({'error': 'Photo not found'}), 404
        file_path = record['file_path']
        # 删除数据库行
        cur.execute("DELETE FROM food_photos WHERE id = %s", (photo_id,))
        conn.commit()
        cur.close()
        conn.close()
        # 删除物理文件
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove file: {file_path}, err: {e}")
            # 返回200也行，只是警告
        return jsonify({'message': 'Photo and record deleted'}), 200
    except Exception as e:
        print(f"Error deleting food photo: {e}")
        return jsonify({'error': 'Failed to delete record'}), 500


# 在食物部分添加质量
@app.route('/getScaleMode', methods=['GET'])
def get_scale_mode():
    """
    查询 scale_weight 表最后一条记录的 mode 状态。
    返回: { "canSwitch": true/false, "last_mode": "food"/"water" }
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500
    try:
        cur = conn.cursor()
        cur.execute("SELECT mode FROM scale_weight ORDER BY time DESC LIMIT 1")
        row = cur.fetchone()
        last_mode = row[0] if row else None
        cur.close()
        conn.close()

        can_switch = last_mode == 'food'
        return jsonify({'canSwitch': can_switch, 'last_mode': last_mode}), 200
    except Exception as e:
        print(f"getScaleMode error: {e}")
        return jsonify({'error': 'Failed to check mode.'}), 500


@app.route('/getLatestScaleWeight', methods=['GET'])
def get_latest_scale_weight():
    """
    获取最新一条 scale_weight 的 weight（mode=food）
    GET /getLatestScaleWeight?mode=food
    """
    mode = request.args.get('mode', 'food')
    if mode != 'food':
        return jsonify({'error': 'Only food mode supported'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            "SELECT weight FROM scale_weight WHERE mode=%s ORDER BY time DESC LIMIT 1",
            ('food',)
        )
        data = cur.fetchone()
        cur.close()
        conn.close()
        weight_val = float(data['weight']) if data else None
        return jsonify({'weight': weight_val})
    except Exception as e:
        print(f"getLatestScaleWeight error: {e}")
        return jsonify({'error': 'Failed to get weight.'}), 500


@app.route('/setLatestFoodPhotoWeight', methods=['POST'])
def set_latest_food_photo_weight():
    """
    更新 food_photos 表最后一行的 weight 列
    请求: { "weight": xx }
    """
    data = request.get_json()
    weight = data.get('weight')
    if weight is None:
        return jsonify({'error': 'weight required'}), 400
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500
    try:
        cur = conn.cursor()
        # 查找最后一行 id
        cur.execute("SELECT id FROM food_photos ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return jsonify({'error': 'No photo row found'}), 404
        last_id = row[0]
        # 更新 weight 列
        cur.execute("UPDATE food_photos SET weight=%s WHERE id=%s", (weight, last_id))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'message': 'Weight updated', 'id': last_id, 'weight': weight}), 200
    except Exception as e:
        print(f"setLatestFoodPhotoWeight error: {e}")
        return jsonify({'error': 'Failed to update weight'}), 500


# 运动模式

# 全局变量模拟会话（正式应用中建议用数据库/redis/session）
current_workout_session = {
    'session_id': None,
    'start_time': None,
    'exercise_type': None,
    'pushup_count': 0,
    'squat_count': 0,
    'running_data': {
        'distance': 0,
        'speed': 0,
        'route': []
    }
}
# 模拟实时数据
realtime_data_stream = {
    'heart_rate': 80,
    'calories': 100
}


@app.route('/api/workout/start', methods=['POST'])
def start_workout():
    global current_workout_session
    data = request.json
    exercise_type = data.get('exercise_type')
    if not exercise_type:
        return jsonify({"error": "exercise_type is required"}), 400
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM workout_records WHERE end_time IS NULL LIMIT 1")
        if cursor.fetchone():
            conn.close()
            return jsonify({"error": "A workout is already in progress"}), 400
    conn.close()
    session_id = f"session_{int(time.time())}"
    current_workout_session['session_id'] = session_id
    current_workout_session['start_time'] = time.time()
    current_workout_session['exercise_type'] = exercise_type
    current_workout_session['pushup_count'] = 0
    current_workout_session['squat_count'] = 0
    current_workout_session['running_data'] = {'distance': 0, 'speed': 0, 'route': []}
    #调用SportDataHandler
    if exercise_type == 'running':
        sport_handler.start_running()
    elif exercise_type == 'pushup':
        sport_handler.start_pushup()
    elif exercise_type == 'squat':
        sport_handler.start_squat()
    else:
        return jsonify({"error": "Unsupported exercise type"}), 400

    return jsonify({"session_id": session_id}), 200


@app.route('/api/workout/stop', methods=['POST'])
def stop_workout():
    # 不再校验 session_id
    # 直接尝试停止所有未结束的运动（实际是数据库里 end_time is null 的那条）
    try:
        # 检查数据库中有没有未结束的运动
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute("""
                SELECT id, exercise_type FROM workout_records
                WHERE end_time IS NULL
                ORDER BY start_time DESC
                LIMIT 1
            """)
            record = cursor.fetchone()
        conn.close()
        if not record:
            return jsonify({"error": "No active session"}), 400

        exercise_type = record['exercise_type']
        if exercise_type == 'running':
            sport_handler.stop_running()
        elif exercise_type == 'pushup':
            sport_handler.stop_pushup()
        elif exercise_type == 'squat':
            sport_handler.stop_squat()
        else:
            return jsonify({"error": "Unsupported exercise type"}), 400

        return jsonify({"message": "Workout stopped successfully"}), 200
    except Exception as e:
        print(f"Error stopping workout: {e}")
        return jsonify({"error": "Failed to stop workout"}), 500

# @app.route('/api/workout/realtime', methods=['GET'])
# def get_realtime_data():
#     global current_workout_session, realtime_data_stream
#     if current_workout_session['session_id']:
#         # 模拟数据：每次+1
#         realtime_data_stream['heart_rate'] = min(200, realtime_data_stream['heart_rate'] + 1)
#         realtime_data_stream['calories'] = realtime_data_stream['calories'] + 5
#         if current_workout_session['exercise_type'] == 'pushup':
#             current_workout_session['pushup_count'] += 1
#         elif current_workout_session['exercise_type'] == 'squat':
#             current_workout_session['squat_count'] += 1
#         if current_workout_session['exercise_type'] == 'running':
#             current_workout_session['running_data']['speed'] = 6.5
#             last_pos = current_workout_session['running_data']['route'][-1] if current_workout_session['running_data'][
#                 'route'] else [31.2304, 121.4737]
#             new_pos = [last_pos[0] + 0.0001, last_pos[1] + 0.0001]
#             current_workout_session['running_data']['route'].append(new_pos)
#         return jsonify({
#             'heart_rate': realtime_data_stream['heart_rate'],
#             'calories': realtime_data_stream['calories'],
#             'pushup_count': current_workout_session['pushup_count'],
#             'squat_count': current_workout_session['squat_count'],
#             'running_speed': current_workout_session['running_data']['speed'],
#             'running_route': current_workout_session['running_data']['route']
#         })
#     return jsonify({
#         'heart_rate': '--',
#         'calories': '--',
#         'pushup_count': 0,
#         'squat_count': 0,
#         'running_speed': '--',
#         'running_route': []
#     })
from datetime import datetime

@app.route('/api/workout/realtime', methods=['GET'])
def get_realtime_data():
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT *
                FROM workout_records
                WHERE end_time IS NULL
                ORDER BY start_time DESC
                LIMIT 1
            """)
            record = cursor.fetchone()
            if not record:
                # 没有未结束的运动
                return jsonify({'active': False})

            now = datetime.now()
            start_time = record['start_time']
            duration_seconds = (now - start_time).total_seconds()

            cursor.execute("""
                SELECT heart_rate
                FROM band_heart_rate
                WHERE time <= %s
                ORDER BY time DESC
                LIMIT 1
            """, (now,))
            hr_row = cursor.fetchone()
            heart_rate = hr_row['heart_rate'] if hr_row and hr_row['heart_rate'] is not None else '--'

            resp = {
                'active': True,
                'exercise_type': record['exercise_type'],
                'duration': int(duration_seconds),
                'calories': record['total_calories_burned'],
                'pushup_count': record.get('pushup_count', 0),
                'squat_count': record.get('squat_count', 0),
                'distance': int(record.get('running_distance_km', 0) * 1000) if record['exercise_type'] == 'running' else 0,
                'speed': 0,
                'heart_rate': heart_rate,  
                'pace': '--'
            }
            if record['exercise_type'] == 'running':
                d = record.get('running_distance_km', 0) * 1000
                s = duration_seconds if duration_seconds > 0 else 1
                resp['speed'] = round(d / s, 2)
                # 新增：查 band_steps 最新一条步频
                cursor.execute("""
                    SELECT step_frequency FROM band_steps
                    WHERE date(time) = CURRENT_DATE
                    ORDER BY time DESC
                    LIMIT 1
                """)
                freq_row = cursor.fetchone()
                if freq_row and freq_row.get('step_frequency') is not None:
                    resp['pace'] = int(freq_row['step_frequency'])
            return jsonify(resp)
    except Exception as e:
        print(f"Error fetching realtime workout data: {e}")
        return jsonify({"error": "Failed to fetch realtime data"}), 500
    finally:
        conn.close()

@app.route('/api/workout/records', methods=['GET'])
def get_workout_records():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    exercise_type = request.args.get('exercise_type', '').lower()
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            count_query = "SELECT COUNT(*) FROM workout_records"
            if exercise_type:
                count_query += " WHERE exercise_type = %s"
                cursor.execute(count_query, (exercise_type,))
            else:
                cursor.execute(count_query)
            total_records = cursor.fetchone()['count']
            total_pages = (total_records + per_page - 1) // per_page
            offset = (page - 1) * per_page
            query = """
                    SELECT id, 
                           exercise_type, 
                           start_time, 
                           end_time, 
                           total_calories_burned, 
                           pushup_count, 
                           squat_count, 
                           running_distance_km
                    FROM workout_records 
                    """
            params = []
            if exercise_type:
                query += " WHERE exercise_type = %s"
                params.append(exercise_type)
            query += " ORDER BY start_time DESC LIMIT %s OFFSET %s"
            params.extend([per_page, offset])
            cursor.execute(query, tuple(params))
            records = cursor.fetchall()
            formatted_records = []
            for record in records:
                # 动态计算时长（秒）
                start = record['start_time']
                end = record['end_time']
                if end:
                    duration_seconds = (end - start).total_seconds()
                else:
                    # 运动未结束时用当前时间
                    duration_seconds = (datetime.now() - start).total_seconds()
                duration_min = int(duration_seconds // 60)
                duration_sec = int(duration_seconds % 60)
                formatted_record = {
                    'id': record['id'],
                    'date': record['start_time'].strftime('%Y-%m-%d') if record['start_time'] else '',
                    'exercise_type': record['exercise_type'],
                    'duration': f"{duration_min:02d}:{duration_sec:02d}",
                    'calories': round(record['total_calories_burned'], 2) if record['total_calories_burned'] else '--',
                }
                if record['exercise_type'] == 'pushup':
                    formatted_record['details'] = f"Count: {record['pushup_count']}"
                elif record['exercise_type'] == 'squat':
                    formatted_record['details'] = f"Count: {record['squat_count']}"
                elif record['exercise_type'] == 'running':
                    distance = round(record['running_distance_km'], 2) if record['running_distance_km'] is not None else '--'
                    speed = "--"
                    if record['running_distance_km'] is not None and duration_seconds > 0:
                        # km/s * 3600 = km/h
                        speed = round(record['running_distance_km'] / duration_seconds * 3600, 2)
                    formatted_record['details'] = f"Distance: {distance} km, Speed: {speed} km/h"
                formatted_records.append(formatted_record)
    except Exception as e:
        print(f"Error fetching workout records: {e}")
        return jsonify({"error": "Failed to fetch records"}), 500
    finally:
        conn.close()
    return jsonify({
        'records': formatted_records,
        'total_records': total_records,
        'total_pages': total_pages
    }), 200


@app.route('/getCaloriesConsumed', methods=['GET'])
def get_calories_consumed():
    """
    获取指定日期通过食物照片识别得到的总卡路里摄入量（weight * calorie/100）。
    参数: date (YYYY-MM-DD)
    返回: { calories: 总摄入 }
    """
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'date required'}), 400
    try:
        # 支持 date 字符串和 ISO 格式
        query_date = datetime.fromisoformat(date_str.split('T')[0]).date()
    except Exception:
        return jsonify({'error': 'date format error'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
                    SELECT weight, calorie
                    FROM food_photos
                    WHERE photo_date = %s
                    """, (query_date,))
        total_calories = 0.0
        for row in cur.fetchall():
            weight = row['weight'] if row['weight'] is not None else 0
            calorie = row['calorie'] if row['calorie'] is not None else 0
            # calorie单位是/100g，weight单位是g
            # 总卡路里 = weight * calorie / 100
            try:
                weight_val = float(weight)
                calorie_val = float(calorie)
                total_calories += weight_val * calorie_val / 100
            except Exception:
                pass
        cur.close()
        conn.close()
        total_calories = round(total_calories, 2)
        return jsonify({'calories': total_calories}), 200
    except Exception as e:
        print(f"get_calories_consumed error: {e}")
        return jsonify({'error': 'Failed to calculate calories'}), 500


@app.route('/getWaterTarget', methods=['GET'])
def get_water_target():
    """
    查询用户饮水目标设置
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT setting_value FROM user_settings WHERE setting_name = 'water_target'")
        row = cur.fetchone()
        target = int(row['setting_value']) if row else 2000  # 默认2000ml
        cur.close()
        conn.close()
        return jsonify({'target': target}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to get target'}), 500



@app.route('/setWeight', methods=['POST'])
def set_weight():
    """
    设置体重（存入 user_settings 表）
    请求: { "weight": 体重(kg, int/float) }
    """
    data = request.get_json()
    weight = data.get('weight')
    try:
        weight_val = float(weight)
        if weight_val <= 0:
            raise ValueError
    except Exception:
        return jsonify({"error": "Invalid weight. Must be a positive number."}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_settings (setting_name, setting_value, last_updated) VALUES (%s, %s, %s) "
            "ON CONFLICT (setting_name) DO UPDATE SET setting_value = EXCLUDED.setting_value, last_updated = EXCLUDED.last_updated",
            ('weight', str(weight_val), datetime.now())
        )
        conn.commit()
        return jsonify({"message": "Weight set successfully"}), 200
    except Exception as e:
        print(f"Error setting weight: {e}")
        return jsonify({"error": "Failed to set weight"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/getCaloriesTarget', methods=['GET'])
def get_calories_target():
    """
    查询用户卡路里目标设置
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT setting_value FROM user_settings WHERE setting_name = 'calorie_target'")
        row = cur.fetchone()
        target = int(row['setting_value']) if row else 2000  # 默认2000kcal
        cur.close()
        conn.close()
        return jsonify({'target': target}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to get calorie target'}), 500


@app.route('/getStepTarget', methods=['GET'])
def get_step_target():
    """
    查询用户步数目标设置
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT setting_value FROM user_settings WHERE setting_name = 'step_target'")
        row = cur.fetchone()
        target = int(row['setting_value']) if row else 10000  # 默认10000步
        cur.close()
        conn.close()
        return jsonify({'target': target}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to get step target'}), 500

@app.route('/setAge', methods=['POST'])
def set_age():
    """
    设置用户年龄
    请求体: { "age": 25 }
    """
    data = request.get_json()
    age = data.get('age')
    if not isinstance(age, int) or age <= 0:
        return jsonify({"error": "Invalid age. Must be a positive integer."}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_settings (setting_name, setting_value, last_updated) VALUES (%s, %s, %s) "
            "ON CONFLICT (setting_name) DO UPDATE SET setting_value = EXCLUDED.setting_value, last_updated = EXCLUDED.last_updated",
            ('age', str(age), datetime.now())
        )
        conn.commit()
        return jsonify({"message": "Age set successfully"}), 200
    except Exception as e:
        print(f"Error setting age: {e}")
        return jsonify({"error": "Failed to set age"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/setGender', methods=['POST'])
def set_gender():
    """
    设置用户性别
    请求体: { "gender": "male" / "female" }
    """
    data = request.get_json()
    gender = data.get('gender', '').lower()
    if gender not in ['male', 'female']:
        return jsonify({"error": "Invalid gender, must be 'male' or 'female'."}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_settings (setting_name, setting_value, last_updated) VALUES (%s, %s, %s) "
            "ON CONFLICT (setting_name) DO UPDATE SET setting_value = EXCLUDED.setting_value, last_updated = EXCLUDED.last_updated",
            ('gender', gender, datetime.now())
        )
        conn.commit()
        return jsonify({"message": "Gender set successfully"}), 200
    except Exception as e:
        print(f"Error setting gender: {e}")
        return jsonify({"error": "Failed to set gender"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/getGender', methods=['GET'])
def get_gender():
    """
    获取用户性别
    返回: { "gender": "male" / "female" }
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT setting_value FROM user_settings WHERE setting_name = 'gender'")
        row = cur.fetchone()
        gender = row['setting_value'] if row and row['setting_value'] in ('male', 'female') else 'male'
        cur.close()
        conn.close()
        return jsonify({'gender': gender}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to get gender'}), 500



@app.route('/getSedentaryStatus', methods=['GET'])
def get_sedentary_status():
    """
    返回今天已经过去的小时里有stand的小时列表
    """
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        cur = conn.cursor()
        # 查询今天0点到当前时间内所有stand的小时（不重复）
        cur.execute("""
            SELECT DISTINCT EXTRACT(HOUR FROM "time") AS hour
            FROM "band_position"
            WHERE "position" = 'stand'
              AND "time" >= %s
              AND "time" < %s
            ORDER BY hour
        """, (today_start, now))
        rows = cur.fetchall()
        stand_hours = sorted([int(row[0]) for row in rows])
        total_stand = len(stand_hours)
        return jsonify({
            "stand_hours": stand_hours,
            "total_stand": total_stand
        })
    except Exception as e:
        import traceback
        print(f"Error in get_sedentary_status: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


def get_running_route():
    """
    根据 session_id 和 start_time, 获取本次 running 的所有GPS点。
    GET 参数: session_id, start_time (ISO格式)
    返回: { "route": [ {"lat": ..., "lng": ..., "time": ...}, ... ] }
    """
    session_id = request.args.get('session_id')
    start_time_str = request.args.get('start_time')
    if not session_id or not start_time_str:
        return jsonify({'route': []})
    try:
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
    except Exception:
        return jsonify({'route': []})

    # 假设 gps_data 有 session_id 字段（如没有可用时间区间筛选）
    conn = get_db_connection()
    if not conn:
        return jsonify({'route': []})
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # 优先session_id筛选，如果没有则用start_time
        cur.execute("""
            SELECT latitude, longitude, time, speed
            FROM gps_data
            WHERE time >= %s
            ORDER BY time ASC
        """, (start_time,))
        points = []
        for row in cur.fetchall():
            points.append({
                "lat": row['latitude'],
                "lng": row['longitude'],
                "time": row['time'].isoformat(),
                "speed": row['speed']
            })
        return jsonify({'route': points})
    except Exception as e:
        print(f"get_running_route error: {e}")
        return jsonify({'route': []})
    finally:
        conn.close()

@app.route('/getRunningSessionRoute', methods=['GET'])
def get_running_session_route():
    """
    获取当前（或最近一次）running session的GPS轨迹点。
    返回: { "route": [ {"lat": ..., "lng": ..., "time": ...}, ... ] }
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({'route': []})
    try:
        # 查找最新一条running的workout记录（end_time可能为null=进行中，或为已完成）
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT id, start_time, end_time FROM workout_records
            WHERE exercise_type='running'
            ORDER BY start_time DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            return jsonify({'route': []})
        start_time = row['start_time']
        end_time = row['end_time'] or datetime.now()
        # 查找该时间段内所有gps点
        cur.execute("""
            SELECT latitude, longitude, time FROM gps_data
            WHERE time BETWEEN %s AND %s
            ORDER BY time ASC
        """, (start_time, end_time))
        points = []
        for gps in cur.fetchall():
            points.append({
                "lat": gps['latitude'],
                "lng": gps['longitude'],
                "time": gps['time'].isoformat()
            })
        return jsonify({'route': points})
    except Exception as e:
        print(f"get_running_session_route error: {e}")
        return jsonify({'route': []})
    finally:
        conn.close()

@app.route('/getWorkoutRouteById', methods=['GET'])
def get_workout_route_by_id():
    """
    通过workout_records表的id获取该运动的GPS轨迹
    参数: id
    返回: { route: [ {lat, lng, time, speed}, ... ] }
    """
    workout_id = request.args.get('id')
    if not workout_id:
        return jsonify({'error': 'id required'}), 400
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT exercise_type, start_time, end_time FROM workout_records WHERE id = %s", (workout_id,))
        row = cur.fetchone()
        if not row:
            return jsonify({'route': []})
        if row['exercise_type'] != 'running':
            return jsonify({'route': []})
        start_time = row['start_time']
        end_time = row['end_time'] or datetime.now()
        cur.execute("""
            SELECT latitude, longitude, time, speed
            FROM gps_data
            WHERE time BETWEEN %s AND %s
            ORDER BY time ASC
        """, (start_time, end_time))
        points = []
        for gps in cur.fetchall():
            points.append({
                "lat": gps['latitude'],
                "lng": gps['longitude'],
                "time": gps['time'].isoformat(),
                "speed": gps['speed']
            })
        return jsonify({'route': points})
    except Exception as e:
        print(f"get_workout_route_by_id error: {e}")
        return jsonify({'route': []})
    finally:
        conn.close()


if __name__ == '__main__':
    conn = get_db_connection()
    if conn:
        print("Database connected.")
        conn.close()
    else:
        print("Database connection failed.")
    
    # 修改: 关闭自动重载，避免SportDataHandler重复实例化
    app.run(host='0.0.0.0', port=5000, debug=False)
    # 如果需要调试，可以使用下面的选项，禁用自动重载
    # app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
