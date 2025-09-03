import threading
import time
import numpy
import requests
import signal
import sys
from datetime import datetime, timedelta
from datetime import datetime, date
from utils.logger import get_logger
import os
import psycopg2
from models.CNN.predict_wrapper import predict_steps,load_prediction_model
from models.CNN.predict_wrapper_binary import (predict_binary_classification, load_binary_prediction_model)
from models.SVM.heart_rate_predictor import quick_predict
from models.CNN.fall_detection_predictor import FallDetectionPredictor
# 添加单例模式实现
_instance = None
_instance_lock = threading.Lock()

YOLO_GYM_API = "http://192.168.137.185:8888"


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
    
class SportDataHandler:
    # 实现单例模式
    def __new__(cls, conn=None):
        global _instance
        with _instance_lock:
            if _instance is None:
                _instance = super(SportDataHandler, cls).__new__(cls)
                _instance.is_initialized = False
            return _instance
            
    def __init__(self, conn=None):
        # 避免重复初始化
        if hasattr(self, 'is_initialized') and self.is_initialized:
            return
            
        self.logger = get_logger()
        self.logger.info("SportDataHandler 实例化成功")
        self.conn = get_db_connection() if conn is None else conn  # 使用传入的连接或创建新的连接
        if(self.conn is None):
            if self.conn is None:
                self.logger.warning("数据库连接不可用，无法存储数据")
                return
        # 初始化步数        ·       
        self.model, self.extractor, self.device = load_prediction_model(model_path='backend/models/CNN/step.pth', model_type='lightweight')
        # 从数据库里设置步数为今天最新一次的记录
        #设置二分类模型
        self.binary_model, self.binary_extractor, self.binary_device, self.binary_config = load_binary_prediction_model(model_path='backend/models/CNN/best_binary_model.pth', model_type='lightweight')
        
        # 初始化姿态数据
        self.last_position = None
        try:
            if self.conn is not None:
                with self.conn.cursor() as cursor:
                    # 获取最近一次的姿态数据
                    position_query = """
                    SELECT position FROM band_position
                    ORDER BY time DESC LIMIT 1
                    """
                    cursor.execute(position_query)
                    result = cursor.fetchone()
                    if result:
                        self.last_position = result[0]
                        self.logger.info(f"初始化姿态为最近一次记录: {self.last_position}")
                    else:
                        self.logger.info("数据库中没有姿态记录")
        except Exception as e:
            self.logger.error(f"初始化姿态数据时出错: {e}")
        
        # 从数据库里设置步数为今天最新一次的记录
        try:
            if self.conn is not None:
                with self.conn.cursor() as cursor:
                    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    select_query = """
                    SELECT step_count FROM band_steps
                    WHERE time >= %s
                    ORDER BY time DESC LIMIT 1
                    """
                    cursor.execute(select_query, (today_start,))
                    result = cursor.fetchone()
                    if result:
                        self.step = result[0]
                        self.logger.info(f"初始化步数为今天最新一次记录: {self.step}")
                    else:
                        self.step = 0
                        self.logger.info("今天没有步数记录，初始化步数为0")
        except Exception as e:
            self.logger.error(f"初始化步数时出错: {e}")
            self.step = 0
        
        # 添加一个标志来防止步数识别循环被多次启动
        self.step_loop_started = False
        self.start_recognize_step_loop() # 启动步数识别循环
        
        # 只在主线程中注册信号处理器
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, self._handle_exit)
                signal.signal(signal.SIGTERM, self._handle_exit)
                self.logger.info("已在主线程注册信号处理器")
            except ValueError:
                self.logger.warning("无法注册信号处理器，可能不在主线程")
        
        #运动部分
        self.workout_id = None
        self.running_thread_active = False  # 添加线程控制标志
        self.pushup_thread_active = False   # 添加线程控制标志
        self.squat_thread_active = False    # 添加线程控制标志
        self.start_time = None

        #从数据库得到用户的年龄
        try:
            if self.conn is not None:
                with self.conn.cursor() as cursor:
                    inst = """select setting_value from user_settings
                            where setting_name = 'age';
                             """
                    cursor.execute(inst)
                    result = cursor.fetchone()
                    if result:
                        self.age = result[0]
                        self.logger.info(f"用户年龄初始化为: {self.age}")
                    else:
                        self.age = 25  # 默认值
                        self.logger.info("未找到用户年龄，使用默认值: 25")
        except Exception as e:
            self.logger.error(f"获取用户年龄时出错: {e}")
            self.age = 25

        #获取用户的质量
        try:
            if self.conn is not None:
                with self.conn.cursor() as cursor:
                    inst = """SELECT setting_value FROM user_settings
                              WHERE setting_name = 'weight';"""
                    cursor.execute(inst)
                    result = cursor.fetchone()
                    if result:
                        self.weight = float(result[0])
                        self.logger.info(f"用户质量初始化为: {self.weight} kg")
                    else:
                        self.weight = 70.0
                        self.logger.info("未找到用户质量，使用默认值: 70.0 kg")
        except Exception as e:
            self.logger.error(f"获取用户质量时出错: {e}")
            self.weight = 70.0

        #获取用户的性别
        try:
            if self.conn is not None:
                with self.conn.cursor() as cursor:
                    inst = """SELECT setting_value FROM user_settings
                              WHERE setting_name = 'gender';"""
                    cursor.execute(inst)
                    result = cursor.fetchone()
                    if result:
                        self.gender = result[0]
                        self.logger.info(f"用户性别初始化为: {self.gender}")
                    else:
                        self.gender = 'male'
                        self.logger.info("未找到用户性别，使用默认男性")
        except Exception as e:
            self.logger.error(f"获取用户性别时出错: {e}")
            self.gender = 'male'
        self.heart_rate_loop()
        self.schedule_daily_retrain() # 启动心率预测模型的定时重训练

        # 初始化摔倒检测预测器
        # self.fall_predictor = FallDetectionPredictor(model_path="backend/models/CNN/best_fall_detection_lightweight.pth",model_type='lightweight')
        self.calories_loop() #启动总卡路里计算

    #这个文件会自动多线程运行下列即时处理与更新信息的函数
    def RecognizeStep(self):
        #从服务器获取最近20次的步数记录
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法获取步数记录")
            return
        try:
            # 检查当前日期，如果与上次运行不同，则重置步数
            current_date = datetime.now().date()
            if not hasattr(self, 'last_date') or self.last_date != current_date:
                # 重置步数为今天的最新记录或0
                with self.conn.cursor() as cursor:
                    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    select_query = """
                    SELECT step_count FROM band_steps
                    WHERE time >= %s
                    ORDER BY time DESC LIMIT 1
                    """
                    cursor.execute(select_query, (today_start,))
                    result = cursor.fetchone()
                    if result:
                        self.step = result[0]
                        self.logger.info(f"新的一天开始，重置步数为今天最新记录: {self.step}")
                    else:
                        self.step = 0
                        self.logger.info("新的一天开始，重置步数为0")
                self.last_date = current_date
                
            with self.conn.cursor() as cursor:
                # 先检查最新一条加速度记录的时间
                time_check_query = """
                SELECT time FROM band_acceleration ORDER BY time DESC LIMIT 1
                """
                cursor.execute(time_check_query)
                latest_record = cursor.fetchone()
                
                if not latest_record:
                    self.logger.debug("没有找到加速度记录")
                    return
                
                latest_time = latest_record[0]
                current_time = datetime.now()
                time_diff = (current_time - latest_time).total_seconds()
                
                if time_diff > 1:
                    self.logger.warning(f"最新加速度记录时间与当前时间相差 {time_diff} 秒，超过1秒，不更新步数")
                    return
                
                select_query = """
                select x,y,z from band_acceleration order by time desc limit 100
                """
                cursor.execute(select_query)
                data = cursor.fetchall()
                if not data or len(data) < 100:
                    self.logger.debug("没有找到步数记录")
                    return
                data_array = numpy.array(data)
                step_increment = predict_steps(data_array,self.model,self.extractor,self.device)
                self.step += step_increment
                self.logger.info(f"识别到步数: {self.step}")

                # 自动计算最近一分钟步频
                # 查询最近一分钟步数
                minute_ago = datetime.now() - timedelta(minutes=1)
                freq_query = """
                SELECT MAX(step_count), MIN(step_count) FROM band_steps WHERE time >= %s
                """
                cursor.execute(freq_query, (minute_ago,))
                freq_result = cursor.fetchone()
                if freq_result and freq_result[0] is not None and freq_result[1] is not None:
                    step_freq = freq_result[0] - freq_result[1]
                else:
                    step_freq = 0
                self.logger.info(f"最近一分钟步频: {step_freq} 步/分钟")

                #存储到数据库
                current_time = datetime.now()
                insert_query = """
                INSERT INTO band_steps (time, step_count, step_frequency)
                VALUES (%s, %s, %s)
                """
                cursor.execute(insert_query, (current_time, self.step, step_freq))
                self.conn.commit()
                self.logger.debug("成功存储步数和步频记录到数据库")

                #顺便在这里进行姿态识别
                position = predict_binary_classification(data_array, self.binary_model, self.binary_extractor, self.binary_device)
                if position is not None:
                    self.logger.info(f"当前姿态识别结果: {position}")
                    # 只在姿态发生变化时存储
                    if position != self.last_position:
                        self.logger.info(f"姿态变化: {self.last_position} -> {position}，存储到数据库")
                        # 存储姿态识别结果到数据库
                        position_query = """
                        INSERT INTO band_position (time, position) VALUES (%s, %s)
                        """
                        cursor.execute(position_query, (current_time, position))
                        self.conn.commit()
                        # 更新上次姿态
                        self.last_position = position
                    else:
                        self.logger.debug(f"姿态未发生变化，保持为: {position}，不存储到数据库")
                else:
                    self.logger.warning("姿态识别结果为 None，未存储到数据库")
                # #顺便在这里进行摔倒检测
                # pos = self.fall_predictor.predict(data_array)["class_name"]
                # if pos == "摔倒":
                #     self.logger.warning("检测到摔倒，触发警报")
                #     fall_query = """
                #     INSERT INTO band_fall (time) VALUES (%s)
                #     """
                #     cursor.execute(fall_query, (current_time,))
                #     self.conn.commit()
            


        except Exception as e:
            self.logger.error(f"获取步数记录时出错: {e}")
            return
        
    def start_recognize_step_loop(self, interval=2):
        # 添加检查避免多次启动
        if self.step_loop_started:
            self.logger.warning("步数识别循环已经在运行中，不再重复启动")
            return
            
        def loop():
            while True:
                self.RecognizeStep()
                time.sleep(interval)
                
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self.step_loop_started = True
        self.logger.info("步数识别循环已启动")

    def start_running(self):
        # 启动跑步
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法插入运动记录")
            return None
        try:
            with self.conn.cursor() as cursor:
                start_time = datetime.now()
                self.start_time = start_time  # 保存开始时间
                insert_query = """
                INSERT INTO workout_records (
                    exercise_type, start_time, end_time,
                    total_calories_burned, pushup_count, squat_count,
                    running_distance_km, abnormal_heart_rate, abnormal_hr_times
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                ) RETURNING id
                """
                cursor.execute(insert_query, (
                    "running", start_time, None,
                    None, None, None,
                    None, None, None
                ))
                workout_id = cursor.fetchone()[0]
                self.conn.commit()
                self.logger.info(f"新建跑步记录，id={workout_id}")
                self.workout_id = workout_id
                # 启动实时更新跑步距离的线程
                self.start_running_data_updater(workout_id)
                return workout_id
        except Exception as e:
            self.logger.error(f"插入跑步记录时出错: {e}")
            return None
        
    def running_data_updater(self, workout_id):
        # 实时更新跑步距离
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法更新跑步记录")
            return
        try:
            with self.conn.cursor() as cursor:
                # 获取最近两条GPS数据
                gps_query = """
                SELECT time, longitude, latitude FROM gps_data
                ORDER BY time DESC LIMIT 2
                """
                cursor.execute(gps_query)
                gps_points = cursor.fetchall()
                if len(gps_points) < 2:
                    self.logger.debug("GPS数据不足，无法计算距离")
                    return

                # 检查最新GPS数据的时间，若与当前时间相差超过2秒则速度设为0
                current_time = datetime.now()
                latest_gps_time = gps_points[0][0]
                time_diff = (current_time - latest_gps_time).total_seconds()
                
                if time_diff > 2:
                    self.logger.warning(f"GPS数据时间与当前时间相差 {time_diff:.1f} 秒，超过2秒，速度设为0")
                    raw_distance = 0
                else:
                    # 使用 geopy 计算距离（单位：千米）
                    from geopy.distance import geodesic
                    t1, lon1, lat1 = gps_points[1]
                    t2, lon2, lat2 = gps_points[0]
                    
                    # 计算两点间距离
                    raw_distance = geodesic((lat1, lon1), (lat2, lon2)).km
                    
                    # 计算速度（m/s）
                    time_elapsed = (t2 - t1).total_seconds()
                    if time_elapsed > 0:
                        speed_mps = (raw_distance * 1000) / time_elapsed
                        # 忽略小于0.4m/s的速度
                        if speed_mps < 0.4:
                            self.logger.debug(f"当前速度 {speed_mps:.2f} m/s 小于0.4m/s，忽略此段距离")
                            raw_distance = 0
                    else:
                        self.logger.warning("GPS点时间间隔异常，忽略此段距离")
                        raw_distance = 0

                # 距离滑动平均平滑处理
                if not hasattr(self, 'distance_window'):
                    self.distance_window = []
                self.distance_window.append(raw_distance)
                if len(self.distance_window) > 5:  # 取最近5次
                    self.distance_window.pop(0)
                distance = sum(self.distance_window) / len(self.distance_window)

                # 查询当前累计距离（单位：千米）
                select_dist_query = """
                SELECT running_distance_km FROM workout_records WHERE id = %s
                """
                cursor.execute(select_dist_query, (workout_id,))
                result = cursor.fetchone()
                current_distance = result[0] if result and result[0] else 0.0
                new_distance = current_distance + distance

                # 更新累计距离
                update_query = """
                UPDATE workout_records SET running_distance_km = %s WHERE id = %s
                """
                cursor.execute(update_query, (new_distance, workout_id))
                self.conn.commit()
                self.logger.info(f"跑步距离已更新: {new_distance:.3f} km")

                #计算跑步消耗的卡路里
                #计算运动的平均心率
                inst = """
                SELECT AVG(heart_rate)
                FROM band_heart_rate
                WHERE time BETWEEN %s AND %s
                """
                try:
                    cursor.execute(inst, (self.start_time, current_time))
                    avg_heart_rate_result = cursor.fetchone()
                    # 强制转换为 float，防止 Decimal 类型导致报错
                    avg_heart_rate = float(avg_heart_rate_result[0]) if avg_heart_rate_result and avg_heart_rate_result[0] else 0.0
                    if avg_heart_rate > 0:
                        self.logger.info(f"平均心率: {avg_heart_rate} bpm")
                    else:
                        self.logger.warning("没有找到心率数据，无法计算平均心率")
                        avg_heart_rate = 70.0  # 设置默认值
                except Exception as e:
                    self.logger.error(f"计算平均心率时出错: {e}")
                    avg_heart_rate = 70.0

                # Harris-Benedict公式
                # 保持你的原始公式，只做类型转换
                age = float(self.age) if not isinstance(self.age, float) else self.age
                weight = float(self.weight) if not isinstance(self.weight, float) else self.weight
                calories = (age * 0.074 - weight * 0.05741 + avg_heart_rate * 0.4472 - 20.4022) / 4.184
                if self.gender == "male":
                    calories *= 0.63
                else:
                    calories *= 0.55
                # 更新卡路里消耗
                update_calories_query = """
                UPDATE workout_records SET total_calories_burned = %s WHERE id = %s
                """
                cursor.execute(update_calories_query, (calories, workout_id))
                self.conn.commit()
                self.logger.info(f"跑步卡路里消耗已更新: {calories:.2f} kcal")


        except Exception as e:
            self.logger.error(f"更新跑步距离时出错: {e}")
            return
        
   
    #一个新线程每2s更新一次
    def start_running_data_updater(self, workout_id, interval=2):
        if hasattr(self, 'start_running_data_updater_thread'):
            self.logger.warning("跑步数据更新线程已在运行")
            return
        self.running_thread_active = True  # 设置线程活动标志
        def loop():
            while self.running_thread_active:  # 使用标志控制循环
                self.running_data_updater(workout_id)
                time.sleep(interval)
        self.start_running_data_updater_thread = threading.Thread(target=loop, daemon=True)
        self.start_running_data_updater_thread.start()

    def stop_running_data_updater(self):
        # 停止实时更新跑步距离的线程
        if hasattr(self, 'start_running_data_updater_thread'):
            self.running_thread_active = False  # 设置标志以停止循环
            self.start_running_data_updater_thread.join(timeout=1)
            del self.start_running_data_updater_thread
            self.logger.info("跑步数据更新线程已停止")
        else:
            self.logger.warning("没有正在运行的跑步数据更新线程")

    def stop_running(self):
        # 停止跑步
        self.start_time = None  # 清除开始时间
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法停止跑步")
            return
        try:
            with self.conn.cursor() as cursor:
                end_time = datetime.now()
                update_query = """
                UPDATE workout_records SET end_time = %s WHERE id = %s
                """
                cursor.execute(update_query, (end_time, self.workout_id))
                self.conn.commit()
                self.logger.info(f"跑步记录已结束，ID={self.workout_id}")
        except Exception as e:
            self.logger.error(f"停止跑步时出错: {e}")
            return
        self.workout_id = None
        self.stop_running_data_updater()
        self.logger.info("跑步已停止")

    def start_pushup(self):
        """
        启动俯卧撑检测，启动计数线程
        """
        try:
            resp = requests.post(YOLO_GYM_API+"/start")
            if resp.status_code != 200:
                self.logger.error("无法启动俯卧撑检测服务")
                return
            self.logger.info("已启动俯卧撑检测服务，开始计数")
            insert_query = """
            INSERT INTO workout_records (exercise_type, start_time, end_time, total_calories_burned
            ) VALUES (%s, %s, %s, %s) RETURNING id
            """
            with self.conn.cursor() as cursor:
                cursor.execute(insert_query, ("pushup", datetime.now(), None, None))
                self.workout_id = cursor.fetchone()[0]
                self.conn.commit()
                self.logger.info(f"新建俯卧撑记录，id={self.workout_id}")
            # 启动实时更新俯卧撑计数的线程
            self.start_pushup_data_updater()
        except Exception as e:
            self.logger.error(f"启动俯卧撑检测异常: {e}")

    def pushup_data_updater(self):
        """
        轮询获取俯卧撑计数，并存入数据库
        """
        last_count = 0
        while self.pushup_thread_active:  # 使用标志控制循环
            try:
                status_resp = requests.get(YOLO_GYM_API+"/status")
                if status_resp.status_code != 200:
                    self.logger.error("无法获取俯卧撑状态")
                    break
                status = status_resp.json()
                pushup_count = status.get("pushup_count", 0)
                self.logger.info(f"当前俯卧撑计数: {pushup_count}")
                
                if self.conn is not None:
                    with self.conn.cursor() as cursor:
                        if pushup_count > last_count:
                            last_count = pushup_count
                            #存入数据库
                            update_query = """
                            UPDATE workout_records SET pushup_count = %s WHERE id = %s
                            """
                            cursor.execute(update_query, (pushup_count, self.workout_id))
                            self.conn.commit()
                            self.logger.info("俯卧撑计数已存入数据库")

                        #计算运动的平均心率
                        current_time = datetime.now()
                        if not self.start_time:
                            self.start_time = current_time  # 如果没有开始时间，设置为当前时间
                            
                        inst = """
                        SELECT AVG(heart_rate)
                        FROM band_heart_rate
                        WHERE time BETWEEN %s AND %s
                        """
                        try:
                            cursor.execute(inst, (self.start_time, current_time))
                            avg_heart_rate_result = cursor.fetchone()
                            # 强制转换为 float，防止 Decimal 类型导致报错
                            avg_heart_rate = float(avg_heart_rate_result[0]) if avg_heart_rate_result and avg_heart_rate_result[0] else 0.0
                            
                            if avg_heart_rate > 0:
                                self.logger.info(f"平均心率: {avg_heart_rate} bpm")
                            else:
                                self.logger.warning("没有找到心率数据，无法计算平均心率")
                                avg_heart_rate = 70.0  # 设置默认值
                        except Exception as e:
                            self.logger.error(f"计算平均心率时出错: {e}")
                            avg_heart_rate = 70.0

                        # 使用基于MET值的卡路里计算公式
                        # MET公式: 卡路里 = MET值 × 体重(kg) × 运动时长(小时)
                        # 俯卧撑的MET值约为5.75
                        weight = float(self.weight) if not isinstance(self.weight, float) else self.weight
                        exercise_time_hours = (current_time - self.start_time).total_seconds() / 3600.0
                        met_value = 5.75  # 俯卧撑的MET值
                        calories = met_value * weight * exercise_time_hours
                        
                        # 考虑运动强度(通过心率)调整卡路里消耗
                        if avg_heart_rate > 120:
                            intensity_factor = 1.1  # 高强度，增加10%
                        elif avg_heart_rate < 90:
                            intensity_factor = 0.9  # 低强度，减少10%
                        else:
                            intensity_factor = 1.0  # 中等强度
                            
                        calories *= intensity_factor
                            
                        #存储到数据库
                        update_calories_query = """
                        UPDATE workout_records SET total_calories_burned = %s WHERE id = %s
                        """
                        cursor.execute(update_calories_query, (calories, self.workout_id))
                        self.conn.commit()
                        self.logger.info(f"俯卧撑卡路里消耗已更新: {calories:.2f} kcal")
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"俯卧撑计数轮询异常: {e}")
                break

    def start_pushup_data_updater(self):
        """
        启动俯卧撑计数更新线程
        """
        if not hasattr(self, 'pushup_data_updater_thread'):
            self.pushup_thread_active = True  # 设置线程活动标志
            self.pushup_data_updater_thread = threading.Thread(target=self.pushup_data_updater, daemon=True)
            self.pushup_data_updater_thread.start()
            self.logger.info("已启动俯卧撑计数更新线程")
        else:
            self.logger.warning("俯卧撑计数更新线程已在运行")
    
    def stop_pushup_data_updater(self):
        """
        停止俯卧撑计数更新线程
        """
        print("Stopping pushup data updater thread...")
        if hasattr(self, 'pushup_data_updater_thread'):
            self.pushup_thread_active = False  # 设置标志以停止循环
            self.pushup_data_updater_thread.join(timeout=1)  # 使用join等待线程停止
            del self.pushup_data_updater_thread
            self.logger.info("俯卧撑计数更新线程已停止")
        else:
            self.logger.warning("没有正在运行的俯卧撑计数更新线程")



    def stop_pushup(self):
        """
        停止俯卧撑检测，保存最终计数
        """
        try:
            stop_resp = requests.post(YOLO_GYM_API+"/stop")
            if stop_resp.status_code == 200:
                session = stop_resp.json().get("session", {})
                final_count = session.get("pushup_count", 0)
                self.logger.info(f"本次俯卧撑最终计数: {final_count}")
                if self.conn is not None:
                    with self.conn.cursor() as cursor:
                        update_query = """
                        UPDATE workout_records SET pushup_count = %s, end_time = %s WHERE id = %s
                        """
                        end_time = datetime.now()
                        cursor.execute(update_query, (final_count,end_time, self.workout_id))
                        self.conn.commit()
                        self.logger.info("俯卧撑计数已存入数据库")
                self.stop_pushup_data_updater()
            else:
                self.logger.error("无法停止俯卧撑检测服务")
        except Exception as e:
            self.logger.error(f"停止俯卧撑检测异常: {e}")

    
    def start_squat(self):
        """
        启动深蹲检测，启动计数线程
        """
        try:
            resp = requests.post(YOLO_GYM_API+"/start")
            if resp.status_code != 200:
                self.logger.error("无法启动深蹲检测服务")
                return
            self.logger.info("先启动检测")
            resp = requests.post(YOLO_GYM_API+"/switch")
            if resp.status_code != 200:
                self.logger.error("无法切换到深蹲检测服务")
                return
            self.logger.info("已启动深蹲检测服务，开始计数")
            insert_query = """
            INSERT INTO workout_records (exercise_type, start_time, end_time, total_calories_burned
            ) VALUES (%s, %s, %s, %s) RETURNING id
            """
            with self.conn.cursor() as cursor:
                cursor.execute(insert_query, ("squat", datetime.now(), None, None))
                self.workout_id = cursor.fetchone()[0]
                self.conn.commit()
                self.logger.info(f"新建深蹲记录，id={self.workout_id}")
            # 启动实时更新深蹲计数的线程
            self.start_squat_data_updater()
        except Exception as e:
            self.logger.error(f"启动深蹲检测异常: {e}")

    def squat_data_updater(self):
        """
        轮询获取深蹲计数，并存入数据库
        """
        last_count = 0

        while self.squat_thread_active:  # 使用标志控制循环
            try:
                status_resp = requests.get(YOLO_GYM_API+"/status")
                if status_resp.status_code != 200:
                    self.logger.error("无法获取深蹲状态")
                    break
                status = status_resp.json()
                squat_count = status.get("squat_count", 0)
                self.logger.info(f"当前深蹲计数: {squat_count}")

                if self.conn is not None:
                    with self.conn.cursor() as cursor:
                        if squat_count > last_count:
                            last_count = squat_count
                            #存入数据库
                            update_query = """
                            UPDATE workout_records SET squat_count = %s WHERE id = %s
                            """
                            cursor.execute(update_query, (squat_count, self.workout_id))
                            self.conn.commit()
                            self.logger.info("深蹲计数已存入数据库")
                            
                        #计算运动的平均心率
                        current_time = datetime.now()
                        if not self.start_time:
                            self.start_time = current_time  # 如果没有开始时间，设置为当前时间
                            
                        inst = """
                        SELECT AVG(heart_rate)
                        FROM band_heart_rate
                        WHERE time BETWEEN %s AND %s
                        """
                        try:
                            cursor.execute(inst, (self.start_time, current_time))
                            avg_heart_rate_result = cursor.fetchone()
                            # 强制转换为 float，防止 Decimal 类型导致报错
                            avg_heart_rate = float(avg_heart_rate_result[0]) if avg_heart_rate_result and avg_heart_rate_result[0] else 0.0
                            
                            if avg_heart_rate > 0:
                                self.logger.info(f"平均心率: {avg_heart_rate} bpm")
                            else:
                                self.logger.warning("没有找到心率数据，无法计算平均心率")
                                avg_heart_rate = 70.0  # 设置默认值
                        except Exception as e:
                            self.logger.error(f"计算平均心率时出错: {e}")
                            avg_heart_rate = 70.0

                        # 使用基于MET值的卡路里计算公式
                        # MET公式: 卡路里 = MET值 × 体重(kg) × 运动时长(小时)
                        # 深蹲的MET值约为6.75
                        weight = float(self.weight) if not isinstance(self.weight, float) else self.weight
                        exercise_time_hours = (current_time - self.start_time).total_seconds() / 3600.0
                        met_value = 6.75  # 深蹲的MET值
                        calories = met_value * weight * exercise_time_hours
                        
                        # 考虑运动强度(通过心率)调整卡路里消耗
                        if avg_heart_rate > 120:
                            intensity_factor = 1.1  # 高强度，增加10%
                        elif avg_heart_rate < 90:
                            intensity_factor = 0.9  # 低强度，减少10%
                        else:
                            intensity_factor = 1.0  # 中等强度
                            
                        calories *= intensity_factor
                                
                        #存储到数据库
                        update_calories_query = """
                        UPDATE workout_records SET total_calories_burned = %s WHERE id = %s
                        """
                        cursor.execute(update_calories_query, (calories, self.workout_id))
                        self.conn.commit()
                        self.logger.info(f"深蹲卡路里消耗已更新: {calories:.2f} kcal")
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"深蹲计数轮询异常: {e}")
                break

    def start_squat_data_updater(self):
        """
        启动深蹲计数更新线程
        """
        if not hasattr(self, 'squat_data_updater_thread'):
            self.squat_thread_active = True  # 设置线程活动标志
            self.squat_data_updater_thread = threading.Thread(target=self.squat_data_updater, daemon=True)
            self.squat_data_updater_thread.start()
            self.logger.info("已启动深蹲计数更新线程")
        else:
            self.logger.warning("深蹲计数更新线程已在运行")

    def stop_squat_data_updater(self):
        """
        停止深蹲计数更新线程
        """
        if hasattr(self, 'squat_data_updater_thread'):
            self.squat_thread_active = False  # 设置标志以停止循环
            self.squat_data_updater_thread.join(timeout=1)  # 使用join等待线程停止
            del self.squat_data_updater_thread
            self.logger.info("深蹲计数更新线程已停止")
        else:
            self.logger.warning("没有正在运行的深蹲计数更新线程")

    def stop_squat(self):
        """
        停止深蹲检测，保存最终计数
        """
        try:
            stop_resp = requests.post(YOLO_GYM_API+"/stop")
            if stop_resp.status_code == 200:
                session = stop_resp.json().get("session", {})
                final_count = session.get("squat_count", 0)
                self.logger.info(f"本次深蹲最终计数: {final_count}")
                if self.conn is not None:
                    with self.conn.cursor() as cursor:
                        update_query = """
                        UPDATE workout_records SET squat_count = %s, end_time = %s  WHERE id = %s
                        """
                        end_time = datetime.now()
                        cursor.execute(update_query, (final_count, end_time, self.workout_id))
                        self.conn.commit()
                        self.logger.info("深蹲计数已存入数据库")
                self.stop_squat_data_updater()
            else:
                self.logger.error("无法停止深蹲检测服务")
        except Exception as e:
            self.logger.error(f"停止深蹲检测异常: {e}")
    
    def _handle_exit(self, sig, frame):
        """处理进程退出信号，清理资源"""
        self.logger.info("收到退出信号，清理资源...")
        
        # 停止所有运行中的线程
        if self.step_loop_started:
            self.step_loop_started = False
            self.logger.info("已停止步数识别循环")
            
        if hasattr(self, 'running_thread_active') and self.running_thread_active:
            self.stop_running_data_updater()
            
        if hasattr(self, 'pushup_thread_active') and self.pushup_thread_active:
            self.stop_pushup_data_updater()
            
        if hasattr(self, 'squat_thread_active') and self.squat_thread_active:
            self.stop_squat_data_updater()
        self.logger.info("所有资源已清理完毕，退出进程")
        sys.exit(0)
            
        if hasattr(self, 'pushup_thread_active') and self.pushup_thread_active:
            self.stop_pushup_data_updater()
            
        if hasattr(self, 'squat_thread_active') and self.squat_thread_active:
            self.stop_squat_data_updater()
        self.logger.info("所有资源已清理完毕，退出进程")
        sys.exit(0)

    def heart_rate_predict(self):
        """心率预测函数"""
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法进行心率预测")
            return
        try:
            with self.conn.cursor() as cursor:
                #获取最新的速度和步频信息
                speed_query = """
                select time,speed from gps_data order by time desc limit 1;
                """
                cursor.execute(speed_query)
                speed_result = cursor.fetchone()
                if not speed_result:
                    self.logger.warning("没有找到速度数据，无法进行心率预测")
                speed_time, speed = speed_result
                #如果速度的时间与当前时间相差超过5秒，则认为当前速度为0
                current_time = datetime.now()
                time_diff = (current_time - speed_time).total_seconds()
                if time_diff > 5:
                    self.logger.warning(f"速度数据时间与当前时间相差 {time_diff:.1f} 秒，超过5秒，速度设为0")
                    speed = 0.0
                else:
                    self.logger.info(f"当前速度: {speed} m/s")
                #将速度转换为km/h
                speed = speed * 3.6  # m/s to km/h
                #获取最新的步频信息
                step_query = """    
                select time,step_frequency from band_steps order by time desc limit 1;
                """
                cursor.execute(step_query)
                step_result = cursor.fetchone()
                if not step_result:
                    self.logger.warning("没有找到步频数据，无法进行心率预测")
                    return
                step_time, step_frequency = step_result
                #如果步频的时间与当前时间相差超过5秒，则认为当前步频为0
                time_diff = (current_time - step_time).total_seconds()
                if time_diff > 5:
                    self.logger.warning(f"步频数据时间与当前时间相差 {time_diff:.1f} 秒，超过5秒，步频设为0")
                    step_frequency = 0.0
                else:
                    self.logger.info(f"当前步频: {step_frequency} 步/分钟")
                #调用SVM
                predicted = quick_predict(cadence=step_frequency, speed=speed)
                predicted = float(predicted)
                
                updated_query = """
                UPDATE band_heart_rate
                SET predicted_heart_rate = %s
                WHERE time = (SELECT MAX(time) FROM band_heart_rate);
                """
                
                cursor.execute(updated_query, (predicted,))
                self.conn.commit()
                self.logger.info(f"成功存储预测心率数据: {predicted} bpm")

        except Exception as e:
            self.logger.error(f"心率预测时出错: {e}")
            return None
    
    def heart_rate_loop(self, interval=1):
        """心率预测循环"""
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法启动心率获取循环")
            return
        
        def loop():
            while True:
                self.heart_rate_predict()
                time.sleep(interval)
                
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self.logger.info("心率获取循环已启动")

    def retrain_model(self):
        """
        自动重训心率预测模型：
        1. 自动从数据库获取所有的历史训练数据
        2. 训练SVM模型
        3. 保存模型到 backend/models/SVM/SVM_models/YYYYMMDD.pkl
        4. 同时更新 newest.pkl
        """
        import os
        from datetime import datetime
        from utils.get_training_data import get_training_data
        from models.SVM import SVM_HR_train
        import traceback

        try:
            # 1. 获取训练数据
            df = get_training_data()
            if df is None or df.empty or len(df) < 10:
                self.logger.error("自动重训失败：训练数据为空或不足10条")
                return False

            # 2. 训练模型
            cadence = df.iloc[:, 0].values
            speed = df.iloc[:, 1].values
            heart_rate = df.iloc[:, 2].values
            predictor = SVM_HR_train.HeartRatePredictor(C=10.0, epsilon=0.05, gamma='auto')
            success = predictor.train(cadence, speed, heart_rate)
            if not success:
                self.logger.error("自动重训失败：模型训练失败")
                return False

            # 3. 保存模型
            model_dir = os.path.join(os.path.dirname(__file__), 'models', 'SVM', 'SVM_models')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            today_str = datetime.now().strftime('%Y%m%d')
            model_path = os.path.join(model_dir, f'{today_str}.pkl')
            newest_path = os.path.join(model_dir, 'newest.pkl')
            predictor.save_model(model_path)
            predictor.save_model(newest_path)
            self.logger.info(f"模型已保存: {model_path}，并更新为最新模型: {newest_path}")
            return True
        except Exception as e:
            self.logger.error(f"自动重训异常: {e}\n{traceback.format_exc()}")
            return False

    def schedule_daily_retrain(self):
        """
        启动每天凌晨1点自动重训定时任务
        """
        import threading, time
        from datetime import datetime, timedelta
        def retrain_loop():
            while True:
                now = datetime.now()
                # 计算下一个1点的时间
                next_run = now.replace(hour=1, minute=0, second=0, microsecond=0)
                if now >= next_run:
                    next_run += timedelta(days=1)
                wait_seconds = (next_run - now).total_seconds()
                self.logger.info(f"距离下次模型自动重训还有 {wait_seconds/3600:.2f} 小时")
                time.sleep(wait_seconds)
                self.logger.info("开始自动重训心率预测模型...")
                self.retrain_model()
        t = threading.Thread(target=retrain_loop, daemon=True)
        t.start()
        self.logger.info("已启动每日凌晨1点自动重训定时任务")

    def calculate_net_running_steps(self,target_date: date) -> int | None:
        """
        计算某一天的跑步净步数。

        Args:
            target_date (date): 目标日期，例如 datetime.date(2023, 10, 26)。
            db_config (dict): 包含数据库连接信息的字典，例如：
                            {
                                'host': 'your_host',
                                'database': 'your_database',
                                'user': 'your_user',
                                'password': 'your_password'
                            }

        Returns:
            int | None: 跑步产生的净步数，如果当日没有跑步记录或数据异常则返回 None。
        """
        try:
            conn = self.conn  # 使用类的数据库连接
            # 1. 连接到 PostgreSQL 数据库
            cur = conn.cursor()

            # 2. 查询 workout_records 表获取跑步记录的开始和结束时间
            # 请根据你的 workout_records 表的实际列名进行调整
            query_workouts = """
            SELECT start_time, end_time
            FROM workout_records
            WHERE exercise_type = 'running'
            AND DATE(start_time) = %s
            ORDER BY start_time;
            """
            cur.execute(query_workouts, (target_date,))
            running_workouts = cur.fetchall()

            if not running_workouts:
                print(f"在 {target_date} 没有找到跑步记录。")
                return None

            total_net_steps = 0

            for start_time, end_time in running_workouts:
                # 3. 查询 band_steps 表获取跑步开始前的步数
                # 这是一个累计值，所以我们需要找到在 start_time 之前和 end_time 之前的步数
                query_start_steps = """
                SELECT step_count
                FROM band_steps
                WHERE time <= %s
                ORDER BY time DESC
                LIMIT 1;
                """
                cur.execute(query_start_steps, (start_time,))
                start_step_record = cur.fetchone()

                # 4. 查询 band_steps 表获取跑步结束时的步数
                query_end_steps = """
                SELECT step_count
                FROM band_steps
                WHERE time <= %s
                ORDER BY time DESC
                LIMIT 1;
                """
                cur.execute(query_end_steps, (end_time,))
                end_step_record = cur.fetchone()

                if start_step_record and end_step_record:
                    start_steps = start_step_record[0]
                    end_steps = end_step_record[0]

                    # 确保结束步数不小于开始步数
                    if end_steps >= start_steps:
                        net_steps_for_workout = end_steps - start_steps
                        total_net_steps += net_steps_for_workout
                    else:
                        print(f"警告: 跑步记录 ({start_time} - {end_time}) 出现步数倒退，结束步数 ({end_steps}) 小于开始步数 ({start_steps})。该次记录可能存在数据问题。")
                else:
                    print(f"警告: 无法获取跑步记录 ({start_time} - {end_time}) 对应的步数数据。")
            

            return total_net_steps
        except Exception as e:
            print(f"发生错误: {e}")
            return None
    
    def calculate_today_total_calories(self):
        """
        计算今天的总卡路里消耗。
        """
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法计算今天的总卡路里")
            return None
        try:
            with self.conn.cursor() as cursor:
                today = datetime.now().date()
                # 查询所有运动消耗
                query = """
                SELECT SUM(total_calories_burned)
                FROM workout_records
                WHERE DATE(start_time) = %s;
                """
                cursor.execute(query, (today,))
                result = cursor.fetchone()
                total_calories = result[0] if result and result[0] else 0.0

                # 查询当天最新步数
                step_query = """
                SELECT step_count FROM band_steps
                WHERE DATE(time) = %s
                ORDER BY time DESC LIMIT 1;
                """
                cursor.execute(step_query, (today,))
                step_result = cursor.fetchone()
                latest_steps = step_result[0] if step_result else 0

                # 计算当天跑步净步数
                net_running_steps = self.calculate_net_running_steps(today)
                if net_running_steps is None:
                    non_running_steps = latest_steps
                else:
                    non_running_steps = max(0, latest_steps - net_running_steps)

                # 步行卡路里（每1000步约30千卡）
                step_calories = (non_running_steps / 1000) * 30 if non_running_steps else 0
                total_calories += step_calories

                inst_or_update = """
                INSERT INTO daily_activity_summary (activity_date, calories_burned)
                VALUES (%s, %s)
                ON CONFLICT (activity_date) DO UPDATE
                SET calories_burned = EXCLUDED.calories_burned;
                """
                cursor.execute(inst_or_update, (today, total_calories))
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"计算今天的总卡路里时出错: {e}")
            return None
        
    def calories_loop(self, interval=5):
        """
        启动一个循环，每隔指定时间计算一次今天的总卡路里消耗。
        """
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法启动卡路里计算循环")
            return
        
        def loop():
            while True:
                self.calculate_today_total_calories()
                time.sleep(interval)
                
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self.logger.info("卡路里计算循环已启动")
        
if __name__ == "__main__":
    sportdatahandler = SportDataHandler()
    print(sportdatahandler.calculate_net_running_steps(datetime(2025, 7, 15).date()))







