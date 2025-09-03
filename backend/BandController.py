from MicroBitController import MicroBitController
import threading
from datetime import datetime
import time

from utils.logger import get_logger

class BandController(MicroBitController):
    def __init__(self, MAC, conn=None):
        self.logger = get_logger()
        super().__init__(MAC)
        self.logger.info(f"BandController 实例化成功，MAC地址: {MAC}")
        self.conn = conn  # 数据库连接
        self.step = 0  # 初始化步数
        self.last_heart_rate_time = None  # 上一次心率存储时间

    def run(self):
        super().run()  # 调用父类的 run 方法以连接到 micro:bit 并开始订阅数据流
        self.start_heart_rate_loop()  # 启动心率获取循环
        self.start_load_step_data_loop()  # 启动步数加载循环
        self.start_stand_detection_loop()  # 启动站立检测循环
        self.check_heart_rate_loop()  # 启动心率检查循环
        
    def accelerometer_data_received(self, x, y, z):
        #self.logger.debug(f"Band 收到加速度计数据: X={x}, Y={y}, Z={z}")
        
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法存储数据")
            return
            
        try:
            with self.conn.cursor() as cursor:
                current_time = datetime.now()
                insert_query = """
                INSERT INTO band_acceleration (time, x, y, z)
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(insert_query, (current_time, x, y, z))
                self.conn.commit()
                #self.logger.debug("成功存储加速度计数据到数据库")
                
        except Exception as e:
            self.logger.error(f"存储数据时出错: {e}")
            # 如果连接出现问题，尝试重新连接
            if not self.conn or self.conn.closed:
                self._init_db_connection()

    def uart_data_received(self, data):
        #self.logger.debug(f"Band 收到 UART 数据: {data}")
        try:
            # 处理 dbus.Array 类型
            if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                # 将 dbus.Byte 转换成字节列表
                byte_list = [int(byte) for byte in data]
                # 转换成字节对象
                byte_data = bytes(byte_list)
                # 解码成字符串
                received_data = byte_data.decode('utf-8').strip()
                self.logger.debug(f"从 micro:bit 收到 UART 数据 (解码后): {received_data}")
                
                        
            elif isinstance(data, bytes):
                received_data = data.decode('utf-8').strip()
                self.logger.debug(f"从 micro:bit 收到 UART 数据: {received_data}")
            else:
                received_data = str(data).strip()
                self.logger.debug(f"从 micro:bit 收到 UART 数据 (转换为字符串): {received_data}")
        except (UnicodeDecodeError, ValueError) as e:
            self.logger.error(f"从 micro:bit 收到原始 UART 数据 (无法解码): {data}, 错误: {e}")
        self.msgHandler(received_data)
    
    def send_uart_message(self, message):
        self.logger.info(f"正在向 Band 发送: {message.strip()}")
        return super().send_uart_message(message)
    
    def msgHandler(self, message: str):
        command = message.split(":")
        if command[0] == "getTime":  # 获取时间的命令
            now = datetime.now()
            today_zero = now.replace(hour=0, minute=0, second=0, microsecond=0)
            seconds_since_midnight = int((now - today_zero).total_seconds())
            response = f"time_upd:{seconds_since_midnight}"+"\n"
            self.send_uart_message(response)
        elif command[0] == "HeartRate":  # 心率的命令
            self.HeartRateHandler(command[1])
        elif command[0] == "getStep":  # 获取步数的命令
            response = f"StepUpd:{self.step}\n"
            self.send_uart_message(response)

            
    #Heart Rate相关
    def getHeartRate(self):
        msg = "getHeartRate"+"\n"
        self.logger.debug(f"正在向 Band 发送心率请求: {msg.strip()}")
        self.send_uart_message(msg)
    
    def HeartRateHandler(self, data):
        try:
            heart_rate = float(data)
            self.logger.info(f"从 Band 收到心率数据: {heart_rate}")
            now = datetime.now()
            # 忽略距离上次存储不足0.3秒的数据
            if self.last_heart_rate_time is not None:
                elapsed = (now - self.last_heart_rate_time).total_seconds()
                if elapsed < 0.3:
                    self.logger.debug("心率数据间隔小于0.3秒，自动忽略")
                    return
            self.last_heart_rate_time = now

            if self.conn is None:
                self.logger.warning("数据库连接不可用，无法存储心率数据")
                return
            
            with self.conn.cursor() as cursor:
                current_time = now
                insert_query = """
                INSERT INTO band_heart_rate (time, heart_rate)
                VALUES (%s, %s)
                """
                cursor.execute(insert_query, (current_time, heart_rate))
                self.conn.commit()
                self.logger.info("成功存储心率数据到数据库")
        except ValueError as e:
            self.logger.error(f"处理心率数据时出错: {e}")

    def start_heart_rate_loop(self, interval=2): # 默认每2秒获取一次心率
        def loop():
            while True:
                self.getHeartRate()
                time.sleep(interval)
        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def load_step_data(self):
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法加载步数数据")
            return
        try:
            with self.conn.cursor() as cursor:
                select_query = "SELECT step_count FROM band_steps ORDER BY time LIMIT 1"
                cursor.execute(select_query)
                result = cursor.fetchone()
                if result:
                    self.step = result[0]
                    self.logger.info(f"加载步数数据成功: {self.step}")
                else:
                    self.logger.warning("未找到步数数据")
                    self.step = 0
        except Exception as e:
            self.logger.error(f"加载步数数据时出错: {e}")
            return []
        return self.step
    
    def start_load_step_data_loop(self, interval=5):
        def loop():
            while True:
                self.load_step_data()
                time.sleep(interval)
        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def stand_announce(self): # 站立通知
        #从数据库里获取状态
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法获取站立状态")
            return
        try:
            with self.conn.cursor() as cursor:
                select_query = "SELECT position FROM band_position ORDER BY time DESC LIMIT 1"
                cursor.execute(select_query)
                result = cursor.fetchone()
                if result:
                    stand_status = result[0]
                    if stand_status == "stand":
                        self.logger.info("站立状态: 已站立")
                        return True
                    else:
                        self.logger.info("站立状态: 未站立")
                        return False
                else:
                    self.logger.warning("未找到站立状态数据")
                    return False
        except Exception as e:
            self.logger.error(f"获取站立状态时出错: {e}")
            return False

    def start_stand_detection_loop(self, interval=30, sit_timeout=120):
        """
        每 interval 秒检测一次是否需要发送站立提醒
        sit_timeout: 坐着超过多少秒后提醒
        """
        def loop():
            while True:
                self.check_stand_reminder(sit_timeout=sit_timeout)
                time.sleep(interval)
        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def check_stand_reminder(self, sit_timeout=120):
        """
        检查用户是否已经坐着超过 sit_timeout 秒，如果是则发送提醒
        """
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法检测站立状态")
            return
        try:
            with self.conn.cursor() as cursor:
                # 获取最近一次的position和时间
                select_query = "SELECT position, time FROM band_position ORDER BY time DESC LIMIT 1"
                cursor.execute(select_query)
                result = cursor.fetchone()
                if result:
                    position, last_time = result
                    if position == "sit":
                        now = datetime.now()
                        # last_time 可能是 datetime 类型，也可能是字符串
                        if isinstance(last_time, str):
                            last_time = datetime.fromisoformat(last_time)
                        elapsed = (now - last_time).total_seconds()
                        if elapsed >= sit_timeout:
                            self.logger.info(f"检测到用户已坐着超过{sit_timeout}秒，发送站立提醒")
                            self.send_uart_message("standAnnounce\n")
                # else: 没有数据则不提醒
        except Exception as e:
            self.logger.error(f"检测站立状态时出错: {e}")

    def check_heart_rate(self, threshold=30):
        """
        检查预测值-心率是否超过阈值，如果超过则发送提醒
        """
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法检查心率")
            return
        try:
            with self.conn.cursor() as cursor:
                select_query = "select heart_rate,predicted_heart_rate from band_heart_rate order by time desc limit 1 offset 1;"
                cursor.execute(select_query)
                result = cursor.fetchone()
                if result:
                    heart_rate, predicted_heart_rate = result
                    if predicted_heart_rate is None:
                        self.logger.warning("预测心率数据不可用，无法进行心率检查")
                        return
                    if heart_rate - predicted_heart_rate > threshold:
                        self.logger.info(f"检测到心率异常: 实际心率={heart_rate}, 预测心率={predicted_heart_rate}")
                        self.send_uart_message("HRWarning\n")
                else:
                    self.logger.warning("未找到心率数据")
        except Exception as e:
            self.logger.error(f"检查心率时出错: {e}")
    
    def check_heart_rate_loop(self, interval=10, threshold=30):
        """
        每 interval 秒检查一次心率是否异常
        """
        def loop():
            while True:
                self.check_heart_rate(threshold=threshold)
                time.sleep(interval)
        t = threading.Thread(target=loop, daemon=True)
        t.start()


