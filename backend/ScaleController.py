from MicroBitController import MicroBitController
import threading
from datetime import datetime, timedelta
import time
from utils.logger import get_logger

class ScaleController(MicroBitController):
    def __init__(self, MAC, conn=None):
        self.logger = get_logger()
        super().__init__(MAC)
        self.logger.info(f"ScaleController 实例化成功，MAC地址: {MAC}")
        self.conn = conn  # 数据库连接
        self.last_water_weight = None
        self.last_drink_time = datetime.now()
        if(self.conn is None):
            if self.conn is None:
                self.logger.warning("数据库连接不可用，无法存储数据")
                return
        else:
            cursor = self.conn.cursor()
            two_hours_ago = datetime.now() - timedelta(hours=2)
            cursor.execute("""
                SELECT weight FROM scale_weight
                WHERE mode='water' AND time >= %s
                ORDER BY time DESC LIMIT 1;
            """, (two_hours_ago,))
            result = cursor.fetchone()
            if result:
                self.last_water_weight = result[0]
        

    def run(self):
        super().run()  # 调用父类的 run 方法以连接到 micro:bit 并开始订阅数据流
        time.sleep(2)  # 等待连接稳定
        self.logger.info("ScaleController 正在运行...")
        self.send_uart_message("calibrate")
        self.logger.info("ScaleController 已启动，开始接收和处理数据")
        self.start_weight_loop()  # 启动重量读取循环
        self.start_drink_reminder_loop()  # 启动喝水提醒循环
        self.get_today_water_amount_loop()  # 启动今日喝水量获取循环


    def uart_data_received(self, data):
        try:
            # 处理 dbus.Array 类型
            if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                # 将 dbus.Byte 转换成字节列表
                byte_list = [int(byte) for byte in data]
                # 转换成字节对象
                byte_data = bytes(byte_list)
                # 解码成字符串
                received_data = byte_data.decode('utf-8').strip()
                self.logger.debug(f"从 Scale 收到 UART 数据 (解码后): {received_data}")
            elif isinstance(data, bytes):
                received_data = data.decode('utf-8').strip()
                self.logger.debug(f"从 Scale 收到 UART 数据: {received_data}")
            else:
                received_data = str(data).strip()
                self.logger.debug(f"从 Scale 收到 UART 数据 (转换为字符串): {received_data}")
        except (UnicodeDecodeError, ValueError) as e:
            self.logger.error(f"从 Scale 收到原始 UART 数据 (无法解码): {data}, 错误: {e}")
        self.msgHandler(received_data)
    
    def send_uart_message(self, message):
        self.logger.info(f"正在向 Scale 发送: {message.strip()}")
        return super().send_uart_message(message)
    
    def msgHandler(self, message: str):
        command = message.split(":")
        if command[0] == "getTime":  # 获取时间的命令
            now = datetime.now()
            today_zero = now.replace(hour=0, minute=0, second=0, microsecond=0)
            seconds_since_midnight = int((now - today_zero).total_seconds())
            response = f"time_upd:{seconds_since_midnight}"+"\n"
            self.send_uart_message(response)
        elif command[0] == "Weight":  # 更新重量
            try:
                # 单位为克（g）
                weight = int(float(command[1]))  # 体重单位为g
                if(command[2] == "w"):
                    mode = "water"
                elif(command[2] == "f"):
                    mode = "food"
                else:
                    mode = "NA"
                now = datetime.now()
                self.logger.debug(f"从 Scale 收到水杯数据: {weight} g, 模式: {mode}")
                # 新增：连续三次读数不变且变化大于20g才存入数据库，或模式变化时也存
                if not hasattr(self, '_last_recorded_weight'):
                    self._last_recorded_weight = None
                if not hasattr(self, '_last_recorded_mode'):
                    self._last_recorded_mode = None
                if not hasattr(self, '_weight_buffer'):
                    self._weight_buffer = []
                self._weight_buffer.append(weight)
                if len(self._weight_buffer) > 3:
                    self._weight_buffer.pop(0)
                # 判断是否需要存储：模式变化 或 连续三次读数一致且重量变化大于20g
                mode_changed = (self._last_recorded_mode is None or mode != self._last_recorded_mode)
                stable_weight = (len(self._weight_buffer) == 3 and all(w == self._weight_buffer[0] for w in self._weight_buffer))
                weight_changed = (self._last_recorded_weight is None or abs(weight - self._last_recorded_weight) > 20)
                if mode_changed or (stable_weight and weight_changed):
                    if self.conn is None:
                        self.logger.warning("数据库连接不可用，无法存储体重数据")
                        return
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        INSERT INTO scale_weight (time, weight, mode)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (time) DO NOTHING;
                    """, (now, weight, mode))
                    self.conn.commit()
                    self.logger.debug(f"成功存储重量数据: {weight} g, 时间: {now}, 模式: {mode}")
                    self._last_recorded_weight = weight
                    self._last_recorded_mode = mode
                    # 新增：水杯模式下调用 update_water_amount
                    if mode == "water":
                        self.update_water_amount(weight, now)
                else:
                    self.logger.debug(f"未满足存储条件（模式未变且重量变化小于20g或未稳定），未存储")
            except ValueError as e:
                self.logger.error(f"无法解析质量数据: {command[1]}, 错误: {e}")

    def getWeight(self):
        msg = "getWeight"+"\n"
        self.logger.debug(f"正在向 Scale 发送重量请求: {msg.strip()}")
        self.send_uart_message(msg)

    def start_weight_loop(self,interval=5): #每5s更新一次
        def loop():
            while True:
                self.getWeight()  # 获取当前重量
                time.sleep(interval)
        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def update_water_amount(self, weight, now):
        # 忽略小于10g的数据，等待用户喝完水再放下
        if weight < 10:
            self.logger.debug("水杯质量小于10g，可能正在喝水，等待放下")
            return
        if self.last_water_weight is None:
            self.last_water_weight = weight
            self.logger.debug(f"初始化水杯质量: {weight} g")
            return
        if weight > self.last_water_weight:
            # 用户加水，重置基准
            self.logger.info(f"检测到加水，重置水量基准: {weight} g")
            self.last_water_weight = weight
            return
        elif weight < self.last_water_weight:
            # 用户喝水，计算喝水量
            drink_amount = self.last_water_weight - weight
            self.logger.info(f"用户喝水: {drink_amount} g")
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO water_drink (time, amount)
                VALUES (%s, %s);
            """, (now, drink_amount))
            self.conn.commit()
            self.logger.debug(f"成功存储喝水数据: {drink_amount} g, 时间: {now}")
            self.last_water_weight = weight
            self.last_drink_time = now

    def start_drink_reminder_loop(self, interval=3, remind_gap=120):
        # interval: 检查间隔秒数，remind_gap: 距离上次喝水多少秒后提醒
        def reminder_loop():
            while True:
                now = datetime.now()
                if (now - self.last_drink_time).total_seconds() > remind_gap:
                    self.logger.info("距离上次喝水已超过"+str(int(remind_gap/60))+"分钟，发送喝水提醒")
                    self.send_uart_message("drinkAnnounce\n")
                    #更新 last_drink_time 为当前时间
                    self.last_drink_time = now
                time.sleep(interval)
        t = threading.Thread(target=reminder_loop, daemon=True)
        t.start()

    def get_today_water_amount(self):
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法获取今日喝水量")
            return 0
        cursor = self.conn.cursor()
        today_zero = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cursor.execute("""
            SELECT SUM(amount) FROM water_drink
            WHERE time >= %s;
        """, (today_zero,))
        result = cursor.fetchone()
        total_amount = result[0] if result[0] is not None else 0
        self.send_uart_message(f"todayWater:{total_amount}\n")
        self.logger.debug(f"今日喝水总量: {total_amount} g")

    def get_today_water_amount_loop(self, interval=60):
        # 每隔60秒获取一次今日喝水量
        def loop():
            while True:
                self.get_today_water_amount()
                time.sleep(interval)
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self.logger.info("ScaleController 已启动喝水量获取循环")






















