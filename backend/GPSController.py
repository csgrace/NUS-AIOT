from bluezero import adapter, GATT
from bluezero import device  # 新增导入
from gi.repository import GLib
import time
import json
import threading
from datetime import datetime
from utils.logger import get_logger
from geopy.distance import geodesic

class GPSController():
    def __init__(self, MAC, conn=None):
        self.mac_address = MAC
        self.logger = get_logger()
        bluetooth_adapter = adapter.Adapter()
        bluetooth_adapter.start_discovery()
        print("正在扫描附近的蓝牙设备...")
        time.sleep(1)  
        bluetooth_adapter.stop_discovery()
        self.logger.info(f"GPSController 实例化成功，MAC地址: {MAC}")
        self.conn = conn  # 数据库连接
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法存储数据")
            
        # 状态变量
        self.gps_device = None
        self.connected = False
        self.running = False
        self.last_position = None
        
        # 数据存储
        self.positions = []
        self.speeds = []
        self.timestamps = []
        self.total_distance = 0.0
        self.max_speed = 0.0
        self.avg_speed = 0.0
        self.start_time = None
        
        # 创建事件循环
        self.mainloop = None
        self._reconnect_thread = None
        self._auto_reconnect = True
        
        # GPS设备特征值UUID - 确保与ESP32端匹配
        self.SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
        self.CHARACTERISTIC_UUID = "87654321-4321-4321-4321-cba987654321"
        
        # 特性对象
        self._gps_char = None
    
    @property
    def is_connected(self):
        """返回当前连接状态"""
        return self.connected
    
    def scan_devices(self):
        """扫描附近的GPS设备"""
        self.logger.info("开始扫描GPS设备")
        try:
            bluetooth_adapter = adapter.Adapter()
            # 启动扫描
            bluetooth_adapter.start_discovery()
            self.logger.info("正在扫描附近的蓝牙设备...")
            time.sleep(3)  # 给一些时间进行扫描
            
            found_devices = bluetooth_adapter.devices()
            bluetooth_adapter.stop_discovery()
            
            for device in found_devices:
                device_addr = device['address']
                device_name = device.get('name', '')
                self.logger.debug(f"发现设备: {device_name} ({device_addr})")
                
                # 根据名称匹配GPS设备
                if device_name and ("GPS" in device_name or "ESP32" in device_name):
                    self.logger.info(f"找到目标GPS设备: {device_name} ({device_addr})")
                    return device_addr
                
            self.logger.warning("未找到匹配的GPS设备")
            return None
            
        except Exception as e:
            self.logger.error(f"扫描设备时出错: {e}")
            return None
    
    def connect(self):
        """连接到GPS设备"""
        if self.connected:
            self.logger.info("已经连接到GPS设备")
            return True
        
        try:
            # 如果没有MAC地址，尝试扫描
            if not self.mac_address:
                self.mac_address = self.scan_devices()
                if not self.mac_address:
                    self.logger.error("无法找到GPS设备")
                    return False
            
            self.logger.info(f"尝试连接到设备: {self.mac_address}")
            
            # 获取本地蓝牙适配器地址
            adapter_addr = adapter.Adapter().address

            # 创建Device对象（需要adapter_addr和device_addr）
            self.gps_device = device.Device(adapter_addr, self.mac_address)

            # 连接设备
            self.gps_device.connect()
            
            # 等待服务解析
            timeout = 10
            while not self.gps_device.services_resolved and timeout > 0:
                time.sleep(1)
                timeout -= 1
            if not self.gps_device.services_resolved:
                self.logger.error("设备服务未解析")
                return False

            # 获取服务对象
            service_obj = GATT.Service(adapter_addr, self.mac_address, self.SERVICE_UUID)
            service_obj.resolve_gatt()  # 必须解析GATT
            if not service_obj.service_props:
                self.logger.error(f"找不到服务: {self.SERVICE_UUID}")
                return False

            # 获取特征对象
            char_obj = GATT.Characteristic(adapter_addr, self.mac_address, self.SERVICE_UUID, self.CHARACTERISTIC_UUID)
            char_obj.resolve_gatt()  # 必须解析GATT
            if not char_obj.characteristic_props:
                self.logger.error(f"找不到特性: {self.CHARACTERISTIC_UUID}")
                return False

            self._gps_char = char_obj

            # 注册通知回调
            self._gps_char.add_characteristic_cb(self._gps_data_callback)
            self._gps_char.start_notify()
            # 主动读取一次特征值，确保订阅生效
            try:
                self._gps_char.read_raw_value()
            except Exception as e:
                self.logger.warning(f"主动读取特征值失败: {e}")

            self.connected = True
            self.logger.info("成功连接到GPS设备并启用通知")
            
            # 发送状态请求
            time.sleep(1)
            self.send_command("STATUS")
            
            return True
            
        except Exception as e:
            self.logger.error(f"连接GPS设备时出错: {e}")
            return False
    
    def disconnect(self):
        """断开与GPS设备的连接"""
        if self.connected and self._gps_char:
            try:
                self._gps_char.stop_notify()
                self.connected = False
                self.logger.info("已断开与GPS设备的连接")
            except Exception as e:
                self.logger.error(f"断开连接时出错: {e}")
    
    def send_command(self, command):
        """向GPS设备发送命令"""
        if not self.connected or not self._gps_char:
            self.logger.warning("设备未连接，无法发送命令")
            return False
        
        try:
            # 发送命令时，必须是字节数组（每个字节为一个元素）
            cmd_bytes = command.encode('utf-8')
            self._gps_char.write_value(list(cmd_bytes))
            self.logger.debug(f"发送命令: {command}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送命令失败: {e}")
            return False
    
    def _gps_data_callback(self, value, *args, **kwargs):
        """处理接收到的GPS数据通知"""
        try:
            if value:
                # 兼容 dbus.String、str、bytes、list 类型
                if isinstance(value, list):
                    byte_data = bytes(value)
                    message = byte_data.decode('utf-8').strip()
                elif isinstance(value, bytes):
                    message = value.decode('utf-8').strip()
                else:
                    # dbus.String 或 str
                    message = str(value).strip()
                
                current_time = datetime.now()
                
                # 尝试解析JSON格式的GPS数据
                if message.startswith('{') and message.endswith('}'):
                    try:
                        gps_data = json.loads(message)
                        self.process_gps_data(gps_data, current_time)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON解析失败: {e}")
                # 新增：解析NMEA格式
                elif message.startswith("$GPGGA"):
                    gps_data = self.parse_nmea_gpgga(message)
                    if gps_data:
                        self.process_gps_data(gps_data, current_time)
                    else:
                        self.logger.warning(f"NMEA解析失败: {message}")
                else:
                    # 处理普通响应消息
                    self.process_response(message)
        except Exception as e:
            self.logger.error(f"数据处理错误: {e}")

    def parse_nmea_gpgga(self, nmea_str):
        """解析GPGGA NMEA字符串为字典"""
        try:
            parts = nmea_str.split(',')
            if len(parts) < 10:
                return None
            # 只提取常用字段
            return {
                'valid': 'A' if parts[6] != '0' else 'V',
                'latitude': parts[2],
                'longitude': parts[4],
                'speed': '0',  # GPGGA没有速度
                'status': 'GPS_FIXED' if parts[6] != '0' else 'GPS_SEARCHING'
            }
        except Exception as e:
            self.logger.error(f"NMEA解析异常: {e}")
            return None
    
    def process_response(self, message):
        """处理GPS设备的响应消息"""
        if message == "ESP32_GPS_ONLINE" or message == "GPS_ONLINE":
            self.logger.info("GPS设备在线")
        elif message == "PONG":
            self.logger.debug("收到PING响应")
        elif message.startswith("GPS_"):
            self.logger.info(f"GPS状态: {message}")
        elif message == "org.bluez.GattCharacteristic1":
            pass  # 可能是特性对象的响应，不需要处理
        else:
            self.logger.debug(f"收到未知响应: {message}")
    
    def process_gps_data(self, gps_data, timestamp):
        """处理GPS数据并存入数据库"""
        # 检查数据状态
        if gps_data.get('status') == 'GPS_SEARCHING':
            self.logger.info("GPS正在搜索信号...")
            return
        
        if 'valid' not in gps_data:
            self.logger.warning(f"收到无效GPS数据: {gps_data}")
            return
        
        valid = gps_data.get('valid', '')
        latitude = gps_data.get('latitude', '')
        longitude = gps_data.get('longitude', '')
        raw_speed = gps_data.get('speed', '0')
        
        if valid == 'A' and latitude and longitude:
            # 转换坐标格式
            lat_decimal = self.convert_coordinate(latitude)
            lon_decimal = self.convert_coordinate(longitude)
            
            if lat_decimal is None or lon_decimal is None:
                self.logger.warning(f"坐标转换失败: {latitude}, {longitude}")
                return
            
            # 处理速度
            speed_ms = 0.0
            if raw_speed:
                try:
                    # 原始速度单位为节，转换为m/s
                    speed_knots = float(raw_speed)
                    speed_ms = speed_knots * 0.514444
                    self.logger.debug(f"速度: {speed_ms:.2f} m/s")
                except ValueError:
                    self.logger.warning(f"速度转换失败: {raw_speed}")
            
            # 更新统计数据
            self.update_stats(lat_decimal, lon_decimal, speed_ms, timestamp)
            
            # 存储到数据库
            self.save_to_database(lat_decimal, lon_decimal, speed_ms, timestamp)
            
            # 记录日志
            self.logger.info(f"GPS数据: 位置({lat_decimal:.6f}, {lon_decimal:.6f}), 速度: {speed_ms:.2f} m/s")
        else:
            self.logger.debug(f"GPS数据无效或不完整: {gps_data}")
    
    def convert_coordinate(self, coord_str):
        """转换NMEA坐标格式到十进制度数"""
        try:
            if not coord_str or len(coord_str) < 4:
                return None
                
            # DDMM.MMMM格式转换
            if '.' in coord_str:
                dot_pos = coord_str.find('.')
                degrees = int(coord_str[:dot_pos-2])
                minutes = float(coord_str[dot_pos-2:])
                return degrees + minutes / 60.0
            else:
                return float(coord_str)
        except:
            return None
    
    def update_stats(self, lat, lon, speed, timestamp):
        """更新GPS统计数据"""
        position = (lat, lon)
        self.positions.append(position)
        self.speeds.append(speed)
        self.timestamps.append(timestamp)
        
        if self.start_time is None:
            self.start_time = timestamp
        
        # 计算距离
        if self.last_position:
            try:
                distance = geodesic(self.last_position, position).kilometers
                self.total_distance += distance
            except:
                self.logger.warning("距离计算失败")
        
        self.last_position = position
        
        # 更新速度统计
        if speed > self.max_speed:
            self.max_speed = speed
        
        if self.speeds:
            self.avg_speed = sum(self.speeds) / len(self.speeds)
    
    def save_to_database(self, latitude, longitude, speed, timestamp):
        """将GPS数据保存到数据库"""
        if self.conn is None:
            self.logger.warning("数据库连接不可用，无法存储数据")
            return
            
        try:
            cursor = self.conn.cursor()
            
            # 按照表结构插入数据，speed为m/s
            sql = """
            INSERT INTO gps_data (time, longitude, latitude, speed) 
            VALUES (%s, %s, %s, %s)
            """
            
            # 执行SQL
            cursor.execute(sql, (timestamp, longitude, latitude, speed))
            self.conn.commit()
            
            self.logger.debug("GPS数据已保存到数据库")
            
        except Exception as e:
            self.logger.error(f"保存数据到数据库时出错: {e}")
    
    def get_stats(self):
        """获取统计信息"""
        if not self.positions:
            return None
            
        duration = (self.timestamps[-1] - self.start_time).total_seconds() / 3600 if len(self.timestamps) > 1 else 0
        
        return {
            'total_points': len(self.positions),
            'total_distance': self.total_distance,
            'max_speed': self.max_speed,   # 单位:m/s
            'avg_speed': self.avg_speed,   # 单位:m/s
            'duration': duration,
            'start_time': self.start_time,
            'end_time': self.timestamps[-1] if self.timestamps else None
        }
    
    def run(self):
        """主运行方法，启动GPS数据收集"""
        self.running = True
        self.logger.info("启动GPS控制器")
        
        try:
            # 连接设备
            if not self.connect():
                self.logger.error("无法连接到GPS设备，退出运行")
                return
                
            self.logger.info("GPS控制器已启动，开始接收数据")
            
            # 启动自动重连线程
            if self._reconnect_thread is None or not self._reconnect_thread.is_alive():
                self._reconnect_thread = threading.Thread(target=self._auto_reconnect_loop, daemon=True)
                self._reconnect_thread.start()
                self.logger.info("GPS设备自动重连检测线程已启动")
            
            # 启动主动轮询GPS数据线程
            polling_thread = threading.Thread(target=self._poll_gps_data, daemon=True)
            polling_thread.start()
            self.logger.info("GPS数据主动轮询线程已启动")
            
            # 启动GLib主循环
            self.mainloop = GLib.MainLoop()
            self.mainloop.run()
                
        except Exception as e:
            self.logger.error(f"GPS控制器运行时出错: {e}")
        finally:
            if self.connected:
                self.disconnect()
            self.running = False
            self.logger.info("GPS控制器已停止")

    def _poll_gps_data(self):
        """主动轮询GPS特征值"""
        while self.running and self.connected and self._gps_char:
            try:
                value = self._gps_char.read_raw_value()
                if value:
                    self._gps_data_callback(value)
            except Exception as e:
                self.logger.warning(f"主动读取GPS数据失败: {e}")
            time.sleep(2)  # 每2秒轮询一次
    
    def stop(self):
        """停止GPS控制器"""
        self.logger.info("正在停止GPS控制器...")
        self._auto_reconnect = False
        self.running = False
        
        if self.mainloop and self.mainloop.is_running():
            GLib.idle_add(self.mainloop.quit)
            
        if self.connected:
            self.disconnect()
            
        self.logger.info("GPS控制器已停止")
    
    def start(self):
        """在单独的线程中启动GPS控制器"""
        if self.running:
            self.logger.warning("GPS控制器已在运行")
            return
            
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        self.logger.info("GPS控制器在后台线程中启动")
        return thread
    
    def _auto_reconnect_loop(self):
        """自动重连检测线程"""
        while self._auto_reconnect and self.running:
            if not self.connected:
                self.logger.warning("GPS设备连接已断开，尝试重新连接")
                try:
                    self.connect()
                except Exception as e:
                    self.logger.error(f"GPS设备重连失败: {e}")
            time.sleep(5)  # 每5秒检测一次