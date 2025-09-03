from bluezero import microbit, adapter
from gi.repository import GLib
import time
import threading
from utils.logger import get_logger


class MicroBitController:
    def __init__(self, MAC):
        bluetooth_adapter = adapter.Adapter()
        # 启动扫描
        bluetooth_adapter.start_discovery()
        print("正在扫描附近的蓝牙设备...")
        time.sleep(1)  
        bluetooth_adapter.stop_discovery()
        self.logger = get_logger()
        self.ubit = microbit.Microbit(device_addr=MAC)
        self._auto_reconnect = True  # 自动重连标志
        self._reconnect_thread = None

        
    @property
    def connected(self):
        return self.ubit.connected
    
    def accelerometer_data_received(self,x, y, z):
        #self.logger.debug(f"收到加速度计数据: X={x}, Y={y}, Z={z}")
        pass

    def uart_data_received(self,data):
        self.logger.debug(f"uart_data_received. 原始数据: {data}")
        try:
            # 处理 dbus.Array 类型
            if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                # 将 dbus.Byte 转换成字节列表
                byte_list = [int(byte) for byte in data]
                # 转换成字节对象
                byte_data = bytes(byte_list)
                # 解码成字符串
                text = byte_data.decode('utf-8').strip()
                self.logger.info(f"从 micro:bit 收到 UART 数据 (解码后): {text}")
                
                        
            elif isinstance(data, bytes):
                received_data = data.decode('utf-8').strip()
                self.logger.info(f"从 micro:bit 收到 UART 数据: {received_data}")
            else:
                received_data = str(data).strip()
                self.logger.info(f"从 micro:bit 收到 UART 数据 (转换为字符串): {received_data}")
        except (UnicodeDecodeError, ValueError) as e:
            self.logger.error(f"从 micro:bit 收到原始 UART 数据 (无法解码): {data}, 错误: {e}")

    def send_uart_message(self, message):
        """向micro:bit发送UART消息"""
        self.logger.info(f"正在向micro:bit发送: {message.strip()}")
        try:
            self.ubit.uart = message
            self.logger.info("消息已发送")
            return True
        except Exception as e:
            self.logger.error(f"发送消息时出错: {e}")
            return False

    
    def run(self):
    
        if not self.connected:
            self.logger.info("micro:bit 未连接，尝试连接...")
            try:
                self.connect()
            except Exception as e:
                self.logger.error(f"连接micro:bit失败: {e}")
                return
        time.sleep(2)  # 等待连接稳定
        try:
            if hasattr(self.ubit, '_accel_period'):
                self.ubit._accel_period.value = [20, 0] # 设置加速度计周期为20ms
        except Exception as e:
            self.logger.error(f"设置加速度计周期失败: {e}")
        
        try:
            if hasattr(self.ubit, '_accel_data'):
                try:
                    self.ubit.user_accel_cb = self.accelerometer_data_received
                    self.ubit._accel_data.add_characteristic_cb(self.ubit._decode_accel)
                    self.ubit._accel_data.start_notify()
                except Exception as notify_err:
                    self.logger.error(f"手动启用加速度计通知失败: {notify_err}")
            
            self.ubit.subscribe_accelerometer(self.accelerometer_data_received)
            self.logger.info("加速度计订阅成功")
        except Exception as e:
            self.logger.error(f"加速度计订阅失败: {e}")
            
        try:
            if hasattr(self.ubit, '_uart_tx'):
                try:
                    self.ubit.uart_tx_cb = self.uart_data_received
                    self.ubit._uart_tx.add_characteristic_cb(self.ubit._uart_read)
                    self.ubit._uart_tx.start_notify()
                    self.logger.info("手动启用UART通知")
                except Exception as notify_err:
                    self.logger.error(f"手动启用UART通知失败: {notify_err}")
            
            # 正常使用 subscribe_uart
            self.ubit.subscribe_uart(self.uart_data_received)
            self.logger.info("UART订阅成功")
        except Exception as e:
            self.logger.error(f"UART订阅失败: {e}")


        # 启动事件循环
        self.event_loop_thread = threading.Thread(target=self.ubit.run_async, daemon=True)
        self.event_loop_thread.start()
        self.logger.info("micro:bit 事件循环已在后台线程中启动。")

        # 启动自动重连检测线程
        if self._reconnect_thread is None or not self._reconnect_thread.is_alive():
            self._reconnect_thread = threading.Thread(target=self._auto_reconnect_loop, daemon=True)
            self._reconnect_thread.start()
            self.logger.info("micro:bit 自动重连检测线程已启动。")

    def _auto_reconnect_loop(self):
        """后台线程，定期检测连接状态，掉线后自动重连"""
        while self._auto_reconnect:
            if not self.connected:
                self.logger.warning("micro:bit 掉线，尝试自动重连...")
                try:
                    self.connect()
                except Exception as e:
                    self.logger.error(f"自动重连失败: {e}")
            time.sleep(5)  # 每5秒检测一次

    def stop_auto_reconnect(self):
        """停止自动重连"""
        self._auto_reconnect = False
        self.logger.info("已停止 micro:bit 自动重连检测线程。")

    def connect(self):
        bluetooth_adapter = adapter.Adapter()
        # 启动扫描
        bluetooth_adapter.start_discovery()
        print("正在扫描附近的蓝牙设备...")
        time.sleep(1)  
        bluetooth_adapter.stop_discovery()
        try:
            self.ubit.connect()
            self.logger.info("micro:bit 连接成功")
        except Exception as e:
            self.logger.error(f"连接micro:bit失败: {e}")

    def disconnect(self):
        try:
            self._auto_reconnect = False  # 断开时停止自动重连
            self.ubit.disconnect()
            self.logger.info("micro:bit 断开连接成功")
        except Exception as e:
            self.logger.error(f"断开micro:bit连接失败: {e}")



