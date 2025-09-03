import asyncio
import json
import time
import csv
import os
import webbrowser
from datetime import datetime
from bleak import BleakClient, BleakScanner
import logging
import folium
import pandas as pd
from geopy.distance import geodesic
import threading
import queue

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPSTracker:
    def __init__(self):
        self.positions = []
        self.speeds = []
        self.timestamps = []
        self.total_distance = 0.0
        self.max_speed = 0.0
        self.avg_speed = 0.0
        self.start_time = None
        self.last_position = None
        
    def add_position(self, lat, lon, speed, timestamp):
        """添加新的GPS位置"""
        if lat and lon:
            position = (lat, lon)
            self.positions.append(position)
            self.speeds.append(speed)
            self.timestamps.append(timestamp)
            
            if self.start_time is None:
                self.start_time = timestamp
            
            # 计算距离
            if self.last_position:
                distance = geodesic(self.last_position, position).kilometers
                self.total_distance += distance
            
            self.last_position = position
            
            # 更新速度统计
            if speed > self.max_speed:
                self.max_speed = speed
            
            if self.speeds:
                self.avg_speed = sum(self.speeds) / len(self.speeds)
    
    def get_stats(self):
        """获取统计信息"""
        if not self.positions:
            return None
            
        duration = (self.timestamps[-1] - self.start_time).total_seconds() / 3600 if len(self.timestamps) > 1 else 0
        
        return {
            'total_points': len(self.positions),
            'total_distance': self.total_distance,
            'max_speed': self.max_speed,
            'avg_speed': self.avg_speed,
            'duration': duration,
            'start_time': self.start_time,
            'end_time': self.timestamps[-1] if self.timestamps else None
        }

class ESP32GPSClient:
    def __init__(self):
        self.SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
        self.CHARACTERISTIC_UUID = "87654321-4321-4321-4321-cba987654321"
        self.client = None
        self.device_address = None
        self.connected = False
        self.gps_data = {}
        self.data_count = 0
        self.start_time = time.time()
        self.tracker = GPSTracker()
        self.command_queue = queue.Queue()
        
        # 创建数据存储目录
        self.data_dir = "gps_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 地图文件路径
        self.map_file = os.path.join(self.data_dir, "gps_track.html")
        
    async def scan_devices(self, timeout=10):
        """扫描ESP32设备"""
        print("🔍 扫描ESP32 GPS设备...")
        
        try:
            devices = await BleakScanner.discover(timeout=timeout)
            
            for device in devices:
                if device.name and "ESP32_GPS" in device.name:
                    print(f"✅ 找到设备: {device.name} ({device.address})")
                    self.device_address = device.address
                    return True
                    
            print("❌ 未找到ESP32 GPS设备")
            return False
            
        except Exception as e:
            print(f"❌ 扫描设备失败: {e}")
            return False
    
    async def connect(self):
        """连接到ESP32设备"""
        if not self.device_address:
            if not await self.scan_devices():
                return False
        
        try:
            print(f"📡 连接到设备: {self.device_address}")
            self.client = BleakClient(self.device_address)
            await self.client.connect()
            
            # 启用通知
            await self.client.start_notify(self.CHARACTERISTIC_UUID, self.notification_handler)
            
            self.connected = True
            print("✅ 连接成功！")
            
            # 发送初始状态请求
            await asyncio.sleep(1)
            await self.send_command("STATUS")
            
            return True
            
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        if self.client and self.connected:
            try:
                await self.client.stop_notify(self.CHARACTERISTIC_UUID)
                await self.client.disconnect()
                self.connected = False
                print("📴 已断开连接")
            except Exception as e:
                print(f"⚠️ 断开连接时出错: {e}")
    
    def notification_handler(self, sender, data):
        """处理接收到的通知数据"""
        try:
            message = data.decode('utf-8')
            current_time = datetime.now()
            
            # 尝试解析JSON格式的GPS数据
            if message.startswith('{') and message.endswith('}'):
                try:
                    gps_data = json.loads(message)
                    self.process_gps_data(gps_data, current_time)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON解析失败: {e}")
            else:
                # 处理普通响应消息
                self.process_response(message)
                
        except Exception as e:
            print(f"❌ 数据处理错误: {e}")
    
    def process_response(self, message):
        """处理普通响应消息"""
        if message == "ESP32_GPS_ONLINE":
            print("✅ ESP32 GPS系统在线")
        elif message == "PONG":
            print("🏓 Ping响应正常")
        elif message.startswith("GPS_"):
            print(f"🛰️ GPS状态: {message}")
    
    def process_gps_data(self, gps_data, timestamp):
        """处理GPS数据"""
        self.gps_data = gps_data
        self.data_count += 1
        
        # 检查数据状态
        if gps_data.get('status') == 'GPS_SEARCHING':
            self.display_searching_status(gps_data)
        elif 'valid' in gps_data:
            # 解析GPS坐标数据
            self.display_gps_info(gps_data, timestamp)
            self.save_gps_data(gps_data, timestamp)
        elif 'error' in gps_data:
            print(f"❌ GPS错误: {gps_data['error']}")
    
    def display_searching_status(self, gps_data):
        """显示搜索状态"""
        print(f"\r🔍 GPS搜索中... [{self.data_count}] ", end="", flush=True)
    
    def display_gps_info(self, gps_data, timestamp):
        """显示GPS信息 - 清晰格式"""
        valid = gps_data.get('valid', '')
        latitude = gps_data.get('latitude', '')
        longitude = gps_data.get('longitude', '')
        speed = gps_data.get('speed', '')
        
        # 清屏并显示最新信息
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🛰️" + "="*70 + "🛰️")
        print(f"                    GPS实时追踪 #{self.data_count}")
        print("🛰️" + "="*70 + "🛰️")
        
        if valid == 'A':
            # 有效定位
            lat_decimal = self.convert_coordinate(latitude)
            lon_decimal = self.convert_coordinate(longitude)
            
            if lat_decimal and lon_decimal:
                print(f"📍 状态: ✅ 定位成功")
                print(f"📍 位置: {lat_decimal:.6f}°N, {lon_decimal:.6f}°E")
                print(f"🗺️  地图: https://maps.google.com/?q={lat_decimal},{lon_decimal}")
                
                # 处理速度
                if speed:
                    try:
                        speed_knots = float(speed)
                        speed_kmh = speed_knots * 1.852
                        speed_ms = speed_kmh / 3.6
                        
                        print(f"🚗 速度: {speed_kmh:.1f} km/h ({speed_ms:.1f} m/s)")
                        
                        # 添加到追踪器
                        self.tracker.add_position(lat_decimal, lon_decimal, speed_kmh, timestamp)
                        
                        # 显示追踪统计
                        self.display_tracking_stats()
                        
                        # 更新地图
                        self.update_map()
                        
                    except:
                        print(f"🚗 速度: {speed} 节")
                else:
                    print("🚗 速度: 未知")
                    
        else:
            print(f"📍 状态: ❌ 信号弱/搜索中")
            print(f"📡 原始数据: {latitude}, {longitude}")
        
        print(f"⏰ 时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 显示统计信息
        elapsed = time.time() - self.start_time
        rate = self.data_count / elapsed if elapsed > 0 else 0
        print(f"📊 接收: {self.data_count} 条数据, 速率: {rate:.1f} 条/秒")
        
        print("="*76)
        print("💡 按 Ctrl+C 退出程序")
        print("="*76)
    
    def display_tracking_stats(self):
        """显示追踪统计"""
        stats = self.tracker.get_stats()
        if stats:
            print(f"\n📈 追踪统计:")
            print(f"   📍 记录点数: {stats['total_points']}")
            print(f"   📏 总距离: {stats['total_distance']:.2f} km")
            print(f"   🏃 最高速度: {stats['max_speed']:.1f} km/h")
            print(f"   ⚡ 平均速度: {stats['avg_speed']:.1f} km/h")
            print(f"   ⏱️  运行时间: {stats['duration']:.1f} 小时")
    
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
    
    def update_map(self):
        """更新地图显示"""
        if len(self.tracker.positions) < 1:
            return
            
        try:
            # 创建地图，以最新位置为中心
            latest_pos = self.tracker.positions[-1]
            m = folium.Map(
                location=latest_pos,
                zoom_start=15,
                tiles='OpenStreetMap'
            )
            
            # 添加路径
            if len(self.tracker.positions) > 1:
                folium.PolyLine(
                    locations=self.tracker.positions,
                    color='red',
                    weight=3,
                    opacity=0.8,
                    popup='GPS轨迹'
                ).add_to(m)
            
            # 添加起点标记
            if self.tracker.positions:
                folium.Marker(
                    self.tracker.positions[0],
                    popup='起点',
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)
            
            # 添加当前位置标记
            current_speed = self.tracker.speeds[-1] if self.tracker.speeds else 0
            folium.Marker(
                latest_pos,
                popup=f'当前位置<br>速度: {current_speed:.1f} km/h<br>时间: {datetime.now().strftime("%H:%M:%S")}',
                icon=folium.Icon(color='red', icon='record')
            ).add_to(m)
            
            # 添加统计信息
            stats = self.tracker.get_stats()
            if stats:
                stats_html = f"""
                <div style="position: fixed; 
                           top: 10px; right: 10px; width: 200px; height: 120px; 
                           background-color: white; border:2px solid grey; z-index:9999; 
                           font-size:12px; padding: 10px">
                <h4>📊 GPS统计</h4>
                <p>📍 点数: {stats['total_points']}</p>
                <p>📏 距离: {stats['total_distance']:.2f} km</p>
                <p>🏃 最高速度: {stats['max_speed']:.1f} km/h</p>
                <p>⚡ 平均速度: {stats['avg_speed']:.1f} km/h</p>
                </div>
                """
                m.get_root().html.add_child(folium.Element(stats_html))
            
            # 保存地图
            m.save(self.map_file)
            
        except Exception as e:
            print(f"❌ 地图更新失败: {e}")
    
    def save_gps_data(self, gps_data, timestamp):
        """保存GPS数据到文件"""
        try:
            # 保存到文本日志
            log_file = os.path.join(self.data_dir, "gps_log.txt")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {json.dumps(gps_data, ensure_ascii=False)}\n")
            
            # 保存到CSV文件
            csv_file = os.path.join(self.data_dir, "gps_data.csv")
            file_exists = os.path.exists(csv_file)
            
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # 写入表头
                if not file_exists:
                    writer.writerow(["时间", "状态", "纬度", "经度", "速度(km/h)", "时间戳"])
                
                # 处理速度转换
                speed_kmh = ""
                if gps_data.get('speed'):
                    try:
                        speed_kmh = float(gps_data['speed']) * 1.852
                    except:
                        speed_kmh = gps_data['speed']
                
                # 写入数据
                writer.writerow([
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    gps_data.get('valid', ''),
                    gps_data.get('latitude', ''),
                    gps_data.get('longitude', ''),
                    speed_kmh,
                    gps_data.get('timestamp', '')
                ])
                
        except Exception as e:
            print(f"❌ 保存文件失败: {e}")
    
    async def send_command(self, command):
        """发送命令到ESP32"""
        if not self.connected or not self.client:
            print("❌ 设备未连接")
            return False
        
        try:
            await self.client.write_gatt_char(self.CHARACTERISTIC_UUID, command.encode())
            await asyncio.sleep(0.5)
            return True
            
        except Exception as e:
            print(f"❌ 发送命令失败: {e}")
            return False
    
    def open_map(self):
        """打开地图文件"""
        if os.path.exists(self.map_file):
            webbrowser.open(f'file://{os.path.abspath(self.map_file)}')
            print(f"🗺️ 地图已在浏览器中打开: {self.map_file}")
        else:
            print("❌ 地图文件不存在")
    
    def export_data(self):
        """导出数据分析"""
        try:
            csv_file = os.path.join(self.data_dir, "gps_data.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                
                # 生成分析报告
                report_file = os.path.join(self.data_dir, "gps_analysis.txt")
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write("GPS数据分析报告\n")
                    f.write("="*50 + "\n")
                    f.write(f"数据记录数: {len(df)}\n")
                    f.write(f"有效定位数: {len(df[df['状态'] == 'A'])}\n")
                    
                    if '速度(km/h)' in df.columns:
                        valid_speeds = pd.to_numeric(df['速度(km/h)'], errors='coerce').dropna()
                        if len(valid_speeds) > 0:
                            f.write(f"最高速度: {valid_speeds.max():.2f} km/h\n")
                            f.write(f"平均速度: {valid_speeds.mean():.2f} km/h\n")
                    
                    f.write(f"数据时间范围: {df['时间'].iloc[0]} - {df['时间'].iloc[-1]}\n")
                
                print(f"📊 数据分析报告已生成: {report_file}")
            else:
                print("❌ 没有找到GPS数据文件")
                
        except Exception as e:
            print(f"❌ 导出数据失败: {e}")

# 主程序
async def main():
    client = ESP32GPSClient()
    
    print("🚀 ESP32 GPS实时追踪系统启动...")
    print("📦 确保已安装: pip install bleak folium pandas geopy")
    print("-" * 60)
    
    # 连接设备
    if not await client.connect():
        print("❌ 无法连接到设备，程序退出")
        return
    
    try:
        print("✅ 系统启动成功！")
        print("💡 GPS数据将实时显示，地图自动更新")
        print("🗺️ 地图文件: " + client.map_file)
        print("📁 数据目录: " + client.data_dir)
        print("-" * 60)
        
        # 主循环 - 持续接收数据
        while True:
            await asyncio.sleep(1)
            
            # 每30秒自动打开地图（可选）
            if client.data_count > 0 and client.data_count % 30 == 0:
                if len(client.tracker.positions) > 0:
                    print("\n🗺️ 自动打开地图...")
                    client.open_map()
                    
    except KeyboardInterrupt:
        print("\n\n👋 程序被中断")
        
        # 生成最终报告
        print("📊 生成最终数据分析...")
        client.export_data()
        
        if len(client.tracker.positions) > 0:
            print("🗺️ 打开最终轨迹地图...")
            client.open_map()
            
    except Exception as e:
        print(f"❌ 程序错误: {e}")
    finally:
        await client.disconnect()
        print("👋 程序结束")

if __name__ == "__main__":
    print("🌟 ESP32 GPS实时追踪系统")
    print("🎯 功能: 实时显示位置、速度、绘制轨迹地图")
    
    # 检查依赖
    try:
        import folium
        import pandas as pd
        from geopy.distance import geodesic
        print("✅ 所有依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("📦 请安装: pip install folium pandas geopy")
        exit(1)
    
    # 运行主程序
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
