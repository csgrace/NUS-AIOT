from dotenv import load_dotenv
import os
from BandController import BandController
from utils.logger import get_logger
import time
import signal
import sys
import threading
import importlib.util
import subprocess
# 获取全局日志实例
logger = get_logger()

# 设置日志级别 (debug, info, warning, error, critical)
# 在开发和调试时可以使用debug级别
logger.set_log_level('debug')

# 正式运行时可以使用info级别
# logger.set_log_level('info')

# 可以在运行中随时更改日志级别
# logger.set_log_level('warning')  # 只显示警告和错误

# 全局变量，用于存储控制器实例
band = None
scale = None
gps = None
db = None
api_server_process = None

def start_api_server():
    """在单独的进程中启动API服务器"""
    global api_server_process
    logger.info("正在启动API服务器...")
    try:
        # 获取apiServer.py的绝对路径
        api_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'apiServer.py')
        
        # 使用子进程启动API服务器
        import subprocess
        
        # 设置环境变量，确保子进程能访问相同的环境
        env = os.environ.copy()
        
        # 启动子进程
        # 重定向标准输出和错误到父进程
        api_process = subprocess.Popen(
            [sys.executable, api_server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env
        )
        
        # 返回进程对象，以便后续控制
        logger.info("API服务器进程已启动")
        return api_process
        
    except Exception as e:
        logger.error(f"启动API服务器失败: {e}")
        return None

def cleanup_and_exit(signum=None, frame=None):
    """清理资源并退出程序"""
    logger.info("正在关闭所有服务...")
    
    # 停止所有服务
    if gps:
        try:
            gps.stop()
            logger.info("GPS服务已停止")
        except Exception as e:
            logger.error(f"停止GPS服务时出错: {e}")
    
    if band:
        try:
            # 如果BandController有stop方法，调用它
            if hasattr(band, 'stop'):
                band.stop()
            logger.info("Band服务已停止")
        except Exception as e:
            logger.error(f"停止Band服务时出错: {e}")
    
    if scale:
        try:
            # 如果ScaleController有stop方法，调用它
            if hasattr(scale, 'stop'):
                scale.stop()
            logger.info("Scale服务已停止")
        except Exception as e:
            logger.error(f"停止Scale服务时出错: {e}")
    
    # 停止API服务器进程
    global api_server_process
    if api_server_process:
        try:
            # 发送终止信号
            import signal as sig
            api_server_process.send_signal(sig.SIGTERM)
            
            # 等待进程终止（最多等待5秒）
            try:
                api_server_process.wait(timeout=5)
                logger.info("API服务器进程已正常终止")
            except subprocess.TimeoutExpired:
                # 如果超时，强制终止
                api_server_process.kill()
                logger.warning("API服务器进程无响应，已强制终止")
                
        except Exception as e:
            logger.error(f"终止API服务器进程时出错: {e}")
    
    # 关闭数据库连接
    if db and hasattr(db, 'close'):
        try:
            db.close()
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接时出错: {e}")
    
    logger.info("所有服务已停止，退出程序")
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, cleanup_and_exit)  # Ctrl+C
signal.signal(signal.SIGTERM, cleanup_and_exit)  # 终止信号

if __name__ == "__main__":
    load_dotenv()
    logger.info("Welcome to use P.U.L.S.E.!")
    # 解析数据库配置
    db_params = {
        'dbname': os.getenv('DB_NAME', 'aiot'),
        'user': os.getenv('DB_USER', 'group4'),
        'password': os.getenv('DB_PASSWORD', 'groupgroup4'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432')
    }
    # 启动数据库
    from DataBase import DataBase
    db = DataBase(db_params=db_params)
    time.sleep(1)  # 等待数据库连接稳定
    try:
        # #启动Band服务    
        BAND_MAC = os.getenv("BAND_MAC")
        band = BandController(MAC=BAND_MAC, conn=db.conn)
        band.run()  # 启动 BandController 的数据收集和处理
        logger.info("Band service started successfully.")
    except Exception as e:
        logger.error(f"Failed to start Band service: {e}")


    # 启动 Scale 服务
    try:
        SCALE_MAC = os.getenv("SCALE_MAC")
        from ScaleController import ScaleController
        scale = ScaleController(MAC=SCALE_MAC, conn=db.conn)
        scale.run()  # 启动 ScaleController 的数据收集和处理
        logger.info("Scale service started successfully.")
    except Exception as e:
        logger.error(f"Failed to start Scale service: {e}")

    # 启动GPS 服务
    try:
        GPS_MAC = os.getenv("GPS_MAC")
        print(f"GPS MAC Address: {GPS_MAC}")
        from GPSController import GPSController
        gps = GPSController(MAC=GPS_MAC, conn=db.conn)
        gps.run()  # 启动 GPSController 的数据收集和处理
        logger.info("GPS service started successfully.")
    except Exception as e:
        logger.error(f"Failed to start GPS service: {e}")

    # 启动API服务器
    try:
        api_server_process = start_api_server()
        if api_server_process:
            logger.info("API服务器进程已启动")
        else:
            logger.error("API服务器进程启动失败")
    except Exception as e:
        logger.error(f"启动API服务器进程失败: {e}")

    try:
        while True:
            try:
                # 用户输入以发送 UART 消息
                command = input("请输入要发送的消息 (或输入 'exit' 退出): ")
                if command.lower() == 'exit':
                    logger.info("退出程序")
                    break
                command += "\n"  # 添加换行符以确保消息正确发送
            
            except KeyboardInterrupt:
                logger.info("\n程序被中断，退出...")
                break
            except Exception as e:
                logger.error(f"发生错误: {e}")
    finally:
        # 确保在退出时清理资源
        cleanup_and_exit()







