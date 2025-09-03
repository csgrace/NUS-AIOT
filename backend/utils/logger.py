import logging
import sys
import os

class Logger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not Logger._initialized:
            # 确保utils目录存在
            os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
            
            self.logger = logging.getLogger('aiot')
            self.logger.setLevel(logging.INFO)  # 默认日志级别
            
            # 创建控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)  # 默认日志级别
            
            # 创建文件处理器
            file_handler = logging.FileHandler('aiot.log')
            file_handler.setLevel(logging.DEBUG)  # 文件始终记录所有日志
            
            # 创建格式器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            # 添加处理器到记录器
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            
            # 存储处理器的引用，便于后续修改级别
            self.console_handler = console_handler
            Logger._initialized = True
    
    def set_log_level(self, level):
        """
        设置日志级别
        level: 'debug', 'info', 'warning', 'error', 'critical'
        """
        level_dict = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        
        if level.lower() not in level_dict:
            self.logger.warning(f"无效的日志级别: {level}，使用默认级别'info'")
            level = 'info'
        
        log_level = level_dict[level.lower()]
        self.logger.setLevel(log_level)
        self.console_handler.setLevel(log_level)
        self.logger.info(f"日志级别设置为: {level.upper()}")
    
    def debug(self, msg):
        self.logger.debug(msg)
        
    def info(self, msg):
        self.logger.info(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def error(self, msg):
        self.logger.error(msg)
        
    def critical(self, msg):
        self.logger.critical(msg)

# 创建全局Logger实例
def get_logger():
    return Logger()
