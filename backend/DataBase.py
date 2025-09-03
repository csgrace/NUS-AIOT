import psycopg2
from datetime import datetime
from utils.logger import get_logger

class DataBase:
    def __init__(self, db_params=None):
        self.logger = get_logger()
        self.db_params = db_params or {
            'dbname': 'aiot',
            'user': 'group4',
            'password': 'groupgroup4',
            'host': 'localhost',
            'port': '5432'
        }
        self.conn = None
        self._init_db_connection()
        self.create_tables()

    def _init_db_connection(self):
        try:
            self.conn = psycopg2.connect(**self.db_params)
            self.logger.info("成功连接到 PostgreSQL 数据库")
        except Exception as e:
            self.logger.error(f"连接数据库失败: {e}")
            self.conn = None

    def create_tables(self):
        if self.conn is None:
            self.logger.error("数据库连接不可用，无法创建表")
            return

        try:
            with self.conn.cursor() as cursor:
                # 创建加速度数据表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS band_acceleration (
                        time TIMESTAMP,
                        x REAL,
                        y REAL,
                        z REAL
                    )
                ''')
                
                # 创建心率数据表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS band_heart_rate (
                        time TIMESTAMP,
                        heart_rate REAL
                    )
                ''')
                
                # 创建步数数据表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS band_steps (
                        time TIMESTAMP,
                        step_count INTEGER
                    )
                ''')
                
                self.conn.commit()
                self.logger.info("数据库表已创建或已存在。")
        except Exception as e:
            self.logger.error(f"创建表时出错: {e}")
            
    