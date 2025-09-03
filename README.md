
# NUS-Summer-Workshop-AIOT

## 项目简介
本项目为新加坡国立大学AIoT暑期工作坊的综合实践项目，融合了物联网设备、AI模型、数据采集与分析、Web后端服务等多种技术，旨在实现运动健康数据采集、食品识别与营养分析、智能健身等功能。

## 目录结构
```
├── backend/           # 后端主目录
│   ├── main.py        # 后端主入口，管理各控制器
│    ├── BandController.py / ScaleController.py / GPSController.py / MicroBitController.py  #设备控制
│   ├── FoodHandler.py # 食品识别与营养分析
│   ├── DataBase.py    # 数据库操作
│   ├── SportDataHandler.py # 运动数据处理
│   ├── web/           # Web API 服务（Flask，含 Swagger 文档）
│   ├── AIGym/         # AI 健身推理与API
│   ├── datasets/      # 数据集（食品、运动等）
│   ├── models/        # AI模型（CNN/SVM/YOLO等）
│   └── utils/         # 工具函数与日志
├── ESP32/             # ESP32端代码（C++/Python，BLE与GPS数据采集）
├── microbits/         # micro:bit 相关脚本
├── models/            # 训练好的模型文件
├── scripts/           # 数据生成与处理脚本
├── requirements.txt   # Python依赖
```

## 主要功能模块
- **多设备数据采集**：支持手环、体重秤、GPS、micro:bit等多种IoT设备的数据采集与管理。
- **食品识别与营养分析**：通过AI模型识别食物图片，结合数据库进行营养成分分析。
- **运动健康分析**：集成AI健身（如俯卧撑/深蹲计数）、心率预测等模型。
- **Web API服务**：基于Flask，提供数据查询、上传、AI推理等RESTful接口，支持Swagger文档。
- **数据可视化与管理**：支持数据的采集、存储、分析与可视化。

## 依赖安装
项目主要依赖于Python 3.8+，核心依赖如下：
```bash
pip install -r requirements.txt
```
主要依赖包包括：Flask, flasgger, psycopg2-binary, ultralytics, torch, numpy, pandas, scikit-learn, matplotlib, bluezero, opencv-python 等。

## 快速上手
1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
2. **启动后端服务**
   ```bash
   cd backend/web
   python apiServer.py
   ```
3. **在具有推理能力的电脑上运行AI健身API**
   
   ```bash
   cd backend/AIGym
   python api.py
   ```
4. **ESP32与micro:bit端**
   - 参考 `ESP32/main.cpp` 和 `microbits/` 目录下脚本，烧录/运行于对应硬件。

## 数据与模型
- 数据集位于 `backend/datasets/`，包括食品、运动等多类原始与处理后数据。
- 训练好的模型文件位于 `models/`。
- SVM心率预测器使用方法详见 `backend/models/SVM/README_HeartRatePredictor.md`。

## 贡献与开发
欢迎参与开发与优化！如有问题请提交issue。

---
本项目为NUS暑期AIoT Workshop课程实践作品，仅供学习交流。
