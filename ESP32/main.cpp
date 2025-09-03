#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <HardwareSerial.h>

// BLE服务和特征UUID定义
#define SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "87654321-4321-4321-4321-cba987654321"

// GPS串口配置
HardwareSerial gpsSerial(1); // 使用Serial1
#define GPS_RX_PIN 16
#define GPS_TX_PIN 17

BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;
String gpsData = "";

// BLE服务器回调类
class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("设备已连接");
  }

  void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("设备已断开连接");
      // 重新开始广播
      BLEDevice::startAdvertising();
  }
};

// BLE特征回调类（处理客户端写入数据）
class MyCharacteristicCallbacks: public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
      std::string value = pCharacteristic->getValue();
      if (value.length() > 0) {
          Serial.println("接收到客户端数据: " + String(value.c_str()));
      }
  }
};

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 BLE GPS 服务器启动中...");
  
  // 初始化GPS串口
  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.println("GPS串口初始化完成 (RX: GPIO16, TX: GPIO17)");
  
  // 初始化BLE设备
  BLEDevice::init("ESP32-GPS-Server");
  
  // 创建BLE服务器
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  
  // 创建BLE服务
  BLEService *pService = pServer->createService(SERVICE_UUID);
  
  // 创建BLE特征
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_WRITE |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );
  
  // 设置特征回调
  pCharacteristic->setCallbacks(new MyCharacteristicCallbacks());
  
  // 添加描述符用于通知
  pCharacteristic->addDescriptor(new BLE2902());
  
  // 启动服务
  pService->start();
  
  // 配置广播
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x0);
  BLEDevice::startAdvertising();
  
  Serial.println("BLE服务器已启动，等待连接...");
  Serial.println("设备名称: ESP32-GPS-Server");
  Serial.println("服务UUID: " + String(SERVICE_UUID));
}

void loop() {
  // 读取GPS数据
  if (gpsSerial.available()) {
      String newGpsData = gpsSerial.readStringUntil('\n');
      newGpsData.trim();
      
      if (newGpsData.length() > 0 && newGpsData.startsWith("$")) {
          gpsData = newGpsData;
          Serial.println("GPS数据: " + gpsData);
          
          // 如果有设备连接，发送GPS数据
          if (deviceConnected && pCharacteristic != NULL) {
              pCharacteristic->setValue(gpsData.c_str());
              pCharacteristic->notify();
              Serial.println("GPS数据已发送到BLE客户端");
          }
      }
  }
  
  // 模拟GPS数据（当没有GPS模块时用于测试）
  static unsigned long lastTime = 0;
  if (millis() - lastTime > 5000) { // 每5秒发送一次模拟数据
      lastTime = millis();
      
      // 模拟NMEA格式的GPS数据
      String simulatedGPS = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47";
      
      if (deviceConnected && pCharacteristic != NULL) {
          pCharacteristic->setValue(simulatedGPS.c_str());
          pCharacteristic->notify();
          Serial.println("模拟GPS数据已发送: " + simulatedGPS);
      }
  }
  
  delay(100);
}