#!/usr/bin/env python3
"""
测试心率预测API
"""

from heart_rate_predictor import HeartRateManager, quick_predict

def test_api():
    print("=== 测试心率预测API ===")
    
    # 测试1: 加载模型并预测
    print("\n1. 测试模型加载和预测:")
    manager = HeartRateManager("heart_rate_model.pkl")
    
    if manager.load_model():
        print("✅ 模型加载成功")
        
        # 测试不同场景
        scenarios = [
            (0, 0, "静息"),
            (90, 4, "慢走"),
            (120, 7, "快走"),
            (150, 10, "慢跑"),
            (160, 12, "跑步"),
            (180, 15, "快跑")
        ]
        
        print("\n🏃 不同运动强度心率预测:")
        for cadence, speed, activity in scenarios:
            hr = manager.predict_single(cadence, speed)
            if hr:
                print(f"  {activity:6}: 步频{cadence:3}, 速度{speed:2}km/h -> {hr:5.1f} bpm")
    
    # 测试2: 快速预测函数
    print("\n2. 测试快速预测函数:")
    try:
        hr = quick_predict(160, 12, "heart_rate_model.pkl")
        print(f"✅ 快速预测成功: 步频160, 速度12km/h -> {hr:.1f} bpm")
    except Exception as e:
        print(f"❌ 快速预测失败: {e}")
    
    # 测试3: 模型信息
    print("\n3. 测试模型信息:")
    info = manager.get_model_info()
    if info['loaded']:
        print(f"✅ 模型信息获取成功:")
        print(f"  模型路径: {info['model_path']}")
        print(f"  训练状态: {info['is_trained']}")
        if 'feature_range' in info:
            fr = info['feature_range']
            print(f"  步频范围: {fr['min_cadence']:.1f} - {fr['max_cadence']:.1f}")
            print(f"  速度范围: {fr['min_speed']:.1f} - {fr['max_speed']:.1f}")

if __name__ == "__main__":
    test_api()
