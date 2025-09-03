#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
心率预测命令行工具
提供简单的命令行接口进行训练和预测
"""

import argparse
import os
import sys
from heart_rate_predictor import HeartRateManager, quick_train, quick_predict

def train_command(args):
    """训练命令"""
    print(f"🏃 开始训练模型...")
    print(f"训练数据: {args.data}")
    print(f"模型保存路径: {args.model}")
    
    try:
        manager = HeartRateManager(args.model)
        result = manager.train_from_csv(args.data, test_split=args.test_split)
        
        print(f"\n🎉 训练成功!")
        if result['metrics']:
            metrics = result['metrics']
            print(f"📊 模型性能:")
            print(f"  MAE: {metrics['mae']:.2f} bpm")
            print(f"  RMSE: {metrics['rmse']:.2f} bpm")
            print(f"  最大误差: {metrics['max_error']:.2f} bpm")
        
        print(f"💾 模型已保存到: {args.model}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        sys.exit(1)

def predict_command(args):
    """预测命令"""
    print(f"🔮 开始预测...")
    
    try:
        manager = HeartRateManager(args.model)
        
        # 加载模型
        if not manager.load_model():
            print(f"❌ 无法加载模型: {args.model}")
            sys.exit(1)
        
        if args.batch:
            # 批量预测
            print(f"📊 批量预测: {args.batch}")
            output_file = args.output or args.batch.replace('.csv', '_predictions.csv')
            
            df = manager.predict_from_csv(args.batch, output_file)
            print(f"✅ 批量预测完成，结果保存到: {output_file}")
            
            # 显示前几行结果
            print(f"\n📋 预测结果预览:")
            print(df.head())
            
        else:
            # 单个预测
            cadence = args.cadence
            speed = args.speed
            
            if cadence is None or speed is None:
                print("❌ 单个预测需要提供 --cadence 和 --speed 参数")
                sys.exit(1)
            
            heart_rate = manager.predict_single(cadence, speed)
            
            if heart_rate is not None:
                print(f"\n🎯 预测结果:")
                print(f"  步频: {cadence} 步/分钟")
                print(f"  速度: {speed} km/h")
                print(f"  预测心率: {heart_rate:.1f} bpm")
            else:
                print("❌ 预测失败")
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        sys.exit(1)

def info_command(args):
    """模型信息命令"""
    print(f"📋 模型信息: {args.model}")
    
    try:
        manager = HeartRateManager(args.model)
        
        if manager.load_model():
            info = manager.get_model_info()
            
            print(f"✅ 模型状态: 已加载")
            print(f"📁 模型路径: {info['model_path']}")
            print(f"🏃 训练状态: {'已训练' if info['is_trained'] else '未训练'}")
            
            if 'feature_range' in info:
                fr = info['feature_range']
                print(f"📊 特征范围:")
                print(f"  步频: {fr['min_cadence']:.1f} - {fr['max_cadence']:.1f} 步/分钟")
                print(f"  速度: {fr['min_speed']:.1f} - {fr['max_speed']:.1f} km/h")
            
            if 'target_range' in info:
                tr = info['target_range']
                print(f"🎯 心率范围: {tr['min_hr']:.1f} - {tr['max_hr']:.1f} bpm")
        else:
            print(f"❌ 无法加载模型: {args.model}")
            
    except Exception as e:
        print(f"❌ 获取模型信息失败: {e}")

def interactive_command(args):
    """交互式预测命令"""
    print(f"🎮 交互式心率预测")
    print(f"模型: {args.model}")
    
    try:
        manager = HeartRateManager(args.model)
        
        if not manager.load_model():
            print(f"❌ 无法加载模型: {args.model}")
            sys.exit(1)
        
        print(f"✅ 模型加载成功")
        print(f"💡 输入 'q' 退出程序")
        
        while True:
            try:
                print(f"\n" + "="*50)
                
                # 获取步频
                cadence_input = input("请输入步频 (步/分钟): ").strip()
                if cadence_input.lower() == 'q':
                    break
                
                # 获取速度
                speed_input = input("请输入速度 (km/h): ").strip()
                if speed_input.lower() == 'q':
                    break
                
                # 转换为数字
                try:
                    cadence = float(cadence_input)
                    speed = float(speed_input)
                except ValueError:
                    print("❌ 请输入有效的数字")
                    continue
                
                # 预测
                heart_rate = manager.predict_single(cadence, speed)
                
                if heart_rate is not None:
                    print(f"\n🎯 预测结果:")
                    print(f"  步频: {cadence:.1f} 步/分钟")
                    print(f"  速度: {speed:.1f} km/h")
                    print(f"  预测心率: {heart_rate:.1f} bpm")
                    
                    # 简单的运动强度评估
                    if heart_rate < 100:
                        intensity = "轻度运动"
                    elif heart_rate < 140:
                        intensity = "中等强度运动"
                    elif heart_rate < 170:
                        intensity = "高强度运动"
                    else:
                        intensity = "极高强度运动"
                    
                    print(f"  运动强度: {intensity}")
                else:
                    print("❌ 预测失败")
                    
            except KeyboardInterrupt:
                print(f"\n👋 程序已退出")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
        
    except Exception as e:
        print(f"❌ 交互式预测失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="心率预测命令行工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('data', help='训练数据CSV文件路径')
    train_parser.add_argument('-m', '--model', default='heart_rate_model.pkl', help='模型保存路径')
    train_parser.add_argument('-t', '--test-split', type=float, default=0.2, help='测试集比例 (默认0.2)')
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='预测心率')
    predict_parser.add_argument('-m', '--model', default='heart_rate_model.pkl', help='模型文件路径')
    predict_parser.add_argument('-c', '--cadence', type=float, help='步频 (步/分钟)')
    predict_parser.add_argument('-s', '--speed', type=float, help='速度 (km/h)')
    predict_parser.add_argument('-b', '--batch', help='批量预测CSV文件路径')
    predict_parser.add_argument('-o', '--output', help='批量预测输出文件路径')
    
    # 模型信息命令
    info_parser = subparsers.add_parser('info', help='查看模型信息')
    info_parser.add_argument('-m', '--model', default='heart_rate_model.pkl', help='模型文件路径')
    
    # 交互式预测命令
    interactive_parser = subparsers.add_parser('interactive', help='交互式预测')
    interactive_parser.add_argument('-m', '--model', default='heart_rate_model.pkl', help='模型文件路径')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'info':
        info_command(args)
    elif args.command == 'interactive':
        interactive_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
