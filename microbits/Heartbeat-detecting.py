# 检测心率  -makecode版
beat_count = 0
measuring = False
start_time = 0
last_beat_time = 0
baseline = 512
threshold_offset = 50
recent_intervals: List[number] = []
last_display_time = 0

# 显示控制变量
display_queue: List[str] = []
last_feedback_time = 0

# 60秒统计相关变量
beat_count_60s = 0
period_start_time = 0
last_60s_display_time = 0

# 状态指示变量
status_blink_time = 0
show_status = False

# 存储最后一次20秒显示的心率值
last_20s_heart_rate = 0

def sum_array(arr):
    """自定义求和函数"""
    total = 0
    for item in arr:
        total += item
    return total

def calculate_20s_heart_rate():
    """20秒显示使用的心率计算方法"""
    global recent_intervals, beat_count, start_time, last_20s_heart_rate
    
    current_time = input.running_time()
    elapsed = (current_time - start_time) / 1000
    
    # 必须满足20秒显示的条件
    if elapsed > 30 and beat_count > 15 and len(recent_intervals) >= 3:
        
        # 使用最近的有效心跳间隔
        if len(recent_intervals) >= 20:
            use_intervals = recent_intervals[-20:]  # 使用最近20次
        else:
            use_intervals = recent_intervals
        
        # 过滤异常值（去除过快或过慢的心跳）
        filtered_intervals = []
        for interval in use_intervals:
            if 300 <= interval <= 2000:  # 对应30-200 BPM
                filtered_intervals.append(interval)
        
        if len(filtered_intervals) >= 3:
            avg_interval = sum_array(filtered_intervals) / len(filtered_intervals)
            bpm = Math.round(60000 / avg_interval)
            
            # 合理性检查
            if 40 <= bpm <= 200:
                last_20s_heart_rate = bpm  # 保存这个值
                return bpm
    
    # 如果无法计算新值，返回上次的值
    return last_20s_heart_rate

def quick_calibrate():
    """快速校准"""
    global baseline
    total = 0
    for i in range(10):
        total += pins.analog_read_pin(AnalogPin.P0)
        basic.pause(10)
    baseline = total // 10

def show_status_indicator(show: bool):
    """状态指示器"""
    if show:
        led.plot(0, 0)
    else:
        led.unplot(0, 0)

def on_button_a():
    """快速启动监测"""
    global beat_count, measuring, start_time, last_beat_time, recent_intervals
    global last_display_time, display_queue, beat_count_60s, period_start_time
    global last_60s_display_time, status_blink_time, show_status, last_20s_heart_rate
    
    quick_calibrate()
    
    # 重置所有计数器
    beat_count = 0
    beat_count_60s = 0
    measuring = True
    start_time = input.running_time()
    period_start_time = start_time
    last_beat_time = 0
    recent_intervals = []
    last_display_time = 0
    last_60s_display_time = 0
    display_queue = []
    status_blink_time = 0
    show_status = False
    last_20s_heart_rate = 0
    
    # 启动提示
    basic.show_icon(IconNames.HEART)
    basic.pause(200)
    basic.clear_screen()

def on_button_b():
    """显示与20秒自动显示相同的心率"""
    global display_queue
    
    if measuring:
        display_queue.append("show_20s_heart_rate")
    else:
        basic.show_string("OFF")
        basic.pause(500)
        basic.clear_screen()

def on_button_ab():
    """停止/重启切换"""
    global measuring, display_queue
    
    if measuring:
        measuring = False
        display_queue.append("stop_msg")
    else:
        on_button_a()

input.on_button_pressed(Button.A, on_button_a)
input.on_button_pressed(Button.B, on_button_b)
input.on_button_pressed(Button.AB, on_button_ab)

def process_display_queue(current_time):
    """处理显示队列"""
    global display_queue, last_feedback_time, beat_count, beat_count_60s
    global recent_intervals, measuring, start_time
    
    if len(display_queue) > 0 and (current_time - last_feedback_time) > 150:
        display_item = display_queue[0]
        
        if display_item == "beat":
            # 快速心跳反馈
            basic.show_icon(IconNames.SMALL_HEART)
            music.ring_tone(800)
            basic.pause(25)
            music.stop_all_sounds()
            basic.clear_screen()
            
        elif display_item == "count_60s":
            # 60秒计数显示
            basic.show_number(beat_count_60s)
            basic.pause(800)
            basic.clear_screen()
            
        elif display_item == "show_20s_heart_rate":
            # B键显示：使用与20秒自动显示相同的算法
            elapsed = (current_time - start_time) / 1000
                      
            # 显示心率（使用20秒显示的算法）
            bpm = calculate_20s_heart_rate()
            if bpm > 0:
                basic.show_number(bpm)
                basic.pause(1000)
            
            basic.clear_screen()
            
        elif display_item == "stop_msg":
            basic.show_string("STOP")
            basic.pause(500)
            basic.show_string("T:")
            basic.pause(200)
            basic.show_number(beat_count)
            basic.pause(800)
            
            # 显示最终心率
            final_bpm = calculate_20s_heart_rate()
            if final_bpm > 0:
                basic.show_number(final_bpm)
                basic.pause(1000)
            
            basic.clear_screen()
        
        # 移除已处理的显示项
        temp_queue = []
        for i in range(1, len(display_queue)):
            temp_queue.append(display_queue[i])
        display_queue = temp_queue
        
        last_feedback_time = current_time
        return True
    return False

def forever():
    """高速检测主循环"""
    global beat_count, measuring, last_beat_time, baseline, threshold_offset
    global recent_intervals, last_display_time, display_queue, beat_count_60s
    global period_start_time, last_60s_display_time, status_blink_time, show_status
    
    current_time = input.running_time()
    
    if measuring:
        sensor_val = pins.analog_read_pin(AnalogPin.P0)
        
        # 心跳检测
        if abs(sensor_val - baseline) > threshold_offset and (current_time - last_beat_time) > 300:
            beat_count += 1
            beat_count_60s += 1
            
            # 计算心跳间隔
            if last_beat_time > 0:
                interval = current_time - last_beat_time
                recent_intervals.append(interval)
                
                # 保持数组长度
                if len(recent_intervals) > 12:
                    temp_intervals = []
                    for i in range(1, len(recent_intervals)):
                        temp_intervals.append(recent_intervals[i])
                    recent_intervals = temp_intervals
            
            last_beat_time = current_time
            display_queue.append("beat")
        
        # 60秒统计显示
        elapsed_period = (current_time - period_start_time) / 1000
        if elapsed_period >= 60 and (current_time - last_60s_display_time) > 2000:
            if beat_count_60s > 0:
                display_queue.append("count_60s")
                last_60s_display_time = current_time
            
            beat_count_60s = 0
            period_start_time = current_time
        
        # 处理显示队列
        display_processed = process_display_queue(current_time)
        
        # ⭐ 20秒定期心率显示（
        if not display_processed:
            elapsed = (current_time - start_time) / 1000
            if elapsed > 30 and beat_count > 15 and (current_time - last_display_time) > 20000:
                # 使用相同的算法计算心率
                bpm = calculate_20s_heart_rate()
                if bpm > 0:
                    basic.show_number(bpm % 100)  # 显示后两位数字
                    basic.pause(400)
                    basic.clear_screen()
                    last_display_time = current_time
        
        # 状态指示灯
        if (current_time - status_blink_time) > 5000:
            show_status_indicator(not show_status)
            show_status = not show_status
            status_blink_time = current_time
    
    else:
        # 待机状态
        process_display_queue(current_time)
        
        if show_status:
            show_status_indicator(False)
            show_status = False
        
        if current_time % 3000 < 100:
            basic.show_icon(IconNames.HEART)
        elif current_time % 3000 < 200:
            basic.clear_screen()
    
    basic.pause(8)

basic.forever(forever)
