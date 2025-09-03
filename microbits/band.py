def refreshTime():
    bluetooth.uart_write_string("getTime")

def on_uart_data_received():
    msgHandler(bluetooth.uart_read_until(serial.delimiters(Delimiters.NEW_LINE)))
bluetooth.on_uart_data_received(serial.delimiters(Delimiters.NEW_LINE),
    on_uart_data_received)

def on_microbit_id_button_a_evt_click():
    global stage
    if stage < 3:
        stage += 1
    else:
        stage = 1
control.on_event(EventBusSource.MICROBIT_ID_BUTTON_A,
    EventBusValue.MICROBIT_BUTTON_EVT_CLICK,
    on_microbit_id_button_a_evt_click)

def on_bluetooth_connected():
    global blue_connected
    blue_connected = 1
    basic.show_icon(IconNames.YES)
    refreshTime()
bluetooth.on_bluetooth_connected(on_bluetooth_connected)

def on_bluetooth_disconnected():
    global blue_connected
    basic.show_icon(IconNames.SAD)
    blue_connected = 0
bluetooth.on_bluetooth_disconnected(on_bluetooth_disconnected)

# 快速校准 (如果需要，可以在启动时调用)
def quick_calibrate():
    global total, baseline
    for index in range(50):
        total += pins.analog_read_pin(AnalogPin.P0)
        basic.pause(20)
    baseline = Math.idiv(total, 50)

def on_microbit_id_button_ab_evt_click():
    global stage
    stage = 1
control.on_event(EventBusSource.MICROBIT_ID_BUTTON_AB,
    EventBusValue.MICROBIT_BUTTON_EVT_CLICK,
    on_microbit_id_button_ab_evt_click)

def msgHandler(message: str):
    global time, step
    command = message.split(":")
    if command[0] == "time_upd":
        # time:秒时间戳
        time = int(command[1])
    elif command[0] == "getHeartRate":
        # 获取心率
        msg = "HeartRate:" + ("" + str(Math.round(heart_rate))) + "\n"
        bluetooth.uart_write_string(msg)
    elif command[0] == "StepUpd":
        step = int(command[1])
    elif command[0] == "standAnnounce":
        basic.clear_screen()
        music.play(music.create_sound_expression(WaveShape.SINE,
                1871,
                1,
                217,
                0,
                267,
                SoundExpressionEffect.NONE,
                InterpolationCurve.LINEAR),
            music.PlaybackMode.UNTIL_DONE)
        basic.show_string("Stand Time!")
    elif command[0] == "HRWarning":
        music._play_default_background(music.built_in_playable_melody(Melodies.WAWAWAWAA),
            music.PlaybackMode.IN_BACKGROUND)
        basic.show_string("HR Overflow")
last_beat_time = 0
in_beat = False
current_time = 0
sensor_val = 0
second = 0
minute = 0
hour = 0
beat_count_in_window = 0
window_start_time = 0
current_time2 = 0
heart_rate = 0
time = 0
total = 0
blue_connected = 0
step = 0
stage = 0
baseline = 0
last_update_time = 0
# 存储心跳间隔的数组
# 全局变量
beat_timestamps: List[number] = []
time_string = ""
bluetooth.start_accelerometer_service()
bluetooth.set_transmit_power(7)
bluetooth.start_uart_service()
baseline = 512
threshold_offset = 50
# 常量
WINDOW_SIZE_MS = 10000
# 10秒滑动窗口
UPDATE_INTERVAL_MS = 1000
# 在启动时进行校准
quick_calibrate()
basic.pause(500)
basic.clear_screen()
stage = 1
# FSM
step = 0

def on_every_interval():
    refreshTime()
loops.every_interval(10000, on_every_interval)

def on_forever():
    global current_time2, window_start_time, beat_count_in_window, heart_rate
    current_time2 = input.running_time()
    # --- 滑动窗口 (修正后) ---
    # 直接修改全局列表，而不是重新赋值
    # 从列表开头开始检查，移除所有窗口外（过期）的时间戳
    window_start_time = current_time2 - WINDOW_SIZE_MS
    while len(beat_timestamps) > 0 and beat_timestamps[0] < window_start_time:
        beat_timestamps.pop(0)
    # --- 计算BPM (改进后，使用平均间隔以获得更平滑的读数) ---
    beat_count_in_window = len(beat_timestamps)
    if beat_count_in_window >= 3:
        # 至少需要3个心跳来计算一个稳定的平均间隔
        # 计算窗口内第一个和最后一个心跳之间的时间跨度
        total_duration_of_intervals = beat_timestamps[beat_count_in_window - 1] - beat_timestamps[0]
        # 计算有多少个间隔
        number_of_intervals = beat_count_in_window - 1
        if number_of_intervals > 0:
            # 计算平均间隔
            avg_interval = total_duration_of_intervals / number_of_intervals
            # 根据平均间隔计算BPM
            if avg_interval > 0:
                bpm = 60000 / avg_interval
            else:
                bpm = 0
        else:
            # 避免除以零
            # 只有一个点，无法计算间隔
            bpm = 0
    else:
        # 如果样本不足，则使用旧的计数方法，以在开始时提供快速反馈
        bpm = beat_count_in_window * 6
    # --- 合理性检查和显示 (阻塞操作) ---
    if 40 <= bpm and bpm <= 200:
        heart_rate = bpm
    else:
        pass
    # --- 发送计算结果到串口 ---
    serial.write_value("bpm", bpm)
    serial.write_value("beats_in_window", beat_count_in_window)
    # 控制这个循环的频率为大约每秒一次
    basic.pause(UPDATE_INTERVAL_MS)
basic.forever(on_forever)

def on_forever2():
    # 步数刷新
    bluetooth.uart_write_string("getStep")
    basic.pause(5000)
basic.forever(on_forever2)

def on_forever3():
    global hour, minute, second, time_string
    # 显示函数
    if stage == 1:
        basic.clear_screen()
        hour = Math.idiv(time, 3600) % 24
        minute = Math.idiv(time, 60) % 60
        second = time % 60
        # time_string = " " + ("" + str(hour)) + ":" + ("" + str(minute)) + ":" + ("" + str(second))
        time_string = "h" + ("" + str(hour)) + ":" + ("" + str(minute))
        basic.pause(5000)
        basic.show_string(time_string, 100)
    elif stage == 2:
        # 显示当前心率
        # 显示当前心率
        basic.clear_screen()
        HR_string = "HR:" + ("" + str(Math.round(heart_rate)))
        basic.show_string(HR_string, 100)
        basic.pause(5000)
    elif stage == 3:
        # 显示当前步数
        basic.clear_screen()
        Step_str = "STEP:" + ("" + str(step))
        basic.show_string(Step_str, 100)
        basic.pause(5000)
basic.forever(on_forever3)

def on_forever4():
    global sensor_val, current_time, in_beat, last_beat_time
    # 1. 连续读取传感器并发送到串口
    sensor_val = pins.analog_read_pin(AnalogPin.P0)
    serial.write_value("sensor", sensor_val)
    # 2. 使用上升沿检测心跳
    current_time = input.running_time()
    if sensor_val > baseline + threshold_offset:
        if not (in_beat):
            in_beat = True
            # 防抖/不应期
            if current_time - last_beat_time > 300:
                last_beat_time = current_time
                beat_timestamps.append(current_time)
    elif sensor_val < baseline:
        in_beat = False
basic.forever(on_forever4)
