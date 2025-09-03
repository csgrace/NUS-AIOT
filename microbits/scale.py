def on_microbit_id_button_b_evt_click():
    HX711.tare(10)
control.on_event(EventBusSource.MICROBIT_ID_BUTTON_B,
    EventBusValue.MICROBIT_BUTTON_EVT_CLICK,
    on_microbit_id_button_b_evt_click)

# 蓝牙事件处理

def on_uart_data_received():
    msgHandler(bluetooth.uart_read_until(serial.delimiters(Delimiters.NEW_LINE)))
bluetooth.on_uart_data_received(serial.delimiters(Delimiters.NEW_LINE),
    on_uart_data_received)

def on_microbit_id_button_a_evt_click():
    global mode
    if mode == 1:
        mode = 0
    else:
        mode = mode + 1
control.on_event(EventBusSource.MICROBIT_ID_BUTTON_A,
    EventBusValue.MICROBIT_BUTTON_EVT_CLICK,
    on_microbit_id_button_a_evt_click)

def on_bluetooth_connected():
    global blue_connected
    blue_connected = 1
    basic.show_icon(IconNames.YES)
    basic.clear_screen()
bluetooth.on_bluetooth_connected(on_bluetooth_connected)

def on_bluetooth_disconnected():
    global blue_connected
    basic.show_icon(IconNames.SAD)
    blue_connected = 0
    basic.clear_screen()
bluetooth.on_bluetooth_disconnected(on_bluetooth_disconnected)

# 蓝牙消息处理
def msgHandler(message: str):
    global time, water_amount
    command = message.split(":")
    if command[0] == "time_upd":
        time = int(command[1])
    elif command[0] == "getWeight":
        mode_str = ""
        if mode == 1:
            mode_str = "f"
        else:
            mode_str = "w"
        msg = "Weight:" + ("" + str(Math.round(current_weight))) + ":" + mode_str
        bluetooth.uart_write_line(msg)
    elif command[0] == "calibrate":
        HX711.tare(10)
    elif command[0] == "drinkAnnounce":
        music.play(music.builtin_playable_sound_effect(soundExpression.happy),
            music.PlaybackMode.UNTIL_DONE)
        basic.show_string("Water Time! Drink Some Water!")
    elif command[0] == "todayWater":
        water_in = int(command[1])
        if water_in > water_amount:
            water_amount = water_in
water_amount = 0
current_weight = 0
time = 0
blue_connected = 0
mode = 0
bluetooth.set_transmit_power(7)
bluetooth.start_accelerometer_service()
bluetooth.start_uart_service()
HX711.SetPIN_DOUT(DigitalPin.P0)
HX711.SetPIN_SCK(DigitalPin.P1)
HX711.set_scale(0)
HX711.begin()
HX711.tare(10)
HX711.set_scale(1049.77)
# 主循环 - 重量读取

def on_forever():
    global current_weight
    current_weight = HX711.get_units(10)
basic.forever(on_forever)

def on_forever2():
    if mode == 0:
        basic.show_string("W" + ("" + str(water_amount)) + "g")
        basic.pause(100)
    else:
        basic.show_string("F")
        basic.pause(100)
basic.forever(on_forever2)
