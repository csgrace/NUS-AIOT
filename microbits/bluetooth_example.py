def refreshTime():
    pass

def on_uart_data_received():
    msgHandler(bluetooth.uart_read_until(serial.delimiters(Delimiters.NEW_LINE)))
bluetooth.on_uart_data_received(serial.delimiters(Delimiters.NEW_LINE),
    on_uart_data_received)

def on_bluetooth_connected():
    global blue_connected
    blue_connected = 1
bluetooth.on_bluetooth_connected(on_bluetooth_connected)

def on_bluetooth_disconnected():
    global blue_connected
    blue_connected = 0
bluetooth.on_bluetooth_disconnected(on_bluetooth_disconnected)

def on_button_pressed_a():
    bluetooth.uart_write_number(114514)
    led.plot(0, 0)
input.on_button_pressed(Button.A, on_button_pressed_a)

def msgHandler(message: str):
    command = message.split(":")
    if command[0] == "time_upd": #time:秒时间戳
        global time
        time = int(command[1])
        refreshTime()
    pass

blue_connected = 0
bluetooth.start_accelerometer_service()
bluetooth.start_uart_service()
bluetooth.set_transmit_power(7)