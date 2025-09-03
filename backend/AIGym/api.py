from flask import Flask, jsonify, request
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO, solutions

app = Flask(__name__)

# 全局状态
state = {
    'running': False,
    'paused': False,
    'current_exercise': 'pushup',
    'pushup_count': 0,
    'squat_count': 0
}

# 运动检测线程
class WorkoutThread(threading.Thread):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.daemon = True
        self.model = YOLO('backend/AIGym/yolo11n-pose.pt')
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        pushup_state = 'up'
        squat_state = 'up'
        last_exercise = self.state['current_exercise']
        while True:
            if not self.state['running']:
                time.sleep(0.2)
                continue
            if self.state['paused']:
                time.sleep(0.2)
                continue
            # 检查运动模式切换
            if self.state['current_exercise'] != last_exercise:
                print(f"[切换] 当前运动: {self.state['current_exercise']}")
                last_exercise = self.state['current_exercise']
            ret, frame = cap.read()
            if not ret:
                continue
            angle = None
            if self.state['current_exercise'] == 'pushup':
                results = self.model(frame)[0]
                if len(results.keypoints.data) > 0:
                    keypoints = results.keypoints.data[0].cpu().numpy()
                    right_shoulder = keypoints[6][:2] if keypoints[6][2] > 0.5 else None
                    right_elbow = keypoints[8][:2] if keypoints[8][2] > 0.5 else None
                    right_wrist = keypoints[10][:2] if keypoints[10][2] > 0.5 else None
                    angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                    if angle is not None:
                        if pushup_state == 'up' and angle < 90.0:
                            pushup_state = 'down'
                        elif pushup_state == 'down' and angle > 160.0:
                            pushup_state = 'up'
                            self.state['pushup_count'] += 1
                            print(f"[计数] 俯卧撑: {self.state['pushup_count']} 当前状态: {pushup_state}")
            else:
                results = self.model(frame)[0]
                if len(results.keypoints.data) > 0:
                    keypoints = results.keypoints.data[0].cpu().numpy()
                    right_hip = keypoints[12][:2] if keypoints[12][2] > 0.5 else None
                    right_knee = keypoints[14][:2] if keypoints[14][2] > 0.5 else None
                    right_ankle = keypoints[16][:2] if keypoints[16][2] > 0.5 else None
                    angle = self.calculate_angle(right_hip, right_knee, right_ankle)
                    if angle is not None:
                        if squat_state == 'up' and angle < 100.0:
                            squat_state = 'down'
                        elif squat_state == 'down' and angle > 160.0:
                            squat_state = 'up'
                            self.state['squat_count'] += 1
                            print(f"[计数] 深蹲: {self.state['squat_count']} 当前状态: {squat_state}")
            # 实时打印当前信息
            print(f"[状态] 当前运动: {self.state['current_exercise']} 俯卧撑: {self.state['pushup_count']} 深蹲: {self.state['squat_count']} 暂停: {self.state['paused']} 运行: {self.state['running']}")
            time.sleep(0.05)
        cap.release()
        # 移除 cv2.destroyAllWindows()
    def calculate_angle(self, a, b, c):
        if a is None or b is None or c is None:
            return None
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

workout_thread = WorkoutThread(state)
workout_thread.start()

@app.route('/start', methods=['POST'])
def start():
    state['running'] = True
    state['paused'] = False
    return jsonify({'msg': 'Started'})

@app.route('/stop', methods=['POST'])
def stop():
    # 先获取本次运动信息
    info = {
        'current_exercise': state['current_exercise'],
        'pushup_count': state['pushup_count'],
        'squat_count': state['squat_count'],
        'paused': state['paused'],
        'running': state['running']
    }
    # 清空状态
    state['running'] = False
    state['paused'] = False
    state['current_exercise'] = 'pushup'
    state['pushup_count'] = 0
    state['squat_count'] = 0
    return jsonify({'msg': 'Stopped', 'session': info})

@app.route('/pause', methods=['POST'])
def pause():
    state['paused'] = True
    return jsonify({'msg': 'Paused'})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'current_exercise': state['current_exercise'],
        'pushup_count': state['pushup_count'],
        'squat_count': state['squat_count'],
        'paused': state['paused'],
        'running': state['running']
    })

@app.route('/switch', methods=['POST'])
def switch():
    if state['current_exercise'] == 'pushup':
        state['current_exercise'] = 'squat'
    else:
        state['current_exercise'] = 'pushup'
    return jsonify({'current_exercise': state['current_exercise']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
