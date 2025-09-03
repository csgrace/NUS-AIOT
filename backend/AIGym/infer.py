from ultralytics import YOLO, solutions
import cv2
import numpy as np

# 加载 YOLOv11n-pose 模型
model = YOLO('backend/AIGym/yolo11n-pose.pt')

# 初始化俯卧撑检测
# 使用右肩（6），右肘（8），右腕（10）关键点
pushup_gym = solutions.AIGym(
    model=model,
    line_width=2,
    show=False,  # 禁用内部显示，由我们控制
    kpts=[6, 8, 10],  # 右肩，右肘，右腕
    up_angle=160.0,   # 手臂伸直时的角度阈值
    down_angle=80.0   # 到达最低点时的角度阈值
)

# 初始化深蹲检测
# 使用右髋（12），右膝（14），右踝（16）关键点
squat_gym = solutions.AIGym(
    model=model,
    line_width=2,
    show=False,  # 禁用内部显示，由我们控制
    kpts=[12, 14, 16],  # 右髋，右膝，右踝
    up_angle=170.0,     # 站立时的角度阈值
    down_angle=90.0     # 深蹲时的角度阈值
)

# 设置摄像头（树莓派官方摄像头建议使用 CAP_V4L2，或直接用 0）
cap = cv2.VideoCapture(0)  # 如遇问题可尝试 cv2.VideoCapture(0, cv2.CAP_V4L2)
assert cap.isOpened(), "Camera not accessible, please check the connection and ensure the camera is enabled in raspi-config."
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # 可根据实际摄像头支持的分辨率调整
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)  # 可根据实际摄像头支持的分辨率调整

# 初始化运动模式和计数器
current_exercise = "pushup"  # 默认从俯卧撑模式开始
pushup_count = 0
squat_count = 0
pushup_state = "up"  # 初始状态设为"up"(顶部位置)
squat_state = "up"   # 初始状态设为"up"(顶部位置)

# 定义角度计算函数
def calculate_angle(a, b, c):
    """
    计算由三个点形成的角度
    """
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

# 创建窗口
cv2.namedWindow("Workout Monitoring", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Unable to read video frame, exiting.")
        break

    # 根据当前模式处理帧
    if current_exercise == "pushup":
        # 处理俯卧撑
        results = model(frame)[0]
        
        # 检查是否检测到姿势
        if len(results.keypoints.data) > 0:
            # 获取关键点
            keypoints = results.keypoints.data[0].cpu().numpy()
            
            # 提取右肩、右肘和右腕的坐标
            right_shoulder = keypoints[6][:2] if keypoints[6][2] > 0.5 else None
            right_elbow = keypoints[8][:2] if keypoints[8][2] > 0.5 else None
            right_wrist = keypoints[10][:2] if keypoints[10][2] > 0.5 else None
            
            # 计算手臂角度
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            if angle is not None:
                # 在图像上显示角度
                cv2.putText(frame, f"Angle: {angle:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # 状态转换和计数
                if pushup_state == "up" and angle < 90.0:
                    pushup_state = "down"
                elif pushup_state == "down" and angle > 160.0:
                    pushup_state = "up"
                    pushup_count += 1
                    
                # 在图像上显示当前状态
                state_color = (0, 255, 0) if pushup_state == "up" else (0, 0, 255)
                cv2.putText(frame, f"State: {pushup_state}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        
        # 绘制姿势估计结果
        frame = results.plot()
        display_text = f"Push-ups: {pushup_count}"
    else:
        # 处理深蹲
        results = model(frame)[0]
        
        # 检查是否检测到姿势
        if len(results.keypoints.data) > 0:
            # 获取关键点
            keypoints = results.keypoints.data[0].cpu().numpy()
            
            # 提取右髋、右膝和右踝的坐标
            right_hip = keypoints[12][:2] if keypoints[12][2] > 0.5 else None
            right_knee = keypoints[14][:2] if keypoints[14][2] > 0.5 else None
            right_ankle = keypoints[16][:2] if keypoints[16][2] > 0.5 else None
            
            # 计算腿部角度
            angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            if angle is not None:
                # 在图像上显示角度
                cv2.putText(frame, f"Angle: {angle:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # 状态转换和计数
                if squat_state == "up" and angle < 100.0:
                    squat_state = "down"
                elif squat_state == "down" and angle > 160.0:
                    squat_state = "up"
                    squat_count += 1
                    
                # 在图像上显示当前状态
                state_color = (0, 255, 0) if squat_state == "up" else (0, 0, 255)
                cv2.putText(frame, f"State: {squat_state}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
                
        # 绘制姿势估计结果
        frame = results.plot()
        display_text = f"Squats: {squat_count}"

    # 在画面上显示提示信息和计数
    cv2.putText(frame, "Press 's' to switch exercise", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, display_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 显示结果
    cv2.imshow("Workout Monitoring", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # 切换运动模式并重置状态
        if current_exercise == "pushup":
            current_exercise = "squat"
            pushup_state = "up"  # 重置俯卧撑状态
        else:
            current_exercise = "pushup"
            squat_state = "up"  # 重置深蹲状态

cap.release()
cv2.destroyAllWindows()

print(f"Workout finished! Push-ups: {pushup_count}, Squats: {squat_count}")