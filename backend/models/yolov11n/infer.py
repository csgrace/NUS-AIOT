import argparse
from ultralytics import YOLO

class YOLOInfer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # 自动加载的类别词表

    def infer(self, image_path, save=False):
        results = self.model(image_path, save=save)
        boxes = results[0].boxes
        detections = []
        for i in range(boxes.shape[0]):
            cls = boxes.cls[i].item() if boxes.cls is not None else None
            conf = boxes.conf[i].item() if boxes.conf is not None else None
            xyxy = boxes.xyxy[i].tolist() if boxes.xyxy is not None else None
            cls_name = self.class_names[int(cls)] if cls is not None else None
            detections.append({
                'class': cls,
                'class_name': cls_name,
                'confidence': conf,
                'coordinates': xyxy
            })
        return detections



    def main():
        parser = argparse.ArgumentParser(description="YOLOv11s 推理 CLI")
        parser.add_argument("--model", type=str, default="/Users/izumedonabe/CS/NUS-Summer-Workshop-AIOT/backend/models/yolov11n/best.pt", help="模型权重路径")
        parser.add_argument("--save", action="store_true", help="是否保存带检测框的图片")
        args = parser.parse_args()

        model = YOLO(args.model)
        class_names = model.names  # 自动加载的类别词表
        while True:
            image_path = input("请输入待检测的图片路径（输入 exit 退出）：")
            if image_path.strip().lower() == "exit":
                break
            try:
                results = model(image_path, save=args.save)
                boxes = results[0].boxes
                for i in range(boxes.shape[0]):
                    cls = boxes.cls[i].item() if boxes.cls is not None else None
                    conf = boxes.conf[i].item() if boxes.conf is not None else None
                    xyxy = boxes.xyxy[i].tolist() if boxes.xyxy is not None else None
                    cls_name = class_names[int(cls)] if cls is not None else None
                    print(f"检测框{i}: 类别={cls}({cls_name}), 置信度={conf}, 坐标={xyxy}")
            except Exception as e:
                print(f"检测失败: {e}")

    if __name__ == "__main__":
        main()