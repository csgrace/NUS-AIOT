import os
import shutil
import random
import yaml

# 数据集根目录
DATASET_DIR = os.path.dirname(os.path.abspath(__file__)) + "/UECFOOD256"

# 读取类别名
category_file = os.path.join(DATASET_DIR, "category.txt")
with open(category_file, "r", encoding="utf-8") as f:
    lines = f.readlines()[1:]
    names = [line.strip().split("\t", 1)[1] for line in lines]

# 生成完整的 names 列表到 yaml
yaml_path = os.path.join(os.path.dirname(DATASET_DIR), "uecfood256.yaml")
yaml_dict = {
    'train': '../UECFOOD256/images/train',
    'val': '../UECFOOD256/images/val',
    'nc': len(names),
    'names': names
}
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(yaml_dict, f, allow_unicode=True)

# 整理图片和标签
img_dirs = [str(i) for i in range(1, 257)]
all_items = []
for d in img_dirs:
    dir_path = os.path.join(DATASET_DIR, d)
    bb_file = os.path.join(dir_path, "bb_info.txt")
    if not os.path.exists(bb_file):
        continue
    with open(bb_file, "r") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            img_id, x1, y1, x2, y2 = parts
            img_file = os.path.join(dir_path, f"{img_id}.jpg")
            if not os.path.exists(img_file):
                continue
            # YOLO标签格式: class x_center y_center w h (归一化)
            w = int(x2) - int(x1)
            h = int(y2) - int(y1)
            x_center = int(x1) + w / 2
            y_center = int(y1) + h / 2
            # 读取图片尺寸
            try:
                from PIL import Image
                with Image.open(img_file) as im:
                    img_w, img_h = im.size
            except:
                continue
            x_center /= img_w
            y_center /= img_h
            w /= img_w
            h /= img_h
            label = f"{int(d)-1} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
            all_items.append((img_file, label))

# 随机划分训练集和验证集
random.seed(42)
random.shuffle(all_items)
train_items = all_items[:int(0.8*len(all_items))]
val_items = all_items[int(0.8*len(all_items)) :]

# 创建目标文件夹
for split, items in zip(["train", "val"], [train_items, val_items]):
    img_out_dir = os.path.join(DATASET_DIR, "images", split)
    label_out_dir = os.path.join(DATASET_DIR, "labels", split)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)
    for idx, (img_path, label) in enumerate(items):
        base = os.path.splitext(os.path.basename(img_path))[0]
        img_dst = os.path.join(img_out_dir, f"{base}.jpg")
        label_dst = os.path.join(label_out_dir, f"{base}.txt")
        shutil.copy(img_path, img_dst)
        with open(label_dst, "w") as f:
            f.write(label)

print("数据集整理完成，配置文件已生成！")
