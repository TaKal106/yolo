import cv2
import os

# 定义输入和输出路径
img_dir = 'data/images'
txt_dir = 'runs_light/inference_bottom/exp/labels'
save_dir = 'runs_light/crop_bottom_result'

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# 定义裁剪的高宽比
aspect_ratio = 3.5/1

for img_file in os.listdir(img_dir):
    if img_file.endswith('.jpg') or img_file.endswith('.png'):
        img_path = os.path.join(img_dir, img_file)
        txt_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        txt_path = os.path.join(txt_dir, txt_file)

        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip().split()
            class_name = line[0]
            x, y, w, h = map(float, line[1:5])

            # 计算中心点坐标
            center_x = x * img_w
            center_y = y * img_h

            # 计算左上角和右下角坐标
            left = int(center_x - w/2 * img_w)
            top = int(center_y - h/2 * img_h)
            right = int(center_x + w/2 * img_w)
            bottom = int(center_y + h/2 * img_h)

            # 计算裁剪区域左上角和右下角坐标
            crop_height = int(w * img_w * aspect_ratio)
            left_crop = max(0, int(center_x - w/2 * img_w))
            right_crop = min(img_w, int(center_x + w/2 * img_w))
            top_crop = max(0, int(center_y - crop_height/2))
            bottom_crop = min(img_h, int(center_y + crop_height/2))

            # 裁剪图像并保存
            crop_img = img[top_crop:bottom_crop, left_crop:right_crop]
            save_path = os.path.join(save_dir, f'{img_file[:-4]}_{class_name}_{i}.jpg')
            cv2.imwrite(save_path, crop_img)
