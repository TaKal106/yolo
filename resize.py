import cv2
import os

img_dir = 'runs_light/crop_bottom_result'
save_dir = 'runs_light/resize_bottom_result'
resize_width = 128

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for img_file in os.listdir(img_dir):
    if img_file.endswith('.jpg') or img_file.endswith('.png'):
        img_path = os.path.join(img_dir, img_file)

        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape
        resize_height = int(resize_width / img_width * img_height)

        img_resized = cv2.resize(img, (resize_width, resize_height))

        save_path = os.path.join(save_dir, img_file)
        cv2.imwrite(save_path, img_resized)
