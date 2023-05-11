import cv2
import os
import glob

# 训练阶段：图片所在文件夹路径、结果保存的文件夹路径
src_folder = 'data/images/'
save_folder = 'runs_light/padding_top_result/'

# 测试阶段：图片所在文件夹路径、结果保存的文件夹路径
# src_folder = 'data/test_images/'
# save_folder = 'run_test/padding_top_result/'

# 保留的上部分高度百分比和下部分高度百分比
top_percent = 0.0 #上部分隐藏%多少高度
bottom_percent = 0.7 #下部分隐藏%多少高度
middle_percent = 0.0 #中间部分保留%多少高度
left_percent = 0.0 #左部分隐藏%多少宽度
right_percent = 0.18 #右部分隐藏%多少宽度

# 获取所有的png图片路径
png_files = glob.glob(os.path.join(src_folder, '*.png'))

for png_file in png_files:
    # 读取图片
    img = cv2.imread(png_file)

    # 图片高度
    h,w, _ = img.shape

    # 上部分高度
    top_height = int(h * top_percent)

    # 下部分高度
    bottom_height = int(h * bottom_percent)

    # 中间部分高度
    middle_height = int(h * middle_percent)

    # 左部分宽度
    left_width = int(w * left_percent)

    # 右部分宽度
    right_width = int(w * right_percent)


    # 将上下左右部分之外的区域涂成黑色
    img[:top_height, :] = 0
    img[h - bottom_height:, :] = 0
    img[:, :left_width, :] = 0
    img[:, w - right_width:, :] = 0

    # 保存结果
    filename = os.path.basename(png_file)
    save_path = os.path.join(save_folder, filename)
    cv2.imwrite(save_path, img)
