import cv2
import numpy as np

img_left = cv2.imread('left.png')
img_right = cv2.imread('right.png')

# 保留的上部分高度百分比和下部分高度百分比
top_percent = 0.13 #上部分隐藏%多少高度
bottom_percent = 0.15 #下部分隐藏%多少高度
left_percent = 0.25 #左部分隐藏%多少宽度
right_percent = 0.18 #右部分隐藏%多少宽度
def process_png_files(img, top_percent, bottom_percent, left_percent, right_percent):

    # 图片高度
    h, w, _ = img.shape

    # 上部分高度
    top_height = int(h * top_percent)

    # 下部分高度
    bottom_height = int(h * bottom_percent)
    

    # 左部分宽度
    left_width = int(w * left_percent)

    # 右部分宽度
    right_width = int(w * right_percent)
    # 矩形框四个顶点坐标
    left_top = (left_width,h-top_height)
    right_top = (w-right_width,h-top_height)
    left_bottom = (left_width,bottom_height)
    right_bottom = (w-right_width,bottom_height)
    # 画矩形框
    # pts = np.array([left_top, right_top, right_bottom, left_bottom], np.int32)
    # cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        # 裁剪图像
    cropped_img = img[bottom_height:h-top_height, left_width:w-right_width]
    # cropped_img = img[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    print(cropped_img.shape)
    return cropped_img
    
    # return img

img_left = process_png_files(img_left,top_percent, bottom_percent, left_percent, right_percent)
img_right = process_png_files(img_right,top_percent, bottom_percent, left_percent, right_percent)
# 绘制矩形框
result = cv2.hconcat([img_left, img_right])

# 保存结果
cv2.imwrite('result5.png', result)
