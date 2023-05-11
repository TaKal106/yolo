import cv2
import os
import glob
import subprocess
import numpy as np
import time
import sys
sys.path.append('tools')
import infer
import argparse
from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer

def padding_top(src_folder, save_folder):

    # 保留的上部分高度百分比和下部分高度百分比
    top_percent = 0.14 #上部分隐藏%多少高度
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

        # 调用yolo的命令行
        # command = ['python', 'tools/infer.py', '--weights', 'weights/yolov6l.pt', '--source', img, '--device', '0', '--save-txt', '--classes', '39']
        # subprocess.run(command, check=True)

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Inference.', add_help=add_help)
    parser.add_argument('--weights', type=str, default='weights/yolov6l.pt', help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default='runs_test/padding_top_result', help='the source path, e.g. image-file/dir.') #输入图像路径
    parser.add_argument('--webcam', action='store_true', help='whether to use webcam.') # 是否使用摄像头进行推理 
    parser.add_argument('--webcam-addr', type=str, default='0', help='the web camera address, local camera or rtsp address.')
    parser.add_argument('--yaml', type=str, default='data/coco.yaml', help='data yaml file.') #数据集的yaml文件路径，用于模型推理，判断object label
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='the image-size(h,w) in inference size.') #输出图像大小
    parser.add_argument('--conf-thres', type=float, default=0.15, help='confidence threshold for inference.') #标检测的置信度阈值
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold for inference.') #非极大值抑制（NMS）的IoU阈值，用于去除重复的检测框
    parser.add_argument('--max-det', type=int, default=1000, help='maximal inferences per image.') #最多检测出来的对象数
    parser.add_argument('--device', default='0', help='device to run our model i.e. 0 or 0,1,2,3 or cpu.')# 模型运行的设备，可以是GPU的id
    parser.add_argument('--save-txt',action='store_true', help='save results to *.txt.', default=True)#是否保存结果为txt文件
    parser.add_argument('--not-save-img', action='store_true', help='do not save visuallized inference results.')#是否保存推理结果的可视化图片
    parser.add_argument('--save-dir', type=str, help='directory to save predictions in. See --save-txt.')#保存结果文件的目录
    parser.add_argument('--view-img', action='store_true', help='show inference results')#是否查看推理结果的可视化图
    parser.add_argument('--classes', nargs='+', type=int, help='filter by classes, e.g. --classes 0, or --classes 0 2 3.')#需要过滤的目标类别，可以是单个类别的整数ID，也可以是多个类别ID的列表
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS.')#
    parser.add_argument('--project', default='runs_test/inference_top', help='save inference results to project/name.')#结果保存目录的项目名称
    parser.add_argument('--name', default='exp', help='save inference results to project/name.')#结果保存目录的项目名称
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels.')  # 隐藏标签
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences.') #隐藏置信度
    parser.add_argument('--half', action='store_true', help='whether to use FP16 half-precision inference.')

    args = parser.parse_args()
    LOGGER.info(args)
    return args

def main():
    # 程序开始时间戳
    start_time = time.time()
    # 训练阶段：图片所在文件夹路径、结果保存的文件夹路径
    src_folder = 'data/test_images/'
    save_folder = 'runs_test/padding_top_result/'
    padding_top(src_folder,save_folder)
    args = get_args_parser()
    # python tools/infer.py --weights weights/yolov6l.pt --source runs_light/padding_top_result/ --device 0 --save-txt True --classes 39
    infer.run(**vars(args))
    # 程序结束时间戳
    end_time = time.time()
    # 计算执行时间
    execution_time = end_time - start_time

    # 输出执行时间
    print(f"程序执行时间为: {execution_time} 秒")

main()





