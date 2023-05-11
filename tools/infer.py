#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import os.path as osp

import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer


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
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt.')#是否保存结果为txt文件
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


@torch.no_grad()
def run(weights=osp.join(ROOT, 'weights/yolov6l.pt'),
        source=osp.join(ROOT, 'runs_test/padding_top_result'),
        webcam=False,
        webcam_addr=0,
        yaml='data/coco.yaml',
        img_size=640,
        conf_thres=0.15,
        iou_thres=0.1,
        max_det=1000,
        device='0',
        save_txt=True,
        not_save_img=False,
        save_dir=None,
        view_img=True,
        classes=39,
        agnostic_nms=False,
        project=osp.join(ROOT, 'runs_test/inference_top'),
        name='exp',
        hide_labels=True,
        hide_conf=False,
        half=False,
        ):
    """ Inference process, supporting inference on one image file or directory which containing images.
    Args:
        weights: The path of model.pt, e.g. yolov6s.pt
        source: Source path, supporting image files or dirs containing images.
        yaml: Data yaml file, .
        img_size: Inference image-size, e.g. 640
        conf_thres: Confidence threshold in inference, e.g. 0.25
        iou_thres: NMS IOU threshold in inference, e.g. 0.45
        max_det: Maximal detections per image, e.g. 1000
        device: Cuda device, e.e. 0, or 0,1,2,3 or cpu
        save_txt: Save results to *.txt
        not_save_img: Do not save visualized inference results
        classes: Filter by class: --class 0, or --class 0 2 3
        agnostic_nms: Class-agnostic NMS
        project: Save results to project/name
        name: Save results to project/name, e.g. 'exp'
        line_thickness: Bounding box thickness (pixels), e.g. 3
        hide_labels: Hide labels, e.g. False
        hide_conf: Hide confidences
        half: Use FP16 half-precision inference, e.g. False
    """
    # create save dir
    if save_dir is None:
        save_dir = osp.join(project, name)
        save_txt_path = osp.join(save_dir, 'labels')
    else:
        save_txt_path = save_dir
    if (not not_save_img or save_txt) and not osp.exists(save_dir):
        os.makedirs(save_dir)
    else:
        LOGGER.warning('Save directory already existed')
    if save_txt:
        save_txt_path = osp.join(save_dir, 'labels')
        if not osp.exists(save_txt_path):
            os.makedirs(save_txt_path)

    # Inference
    inferer = Inferer(source, webcam, webcam_addr, weights, device, yaml, img_size, half)
    inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, not not_save_img, hide_labels, hide_conf, view_img)

    if save_txt or not not_save_img:
        LOGGER.info(f"Results saved to {save_dir}")


def main(args):
    run(**vars(args))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
