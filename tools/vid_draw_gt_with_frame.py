#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import csv
import os
import pickle
import time
from xml.dom import minidom
from loguru import logger

import cv2

from sympy import arg
import torch

from tools.vid_demo_with_feature import _VID_SYNSET_TO_IDX
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets.vid_classes import VID_classes
import numpy as np
from yolox.exp import get_exp

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png",".JPEG"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOV Memory Dissecting Tool!")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "-f",
        "--exp_file",
        default='./exps/yolov/yolov_s.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument('--output_dir', default='./YOLOX_outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--input_dir', default='',
                        help='path where to read input result')
    parser.add_argument('--save_result', default=True)
    parser.add_argument('--input_frame_name', default='000307.JPEG_Input_P3_0.JPEG.JPEG', help='input frame name to check')
    parser.add_argument('--input_xml_path', default='/home/kssong/ILSVRC2015/Annotations/VID/val/ILSVRC2015_val_00118007', help='input frame name to check')
    parser.add_argument('--input_xml_file_name', default='000307.xml', help='input frame name to check')
    return parser


def visualize_result_info_on_frame(args,  frame_save_path):
    input_frame_path = os.path.join(args.input_dir, args.input_frame_name)
    logger.info(f"Visualizing result info on frame: {input_frame_path}")

    frame = cv2.imread(input_frame_path)
    if frame is None:
        logger.error(f"Failed to read image: {input_frame_path}")
        return
    xml_path = os.path.join(args.input_xml_path, args.input_xml_file_name)
    if os.path.exists(xml_path):
        xml_doc = minidom.parse(xml_path)
        root = xml_doc.documentElement
        for obj in root.getElementsByTagName("object"):
            synset = obj.getElementsByTagName("name")[0].firstChild.data
            xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
            ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
            xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
            ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
            cls_idx = _VID_SYNSET_TO_IDX.get(synset, -1)
            label = VID_classes[cls_idx] if cls_idx >= 0 else synset
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(frame, 'GT:' + label, (xmax-10, max(ymin - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            logger.info(f"    GT bbox: ({xmin}, {ymin}, {xmax}, {ymax}), label: {label}")
    else:
        logger.warning("GT xml not found: {}".format(xml_path))
    cv2.imwrite(os.path.join(frame_save_path, f"{args.input_frame_name}_GT.JPEG"), frame)

def main(exp, args):

    current_time = time.localtime()
    file_name = os.path.join(args.output_dir, exp.exp_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    logger.info("Args: {}".format(args))

    os.makedirs(save_folder, exist_ok=True)
    result_save_path = save_folder
    logger.add(os.path.join(result_save_path, "run.log"), mode="w")

    visualize_result_info_on_frame(args, save_folder)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
