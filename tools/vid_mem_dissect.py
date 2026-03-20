#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

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
    return parser

def read_feature_info_and_ref_frame_list(file_path, ref_frame_list_path):
    # feat_info_list: list of (pred_info_p3, pred_info_p4, pred_info_p5) per batchset
    feat_info_list = np.load(file_path, allow_pickle=True)
    logger.info(f"len(feat_info_list): {len(feat_info_list)}")
    ref_frame_list = np.load(ref_frame_list_path, allow_pickle=True)
    for batch_idx, feat_info in enumerate(feat_info_list):
        pred_info_p3, pred_info_p4, pred_info_p5 = feat_info
        logger.info(f"batch_idx: {batch_idx}, pred_info_p3.shape: {pred_info_p3.shape}, pred_info_p4.shape: {pred_info_p4.shape}, pred_info_p5.shape: {pred_info_p5.shape}")



def main(exp, args):

    current_time = time.localtime()
    file_name = os.path.join(args.output_dir, exp.exp_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(args.output_dir,file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    logger.info("Args: {}".format(args))
    feat_info_file = os.path.join(args.input_dir, "my_model_feat_info.npy")
    ref_frame_list_file = os.path.join(args.input_dir, "my_model_ref_frame_list.npy")
    read_feature_info_and_ref_frame_list(feat_info_file, ref_frame_list_file)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
