#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import pickle
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
    parser.add_argument('--gframe', default=0, help='global frame num')
    parser.add_argument('--lframe', default=16, help='local frame num')
    return parser

def read_ref_frame_list(args):
    ref_frame_list_path = os.path.join(args.input_dir, "my_model_ref_frame_list.npy")
    ref_frame_list = np.load(ref_frame_list_path, allow_pickle=True)
    ref_frame_batch_set = []
    ref_frame_batch = []
    for i, ref_frame in enumerate(ref_frame_list):
        ref_frame_batch.append(ref_frame)
        if (i + 1) % args.lframe == 0:
                logger.info(f"Ref frame {i}: {ref_frame}")
                ref_frame_batch_set.append(ref_frame_batch)
                ref_frame_batch = []
    return ref_frame_batch_set

def read_input_feature_info_list(args):
    # feat_info_list: list of (pred_info_p3, pred_info_p4, pred_info_p5) per batchset
    feat_info_save_path = os.path.join(args.input_dir, "my_model_input_feat_info.pkl")
    with open(feat_info_save_path, "rb") as f:
            feat_info_list = pickle.load(f)
    logger.info(f"len(feat_info_list): {len(feat_info_list)}")
    check_batch_set_num = 2
    check_batch_item_num = 2
    for batch_set, feat_info_set in enumerate(feat_info_list):
        if batch_set == check_batch_set_num:
            logger.info(f"Batch set {batch_set} - num batch items: {len(feat_info_set)}")
        for batch_item, feat_info_item in enumerate(feat_info_set):
            p3_list, p4_list, p5_list = feat_info_item
            if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                logger.info(f"  Batch item {batch_item} - p3: {len(p3_list)} items,\
                             p4: {len(p4_list)} items, p5: {len(p5_list)} items")
            for p3 in p3_list:
                 bbox = p3[:4]
                 obj_score = p3[4]
                 cls_score = p3[5]
                 class_pred = p3[6:]
                 class_label = np.argmax(class_pred)
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 if batch_set == 2 and batch_item == 2:
                    logger.info(f"    p3 - bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                obj_score: {obj_score:.3f}, \
                                cls_score: {cls_score:.3f}, \
                                conf_score: {conf_score:.3f}, \
                                class_label: {class_label}, \
                                class_label_name: {class_label_name}")
            for p4 in p4_list:
                 bbox = p4[:4]
                 obj_score = p4[4]
                 cls_score = p4[5]
                 class_pred = p4[6:]
                 class_label = np.argmax(class_pred)
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 if batch_set == 2 and batch_item == 2:
                    logger.info(f"    p4 - bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                obj_score: {obj_score:.3f}, \
                                cls_score: {cls_score:.3f}, \
                                conf_score: {conf_score:.3f}, \
                                class_label: {class_label}, \
                                class_label_name: {class_label_name}")
            for p5 in p5_list:
                 bbox = p5[:4]
                 obj_score = p5[4]
                 cls_score = p5[5]
                 class_pred = p5[6:]
                 class_label = np.argmax(class_pred)
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 if batch_set == 2 and batch_item == 2:
                    logger.info(f"    p5 - bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                obj_score: {obj_score:.3f}, \
                                cls_score: {cls_score:.3f}, \
                                conf_score: {conf_score:.3f}, \
                                class_label: {class_label}, \
                                class_label_name: {class_label_name}")
            if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                    break
        if batch_set == check_batch_set_num:
            break
    return feat_info_list

def read_mem_feature_info_list(args):
    mem_feat_info_save_path = os.path.join(args.input_dir, "my_model_mem_feat_info.pkl")
    with open(mem_feat_info_save_path, "rb") as f:
            mem_feat_info_list = pickle.load(f)
    logger.info(f"len(mem_feat_info_list): {len(mem_feat_info_list)}")
    check_batch_set_num = 1
    for batch_set, mem_feat_info in enumerate(mem_feat_info_list):
         if batch_set == check_batch_set_num:
             p3_mem_info = mem_feat_info['p3']
             p4_mem_info = mem_feat_info['p4']
             p5_mem_info = mem_feat_info['p5']
             logger.info(f"Batch set {batch_set} - mem_feat_info - len(p3_mem_info): {len(p3_mem_info)}, \
                         len(p4_mem_info): {len(p4_mem_info)}, len(p5_mem_info): {len(p5_mem_info)}")

    return mem_feat_info_list

def read_sampled_mem_feature_info_list(args):
    sampled_mem_feat_info_save_path = os.path.join(args.input_dir, "my_model_sampled_mem_feat_info.pkl")
    with open(sampled_mem_feat_info_save_path, "rb") as f:
            sampled_mem_feat_info_list = pickle.load(f)
    logger.info(f"len(sampled_mem_feat_info_list): {len(sampled_mem_feat_info_list)}")
    check_batch_set_num = 1
    for batch_set, sampled_mem_feat_info in enumerate(sampled_mem_feat_info_list):
         if batch_set == check_batch_set_num:
             p3_sampled_mem_info = sampled_mem_feat_info['p3']
             p4_sampled_mem_info = sampled_mem_feat_info['p4']
             p5_sampled_mem_info = sampled_mem_feat_info['p5']
             logger.info(f"Batch set {batch_set} - sampled_mem_feat_info - len(p3_sampled_mem_info): {len(p3_sampled_mem_info)}, \
                         len(p4_sampled_mem_info): {len(p4_sampled_mem_info)}, len(p5_sampled_mem_info): {len(p5_sampled_mem_info)}")

    return sampled_mem_feat_info_list
def read_updated_feature_info_list(args):
    updated_feat_info_save_path = os.path.join(args.input_dir, "my_model_updated_feat_info.pkl")
    with open(updated_feat_info_save_path, "rb") as f:
            updated_feat_info_list = pickle.load(f)
    logger.info(f"len(updated_feat_info_list): {len(updated_feat_info_list)}")
    check_batch_set_num = 2
    check_batch_item_num = 2
    for batch_set, updated_feat_info_set in enumerate(updated_feat_info_list):
        if batch_set == check_batch_set_num:
            logger.info(f"Batch set {batch_set} - num batch items: {len(updated_feat_info_set)}")
        for batch_item, updated_feat_info_item in enumerate(updated_feat_info_set):
            if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                logger.info(f"  Batch item {batch_item} - num features: {len(updated_feat_info_item)}")
                p3_list, p4_list, p5_list = updated_feat_info_item
                logger.info(f"  Batch item {batch_item} - p3: {len(p3_list)} items,\
                             p4: {len(p4_list)} items, p5: {len(p5_list)} items")
                for p3 in p3_list:
                     bbox = p3[:4]
                     obj_score = p3[4]
                     cls_score = p3[5]
                     class_pred = p3[6:]
                     class_label = np.argmax(class_pred)
                     conf_score = cls_score * obj_score
                     class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                     logger.info(f"    p3 - bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                 obj_score: {obj_score:.3f}, \
                                 cls_score: {cls_score:.3f}, \
                                 conf_score: {conf_score:.3f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}")
                for p4 in p4_list:
                     bbox = p4[:4]
                     obj_score = p4[4]
                     cls_score = p4[5]
                     class_pred = p4[6:]
                     class_label = np.argmax(class_pred)
                     conf_score = cls_score * obj_score
                     class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                     logger.info(f"    p4 - bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                 obj_score: {obj_score:.3f}, \
                                 cls_score: {cls_score:.3f}, \
                                 conf_score: {conf_score:.3f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}")
                for p5 in p5_list:
                     bbox = p5[:4]
                     obj_score = p5[4]
                     cls_score = p5[5]
                     class_pred = p5[6:]
                     class_label = np.argmax(class_pred)
                     conf_score = cls_score * obj_score
                     class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                     logger.info(f"    p5 - bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                 obj_score: {obj_score:.3f}, \
                                 cls_score: {cls_score:.3f}, \
                                 conf_score: {conf_score:.3f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}")
                if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                    break
        if batch_set == check_batch_set_num:
            break
    return updated_feat_info_list

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

    ref_frame_batch_set = read_ref_frame_list(args)
    input_feat_info_list = read_input_feature_info_list(args)
    mem_feat_info_list = read_mem_feature_info_list(args)
    sampled_mem_feat_info_list = read_sampled_mem_feature_info_list(args)
    updated_feat_info_list = read_updated_feature_info_list(args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
