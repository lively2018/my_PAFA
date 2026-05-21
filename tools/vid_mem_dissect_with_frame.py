#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import csv
import os
import pickle
import sys
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
    parser.add_argument('--save_result', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--draw_input_feat_info', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--draw_mem_feat_info', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--draw_sampled_mem_feat_info', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--draw_updated_feat_info', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--draw_result_feat_info', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--draw_outputs_feat_info', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--draw_agg_feat_info', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--gframe', default=0, help='global frame num')
    parser.add_argument('--lframe', default=16, help='local frame num')
    parser.add_argument('--test_conf', type=float, default=0.01, help='test confidence threshold')
    parser.add_argument('--tsize', default=640, type=int, help='test image size')
    parser.add_argument('--input_frame_name', default='000000.JPEG', help='input frame name to check')
    parser.add_argument('--input_image_path', default='/home/kssong/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00118007', help='input frame name to check')
    parser.add_argument('--save_result_with_gt', default=False, type=lambda x: x.lower() == 'true')
    parser.add_argument('--save_csv_result', default=True, type=lambda x: x.lower() == 'true')
    return parser
def calculate_iou_gt_img(gt_bbox, img_bbox):
    x1 = max(gt_bbox[0], img_bbox[0])
    y1 = max(gt_bbox[1], img_bbox[1])
    x2 = min(gt_bbox[2], img_bbox[2])
    y2 = min(gt_bbox[3], img_bbox[3])

    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    area_gt  = (gt_bbox[2]  - gt_bbox[0])  * (gt_bbox[3]  - gt_bbox[1])
    area_img = (img_bbox[2] - img_bbox[0]) * (img_bbox[3] - img_bbox[1])
    union = area_gt + area_img - inter
    iou = inter / (union + 1e-12)
    return iou

def find_batch_set_and_item_for_input_frame(ref_frame_batch_set, input_frame_name):
    #logger.info(f"Finding batch set and item for input frame: {input_frame_name}")
    for batch_set_idx, ref_frame_batch in enumerate(ref_frame_batch_set):
        for batch_item_idx, ref_frame in enumerate(ref_frame_batch):
            #logger.info(f"Checking batch set {batch_set_idx}, batch item {batch_item_idx}, ref frame: {ref_frame}")
            ref_frame_name = os.path.basename(ref_frame)
            #logger.info(f"Extracted ref frame name: {ref_frame_name}")
            if ref_frame_name == input_frame_name:
                logger.info(f"Found input frame {input_frame_name} in batch set {batch_set_idx}, batch item {batch_item_idx}")
                return batch_set_idx, batch_item_idx
    logger.warning(f"Input frame {input_frame_name} not found in any batch set or batch item")
    return None, None

def read_ref_frame_list(args, n_frames):
    ref_frame_list_path = os.path.join(args.input_dir, "my_model_ref_frame_list.npy")
    ref_frame_list = np.load(ref_frame_list_path, allow_pickle=True)
    ref_frame_batch_set = []
    ref_frame_batch = []
    logger.info(f"len(ref_frame_list): {len(ref_frame_list)}")
    logger.info(f"n_frames: {n_frames}")
    for i, ref_frame in enumerate(ref_frame_list):
        ref_frame_batch.append(ref_frame)
        if (i + 1) % n_frames == 0:
            logger.info(f"Ref frame {i}: {ref_frame}")
            ref_frame_batch_set.append(ref_frame_batch)
            ref_frame_batch = []
    return ref_frame_batch_set

def read_input_feature_info_list(args, check_batch_set, check_batch_item, save_folder):
    # feat_info_list: list of (pred_info_p3, pred_info_p4, pred_info_p5) per batchset
    feat_info_save_path = os.path.join(args.input_dir, "my_model_input_feat_info.pkl")
    with open(feat_info_save_path, "rb") as f:
            feat_info_list = pickle.load(f)
    logger.info(f"len(feat_info_list): {len(feat_info_list)}")
    check_batch_set_num = check_batch_set
    check_batch_item_num = check_batch_item
    for batch_set, feat_info_set in enumerate(feat_info_list):
        if batch_set == check_batch_set_num:
            logger.info(f"Batch set {batch_set} - num batch items: {len(feat_info_set)}")
        for batch_item, feat_info_item in enumerate(feat_info_set):
            p3_list, p4_list, p5_list = feat_info_item
            if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                logger.info(f"  Batch item {batch_item} - p3: {len(p3_list)} items,\
                             p4: {len(p4_list)} items, p5: {len(p5_list)} items")
            for i, p3 in enumerate(p3_list):
                 if np.all(p3 == 0):
                     continue
                 bbox = p3[:4]
                 obj_score = p3[4]
                 cls_score = p3[5]
                 class_pred = p3[6: 6 + len(VID_classes)]
                 class_label = int(np.argmax(class_pred))
                 feat_num = int(p3[6 + len(VID_classes)])
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                    logger.info(f"    p3 -{i}th - bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                obj_score: {obj_score:.3f}, \
                                cls_score: {cls_score:.3f}, \
                                conf_score: {conf_score:.6f}, \
                                class_pred: {np.array2string(class_pred, precision=6, floatmode='fixed', suppress_small=True)}, \
                                class_label: {class_label}, \
                                class_label_name: {class_label_name}, \
                                feat_num: {feat_num}")
            for i, p4 in enumerate(p4_list):
                 if np.all(p4 == 0):
                     continue
                 bbox = p4[:4]
                 obj_score = p4[4]
                 cls_score = p4[5]
                 class_pred = p4[6: 6 + len(VID_classes)]
                 class_label = int(np.argmax(class_pred))
                 feat_num = int(p4[6 + len(VID_classes)])
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                    logger.info(f"    p4 -{i}th - bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                obj_score: {obj_score:.3f}, \
                                cls_score: {cls_score:.3f}, \
                                conf_score: {conf_score:.6f}, \
                                class_pred: {np.array2string(class_pred, precision=6, floatmode='fixed', suppress_small=True)}, \
                                class_label: {class_label}, \
                                class_label_name: {class_label_name}, \
                                feat_num: {feat_num}")

            for i, p5 in enumerate(p5_list):
                 if np.all(p5 == 0):
                     continue
                 bbox = p5[:4]
                 obj_score = p5[4]
                 cls_score = p5[5]
                 class_pred = p5[6: 6 + len(VID_classes)]
                 class_label = int(np.argmax(class_pred))
                 feat_num = int(p5[6 + len(VID_classes)])
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                    logger.info(f"    p5 -{i}th - bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                obj_score: {obj_score:.3f}, \
                                cls_score: {cls_score:.3f}, \
                                conf_score: {conf_score:.6f}, \
                                class_pred: {np.array2string(class_pred, precision=6, floatmode='fixed', suppress_small=True)}, \
                                class_label: {class_label}, \
                                class_label_name: {class_label_name}, \
                                feat_num: {feat_num}")
            if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                    break
        if batch_set == check_batch_set_num:
            break
    return feat_info_item

def read_outputs_info_list(args, check_batch_set, check_batch_item, save_folder):
    outputs_info_list = []
    outputs_info_save_path = os.path.join(args.input_dir, "my_model_outputs_info.pkl")
    with open(outputs_info_save_path, "rb") as f:
            outputs_info_list = pickle.load(f)
    logger.info(f"len(outputs_info_list): {len(outputs_info_list)}")
    check_batch_set_num = check_batch_set
    check_batch_item_num = check_batch_item
    selected_outputs_info_list = []

    for batch_set, outputs_info_set in enumerate(outputs_info_list):
        if batch_set == check_batch_set_num:
            logger.info(f"Batch set {batch_set} - num batch items: {len(outputs_info_set)}")
        for batch_item, outputs_info_item in enumerate(outputs_info_set):
            if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                logger.info(f"  Batch item {batch_item} - num outputs: {len(outputs_info_item)}")
                for i, output in enumerate(outputs_info_item):
                     selected_outputs_info_list.append(output)
                     #logger.info(f"    output -{i}th - len: {len(output)}")
                     #logger.info(f"    output -{i}th - content: {output}")

                     batch_set_output = output[0]
                     batch_item_output = output[1]
                     #logger.info(f"    output -{i}th - batch_set_output: {batch_set_output}, batch_item_output: {batch_item_output}")

                     bbox = output[2:6]
                     obj_score = float(output[6])
                     cls_score = float(output[7])
                     class_label = int(output[8])
                     feat_num = int(output[9]) if output[9] is not None else -1
                     conf_score = cls_score * obj_score
                     class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                     logger.info(f"    output -{i}th - batch_set: {batch_set_output}, batch_item: {batch_item_output}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                 obj_score: {obj_score:.3f}, \
                                 cls_score: {cls_score:.3f}, \
                                 conf_score: {conf_score:.6f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}, \
                                 feat_num: {feat_num}")
                return selected_outputs_info_list

    return selected_outputs_info_list

def read_mem_feature_info_list(args, check_batch_set, save_folder):
    mem_feat_info_save_path = os.path.join(args.input_dir, "my_model_mem_feat_info.pkl")
    with open(mem_feat_info_save_path, "rb") as f:
            mem_feat_info_list = pickle.load(f)
    logger.info(f"len(mem_feat_info_list): {len(mem_feat_info_list)}")
    check_batch_set_num = check_batch_set
    for batch_set, mem_feat_info in enumerate(mem_feat_info_list):
         if batch_set == check_batch_set_num:
             p3_mem_info = mem_feat_info['p3']
             p4_mem_info = mem_feat_info['p4']
             p5_mem_info = mem_feat_info['p5']
             logger.info(f"Batch set {batch_set} - mem_feat_info - len(p3_mem_info): {len(p3_mem_info)}, \
                         len(p4_mem_info): {len(p4_mem_info)}, len(p5_mem_info): {len(p5_mem_info)}")
             for i, p3 in enumerate(p3_mem_info):
                det = p3[2]
                if torch.all(det == 0):
                    continue
                bbox = det[:4]
                obj_score = det[4]
                cls_score = det[5]
                class_pred = det[6: 6 + len(VID_classes)]
                class_label = int(torch.argmax(class_pred))
                feat_num = int(det[6 + len(VID_classes)])
                conf_score = cls_score * obj_score
                class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                logger.info(f"    p3 -{i}th batch_set: {p3[0]}, batch_item: {p3[1]}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                            obj_score: {obj_score:.3f}, \
                            cls_score: {cls_score:.3f}, \
                            conf_score: {conf_score:.6f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name}, \
                            feat_num: {feat_num}")
             for i, p4 in enumerate(p4_mem_info):
                det = p4[2]
                if torch.all(det == 0):
                    continue
                bbox = det[:4]
                obj_score = det[4]
                cls_score = det[5]
                class_pred = det[6: 6 + len(VID_classes)]
                class_label = int(torch.argmax(class_pred))
                feat_num = int(det[6 + len(VID_classes)])
                conf_score = cls_score * obj_score
                class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                logger.info(f"    p4 -{i}th batch_set: {p4[0]}, batch    _item: {p4[1]}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                            obj_score: {obj_score:.3f}, \
                            cls_score: {cls_score:.3f}, \
                            conf_score: {conf_score:.6f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name}, \
                            feat_num: {feat_num}")
             for i, p5 in enumerate(p5_mem_info):
                det = p5[2]
                if torch.all(det == 0):
                    continue
                bbox = det[:4]
                obj_score = det[4]
                cls_score = det[5]
                class_pred = det[6: 6 + len(VID_classes)]
                class_label = int(torch.argmax(class_pred))
                feat_num = int(det[6 + len(VID_classes)])
                conf_score = cls_score * obj_score
                class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                logger.info(f"    p5 -{i}th batch_set: {p5[0]}, batch    _item: {p5[1]}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                            obj_score: {obj_score:.3f}, \
                            cls_score: {cls_score:.3f}, \
                            conf_score: {conf_score:.6f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name}, \
                            feat_num: {feat_num}")

             break

    return mem_feat_info

def read_sampled_mem_feature_info_list(args, check_batch_set, save_folder):
    sampled_mem_feat_info_save_path = os.path.join(args.input_dir, "my_model_sampled_mem_feat_info.pkl")
    with open(sampled_mem_feat_info_save_path, "rb") as f:
            sampled_mem_feat_info_list = pickle.load(f)
    logger.info(f"len(sampled_mem_feat_info_list): {len(sampled_mem_feat_info_list)}")
    check_batch_set_num = check_batch_set
    for batch_set, sampled_mem_feat_info in enumerate(sampled_mem_feat_info_list):
         if batch_set == check_batch_set_num:
             p3_sampled_mem_info = sampled_mem_feat_info['p3']
             p4_sampled_mem_info = sampled_mem_feat_info['p4']
             p5_sampled_mem_info = sampled_mem_feat_info['p5']
             logger.info(f"Batch set {batch_set} - sampled_mem_feat_info - len(p3_sampled_mem_info): {len(p3_sampled_mem_info)}, \
                         len(p4_sampled_mem_info): {len(p4_sampled_mem_info)}, len(p5_sampled_mem_info): {len(p5_sampled_mem_info)}")
             for i, p3 in enumerate(p3_sampled_mem_info):
                det = p3[2]
                if torch.all(det == 0):
                    continue
                bbox = det[:4]
                obj_score = det[4]
                cls_score = det[5]
                class_pred = det[6: 6 + len(VID_classes)]
                class_label = int(torch.argmax(class_pred))
                feat_num = int(det[6 + len(VID_classes)])
                conf_score = cls_score * obj_score
                class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                logger.info(f"    p3 -{i}th batch_set: {p3[0]}, batch_item: {p3[1]}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                            obj_score: {obj_score:.3f}, \
                            cls_score: {cls_score:.3f}, \
                            conf_score: {conf_score:.6f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name}, \
                            feat_num: {feat_num}")
             for i, p4 in enumerate(p4_sampled_mem_info):
                det = p4[2]
                if torch.all(det == 0):
                    continue
                bbox = det[:4]
                obj_score = det[4]
                cls_score = det[5]
                class_pred = det[6: 6 + len(VID_classes)]
                class_label = int(torch.argmax(class_pred))
                feat_num = int(det[6 + len(VID_classes)])
                conf_score = cls_score * obj_score
                class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                logger.info(f"    p4 -{i}th batch_set: {p4[0]}, batch_item: {p4[1]}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                            obj_score: {obj_score:.3f}, \
                            cls_score: {cls_score:.3f}, \
                            conf_score: {conf_score:.6f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name}, \
                            feat_num: {feat_num}")
             for i, p5 in enumerate(p5_sampled_mem_info):
                det = p5[2]
                if torch.all(det == 0):
                    continue
                bbox = det[:4]
                obj_score = det[4]
                cls_score = det[5]
                class_pred = det[6: 6 + len(VID_classes)]
                class_label = int(torch.argmax(class_pred))
                feat_num = int(det[6 + len(VID_classes)])
                conf_score = cls_score * obj_score
                class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                logger.info(f"    p5 -{i}th batch_set: {p5[0]}, batch_item: {p5[1]}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                            obj_score: {obj_score:.3f}, \
                            cls_score: {cls_score:.3f}, \
                            conf_score: {conf_score:.6f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name}, \
                            feat_num: {feat_num}")
             break
    return sampled_mem_feat_info
def read_updated_feature_info_list(args,check_batch_set, check_batch_item, save_folder):
    updated_feat_info_save_path = os.path.join(args.input_dir, "my_model_updated_feat_info.pkl")
    with open(updated_feat_info_save_path, "rb") as f:
            updated_feat_info_list = pickle.load(f)
    logger.info(f"len(updated_feat_info_list): {len(updated_feat_info_list)}")
    check_batch_set_num = check_batch_set
    check_batch_item_num = check_batch_item

    for batch_set, updated_feat_info_set in enumerate(updated_feat_info_list):
        if batch_set == check_batch_set_num:
            logger.info(f"Batch set {batch_set} - num batch items: {len(updated_feat_info_set)}")
        for batch_item, updated_feat_info_item in enumerate(updated_feat_info_set):
            if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                logger.info(f"  Batch item {batch_item} - num features: {len(updated_feat_info_item)}")
                p3_list, p4_list, p5_list = updated_feat_info_item
                logger.info(f"  Batch item {batch_item} - p3: {len(p3_list)} items,\
                             p4: {len(p4_list)} items, p5: {len(p5_list)} items")
                for i, p3 in enumerate(p3_list):
                     if np.all(p3 == 0):
                         continue
                     bbox = p3[:4]
                     obj_score = p3[4]
                     cls_score = p3[5]
                     class_pred = p3[6: 6 + len(VID_classes)]
                     class_label = int(np.argmax(class_pred))
                     feat_num = int(p3[6 + len(VID_classes)])
                     conf_score = cls_score * obj_score
                     class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                     logger.info(f"    p3 -{i}th batch_set: {batch_set}, batch_item: {batch_item}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                 obj_score: {obj_score:.3f}, \
                                 cls_score: {cls_score:.3f}, \
                                 conf_score: {conf_score:.6f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}, \
                                 feat_num: {feat_num}")
                for i, p4 in enumerate(p4_list):
                     if np.all(p4 == 0):
                         continue
                     bbox = p4[:4]
                     obj_score = p4[4]
                     cls_score = p4[5]
                     class_pred = p4[6: 6 + len(VID_classes)]
                     class_label = int(np.argmax(class_pred))
                     feat_num = int(p4[6 + len(VID_classes)])
                     conf_score = cls_score * obj_score
                     class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                     logger.info(f"    p4 -{i}th batch_set: {batch_set}, batch_item: {batch_item}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                 obj_score: {obj_score:.3f}, \
                                 cls_score: {cls_score:.3f}, \
                                 conf_score: {conf_score:.6f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}, \
                                 feat_num: {feat_num}")
                for i, p5 in enumerate(p5_list):
                     if np.all(p5 == 0):
                         continue
                     bbox = p5[:4]
                     obj_score = p5[4]
                     cls_score = p5[5]
                     class_pred = p5[6: 6 + len(VID_classes)]
                     class_label = int(np.argmax(class_pred))
                     feat_num = int(p5[6 + len(VID_classes)])
                     conf_score = cls_score * obj_score
                     class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                     logger.info(f"    p5 -{i}th batch_set: {batch_set}, batch_item: {batch_item}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                 obj_score: {obj_score:.3f}, \
                                 cls_score: {cls_score:.3f}, \
                                 conf_score: {conf_score:.6f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}, \
                                 feat_num: {feat_num}")
                if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                    break
        if batch_set == check_batch_set_num:
            break
    return updated_feat_info_item
def read_agg_feature_info_list(args, check_batch_set, check_batch_item, save_folder):
    agg_feat_info_save_path = os.path.join(args.input_dir, "my_model_agg_feat_info.pkl")
    with open(agg_feat_info_save_path, "rb") as f:
            agg_feat_info_list = pickle.load(f)
    logger.info(f"len(agg_feat_info_list): {len(agg_feat_info_list)}")
    check_batch_set_num = check_batch_set
    check_batch_item_num = check_batch_item

    for batch_set, agg_feat_info_set in enumerate(agg_feat_info_list):
        if batch_set == check_batch_set_num:
            logger.info(f"Batch set {batch_set} - num batch items: {len(agg_feat_info_set)}")
        for batch_item, agg_feat_info_item in enumerate(agg_feat_info_set):
            if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                logger.info(f"  Batch item {batch_item} - num features: {len(agg_feat_info_item)}")
                p3_list, p4_list, p5_list = agg_feat_info_item
                logger.info(f"  Batch item {batch_item} - p3: {len(p3_list)} items,\
                             p4: {len(p4_list)} items, p5: {len(p5_list)} items")
                for i, p3 in enumerate(p3_list):
                     if np.all(p3 == 0):
                         continue
                     bbox = p3[:4]
                     obj_score = p3[4]
                     cls_score = p3[5]
                     class_pred = p3[6: 6 + len(VID_classes)]
                     class_label = int(np.argmax(class_pred))
                     feat_num = int(p3[6 + len(VID_classes)])
                     conf_score = cls_score * obj_score
                     class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                     logger.info(f"    p3 -{i}th batch_set: {batch_set}, batch_item: {batch_item}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                 obj_score: {obj_score:.3f}, \
                                 cls_score: {cls_score:.3f}, \
                                 conf_score: {conf_score:.6f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}, \
                                 feat_num: {feat_num}")
                for i, p4 in enumerate(p4_list):
                     if np.all(p4 == 0):
                         continue
                     bbox = p4[:4]
                     obj_score = p4[4]
                     cls_score = p4[5]
                     class_pred = p4[6: 6 + len(VID_classes)]
                     class_label = int(np.argmax(class_pred))
                     feat_num = int(p4[6 + len(VID_classes)])
                     conf_score = cls_score * obj_score
                     class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                     logger.info(f"    p4 -{i}th batch_set: {batch_set}, batch_item: {batch_item}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                 obj_score: {obj_score:.3f}, \
                                 cls_score: {cls_score:.3f}, \
                                 conf_score: {conf_score:.6f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}, \
                                 feat_num: {feat_num}")
                for i, p5 in enumerate(p5_list):
                     if np.all(p5 == 0):
                         continue
                     bbox = p5[:4]
                     obj_score = p5[4]
                     cls_score = p5[5]
                     class_pred = p5[6: 6 + len(VID_classes)]
                     class_label = int(np.argmax(class_pred))
                     feat_num = int(p5[6 + len(VID_classes)])
                     conf_score = cls_score * obj_score
                     class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                     logger.info(f"    p5 -{i}th batch_set: {batch_set}, batch_item: {batch_item}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                 obj_score: {obj_score:.3f}, \
                                 cls_score: {cls_score:.3f}, \
                                 conf_score: {conf_score:.6f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}, \
                                 feat_num: {feat_num}")
                if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                    break
        if batch_set == check_batch_set_num:
            break
    return agg_feat_info_item
def read_result_info_list(args):
    result_info_save_path = os.path.join(args.input_dir, "my_model_result_info.pkl")
    logger.info(f'Reading result info from: {result_info_save_path}')
    with open(result_info_save_path, "rb") as f:
            result_info_list = pickle.load(f)
    for file_name_in_result, result_info in result_info_list.items():
        logger.info(f"Checking result info for file name in result: {file_name_in_result}")
        if file_name_in_result == args.input_frame_name:
            logger.info(f"Found result info for input frame {args.input_frame_name}")
            for i, result in enumerate(result_info):
                    class_label = int(result[5])
                    class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                    logger.info(f"Result info -{i}th bbox: ({int(result[0])}, {int(result[1])}, {int(result[2])}, {int(result[3])}), \
                        cls_score: {result[4]:.3f}, \
                        class_label: {class_label}, \
                        class_label_name: {class_label_name}")
            break
    return result_info
def visualize_mem_info_on_frame(args, type_name, feat_info_list, frame_save_path, exp=None, ref_frame_batch_set=None):
    batch_size = 16
    p3_mem_info = feat_info_list['p3']
    p4_mem_info = feat_info_list['p4']
    p5_mem_info = feat_info_list['p5']
    draw_img_flag = False
    if args.draw_mem_feat_info and type_name == "Memory":
        draw_img_flag = True
    elif args.draw_sampled_mem_feat_info and type_name == "Sampled_Memory":
        draw_img_flag = True
    if args.save_csv_result:
        csv_save_path = os.path.join(frame_save_path, f"{type_name}_features.csv")
        csv_file = open(csv_save_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['batch_set', 'batch_item', 'image_name', 'feature_level', 'bbox', 'obj_score', 'cls_score', 'conf_score', 'class_label', 'class_label_name', 'feat_num'])

    for i, p3 in enumerate(p3_mem_info):
         if torch.all(p3[2] == 0):
             continue
         input_frame_path = ref_frame_batch_set[p3[0]][p3[1]]
         input_frame_name = os.path.basename(input_frame_path)
         if draw_img_flag:
            logger.info(f"Visualizing feature info for p3 - {i}th batch_set: {p3[0]}, batch_item: {p3[1]}")
            logger.info(f"Input frame path: {input_frame_path}")
            logger.info(f"Visualizing feature info on frame: {input_frame_path}")
         frame = cv2.imread(input_frame_path)
         height, width = frame.shape[:2]
         if exp is not None:
            ratio = min(exp.test_size[0] / height, exp.test_size[1] / width)
         else:
            ratio = 1.0
         bbox = p3[2][:4] / ratio
         obj_score = p3[2][4]
         cls_score = p3[2][5]
         class_pred = p3[2][6: 6 + len(VID_classes)]
         class_label = int(torch.argmax(class_pred))
         feat_num = int(p3[2][6 + len(VID_classes)])
         conf_score = cls_score * obj_score
         class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
         label_text = f"{class_label_name} {conf_score:.3f}"
         logger.info(f"    p3 -{i}th batch_set: {p3[0]}, batch_item: {p3[1]},\
                     image_name: {input_frame_name}, \
                     im bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                    obj_score: {obj_score:.3f}, \
                    cls_score: {cls_score:.3f}, \
                    conf_score: {conf_score:.6f}, \
                    class_label: {class_label}, \
                    class_label_name: {class_label_name}, \
                    feat_num: {feat_num}")
         if draw_img_flag:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            file_name = input_frame_name + f'_{type_name}_P3_{i}.JPEG'
            cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
            logger.info(f"Saved visualized frame with p3 feature info: {os.path.join(frame_save_path, file_name)}")
            logger.info(f"Finished visualizing p3 feature info for {input_frame_path}")

         if args.save_csv_result:
             csv_writer.writerow([p3[0], p3[1], input_frame_name, 'P3', f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]", f"{obj_score:.3f}", f"{cls_score:.3f}", f"{conf_score:.6f}", \
                                   f"{class_label}", f"{class_label_name}", f"{feat_num}"])
    for i, p4 in enumerate(p4_mem_info):
         if torch.all(p4[2] == 0):
             continue
         input_frame_path = ref_frame_batch_set[p4[0]][p4[1]]
         input_frame_name = os.path.basename(input_frame_path)
         logger.info(f"Visualizing feature info on frame: {input_frame_path}")

         frame = cv2.imread(input_frame_path)
         height, width = frame.shape[:2]
         if exp is not None:
            ratio = min(exp.test_size[0] / height, exp.test_size[1] / width)
         else:
            ratio = 1.0
         bbox = p4[2][:4] / ratio
         obj_score = p4[2][4]
         cls_score = p4[2][5]
         class_pred = p4[2][6: 6 + len(VID_classes)]
         class_label = int(torch.argmax(class_pred))
         feat_num = int(p4[2][6 + len(VID_classes)])
         conf_score = cls_score * obj_score
         class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
         label_text = f"{class_label_name} {conf_score:.3f}"
         logger.info(f"    p4 -{i}th batch_set: {p4[0]}, batch_item: {p4[1]}, \
                     image_name: {input_frame_name}, \
                     bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                    obj_score: {obj_score:.3f}, \
                    cls_score: {cls_score:.3f}, \
                    conf_score: {conf_score:.6f}, \
                    class_label: {class_label}, \
                    class_label_name: {class_label_name}, \
                    feat_num: {feat_num}")
         if draw_img_flag:
             cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
             cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
             file_name = input_frame_name + f'_{type_name}_P4_{i}.JPEG'
             cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
         if args.save_csv_result:
             csv_writer.writerow([p4[0], p4[1], input_frame_name, 'P4', f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]", f"{obj_score:.3f}", f"{cls_score:.3f}", f"{conf_score:.6f}", \
                                     f"{class_label}", f"{class_label_name}", f"{feat_num}"])

    for i, p5 in enumerate(p5_mem_info):
         if torch.all(p5[2] == 0):
             continue
         input_frame_path = ref_frame_batch_set[p5[0]][p5[1]]
         input_frame_name = os.path.basename(input_frame_path)
         logger.info(f"Visualizing feature info on frame: {input_frame_path}")

         frame = cv2.imread(input_frame_path)
         height, width = frame.shape[:2]
         if exp is not None:
            ratio = min(exp.test_size[0] / height, exp.test_size[1] / width)
         else:
            ratio = 1.0
         bbox = p5[2][:4] / ratio
         obj_score = p5[2][4]
         cls_score = p5[2][5]
         class_pred = p5[2][6: 6 + len(VID_classes)]
         class_label = int(torch.argmax(class_pred))
         feat_num = int(p5[2][6 + len(VID_classes)])
         conf_score = cls_score * obj_score
         class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
         label_text = f"{class_label_name} {conf_score:.3f}"
         logger.info(f"    p5 -{i}th batch_set: {p5[0]}, batch_item: {p5[1]}, \
                     image_name: {input_frame_name}, \
                     bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                    obj_score: {obj_score:.3f}, \
                    cls_score: {cls_score:.3f}, \
                    conf_score: {conf_score:.6f}, \
                    class_label: {class_label}, \
                    class_label_name: {class_label_name}, \
                    feat_num: {feat_num}")
         if draw_img_flag:
             cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
             cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
             file_name = input_frame_name + f'_{type_name}_P5_{i}.JPEG'
             cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
         if args.save_csv_result:
             csv_writer.writerow([p5[0], p5[1], input_frame_name, 'P5', f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]", f"{obj_score:.3f}", f"{cls_score:.3f}", f"{conf_score:.6f}", \
                                   f"{class_label}", f"{class_label_name}", f"{feat_num}"])
    if args.save_csv_result:
        csv_file.close()
        logger.info(f"Finished visualizing mem feature info and saved to: {frame_save_path}")
def visualize_feature_info_on_frame(args, type_name, feat_info_list, frame_save_path,exp=None):

    draw_image_flag = False
    if args.draw_input_feat_info and type_name == "Input":
        draw_image_flag = True
    elif args.draw_updated_feat_info and type_name == "Updated":
        draw_image_flag = True
    elif args.draw_agg_feat_info and type_name == "Agg":
        draw_image_flag = True

    input_frame_path = os.path.join(args.input_image_path, args.input_frame_name)
    logger.info(f"Visualizing feature info on frame: {input_frame_path}")
    _frame = cv2.imread(input_frame_path)
    height, width = _frame.shape[:2]
    if exp is not None:
        ratio = min(exp.test_size[0] / height, exp.test_size[1] / width)
    else:
        ratio = 1.0
    if args.save_csv_result:
        csv_save_path = os.path.join(frame_save_path, f"{type_name}_features.csv")
        csv_file = open(csv_save_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["level", "bbox", "obj_score", "cls_score", "conf_score", "class_pred", "class_label", "class_label_name", "feat_num"])

    for feat_info in feat_info_list:
            p3_list, p4_list, p5_list = feat_info
            for i, p3 in enumerate(p3_list):
                 if np.all(p3 == 0):
                     continue
                 if draw_image_flag:
                    frame = cv2.imread(input_frame_path)
                 bbox = p3[:4] / ratio
                 bbox[[0, 2]] = np.clip(bbox[[0, 2]], a_min=0, a_max=None)  # clip x1, x2
                 bbox[[1, 3]] = np.clip(bbox[[1, 3]], a_min=0, a_max=None)  # clip y1, y2
                 obj_score = p3[4]
                 cls_score = p3[5]
                 class_pred = p3[6: 6+len(VID_classes)]
                 class_label = int(np.argmax(class_pred))
                 feat_num = int(p3[6 + len(VID_classes)])
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 label_text = f"{class_label_name} {conf_score:.3f}"
                 logger.info(f"    p3 -{i}th  bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                            obj_score: {obj_score:.3f}, \
                            cls_score: {cls_score:.3f}, \
                            conf_score: {conf_score:.6f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name},\
                            feat_num: {feat_num}")
                 if draw_image_flag:
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    file_name = args.input_frame_name + f'_{type_name}_P3_{i}.JPEG'

                    if args.save_result_with_gt:
                        xml_path = input_frame_path.replace("Data", "Annotations").replace(".JPEG", ".xml").replace(".jpeg", ".xml").replace(".jpg", ".xml")
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
                    cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
                 if args.save_csv_result:
                    csv_writer.writerow(["P3", f"[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]",
                                        f"{obj_score:.3f}", f"{cls_score:.3f}", f"{conf_score:.6f}",\
                                            f"{np.array2string(class_pred, precision=6, floatmode='fixed', suppress_small=True)}", f"{class_label}", f"{class_label_name}", f"{feat_num}"])

            for i, p4 in enumerate(p4_list):
                 if np.all(p4 == 0):
                     continue
                 if draw_image_flag:
                    frame = cv2.imread(input_frame_path)
                 bbox = p4[:4] / ratio
                 bbox[[0, 2]] = np.clip(bbox[[0, 2]], a_min=0, a_max=None)  # clip x1, x2
                 bbox[[1, 3]] = np.clip(bbox[[1, 3]], a_min=0, a_max=None)  # clip y1, y2
                 obj_score = p4[4]
                 cls_score = p4[5]
                 class_pred = p4[6: 6 + len(VID_classes)]
                 class_label = int(np.argmax(class_pred))
                 feat_num = int(p4[6 + len(VID_classes)])
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 label_text = f"{class_label_name} {conf_score:.3f}"
                 if draw_image_flag:
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    file_name = args.input_frame_name + f'_{type_name}_P4_{i}.JPEG'

                    if args.save_result_with_gt:
                        xml_path = input_frame_path.replace("Data", "Annotations").replace(".JPEG", ".xml").replace(".jpeg", ".xml").replace(".jpg", ".xml")
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
                    cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
                 logger.info(f"    p4 -{i}th  bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                            obj_score: {obj_score:.3f}, \
                            cls_score: {cls_score:.3f}, \
                            conf_score: {conf_score:.6f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name},\
                            feat_num: {feat_num}")
                 if args.save_csv_result:
                     csv_writer.writerow(["P4", f"[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]",
                                          f"{obj_score:.3f}", f"{cls_score:.3f}", f"{conf_score:.6f}",\
                                              f"{np.array2string(class_pred, precision=6, floatmode='fixed', suppress_small=True)}", f"{class_label}", f"{class_label_name}", f"{feat_num}"])

            for i, p5 in enumerate(p5_list):
                 if np.all(p5 == 0):
                     continue
                 if draw_image_flag:
                    frame = cv2.imread(input_frame_path)
                 bbox = p5[:4] / ratio
                 bbox[[0, 2]] = np.clip(bbox[[0, 2]], a_min=0, a_max=None)  # clip x1, x2
                 bbox[[1, 3]] = np.clip(bbox[[1, 3]], a_min=0, a_max=None)  # clip y1, y2
                 obj_score = p5[4]
                 cls_score = p5[5]
                 class_pred = p5[6: 6+len(VID_classes)]
                 class_label = int(np.argmax(class_pred))
                 feat_num = int(p5[6 + len(VID_classes)])
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 label_text = f"{class_label_name} {conf_score:.3f}"
                 if draw_image_flag:
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    file_name = args.input_frame_name + f'_{type_name}_P5_{i}.JPEG'

                    if args.save_result_with_gt:
                        xml_path = input_frame_path.replace("Data", "Annotations").replace(".JPEG", ".xml").replace(".jpeg", ".xml").replace(".jpg", ".xml")
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
                    cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
                 logger.info(f"    p5 -{i}th bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                                obj_score: {obj_score:.3f}, \
                                cls_score: {cls_score:.3f}, \
                                conf_score: {conf_score:.6f}, \
                                class_label: {class_label}, \
                                class_label_name: {class_label_name},\
                                feat_num: {feat_num}")
                 if args.save_csv_result:
                     csv_writer.writerow(["P5", f"[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]",
                                          f"{obj_score:.3f}", f"{cls_score:.3f}", f"{conf_score:.6f}",\
                                              f"{np.array2string(class_pred, precision=6, floatmode='fixed', suppress_small=True)}", f"{class_label}", f"{class_label_name}", f"{feat_num}"])
    if args.save_csv_result:
        csv_file.close()
        logger.info(f"Saved input feature info to CSV: {csv_save_path}")

def visualize_result_info_on_frame(args, result_info, frame_save_path):
    input_frame_path = os.path.join(args.input_image_path, args.input_frame_name)
    logger.info(f"Visualizing result info on frame: {input_frame_path}")

    for feat_info in result_info:
            for i, result in enumerate(feat_info):
                 if result[4] < args.test_conf:
                      continue
                 frame = cv2.imread(input_frame_path)
                 bbox = result[:4]
                 cls_score = result[4]
                 class_label = int(result[5])
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 label_text = f"{class_label_name} {cls_score:.3f}"
                 cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                 cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                 file_name = args.input_frame_name + f'_Result_{i}.JPEG'
                 if args.save_result_with_gt:
                    xml_path = input_frame_path.replace("Data", "Annotations").replace(".JPEG", ".xml").replace(".jpeg", ".xml").replace(".jpg", ".xml")
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
                 cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
                 logger.info(f"    Result -{i}th bbox: ({int(result[0])}, {int(result[1])}, {int(result[2])}, {int(result[3])}), \
                                cls_score: {cls_score:.3f}, \
                                class_label: {class_label}, \
                                class_label_name: {class_label_name}")
def visualize_outputs_info_on_frame(args, outputs_info, frame_save_path, exp=None, batch_set=None, batch_item=None):
    input_frame_path = os.path.join(args.input_image_path, args.input_frame_name)
    logger.info(f"Visualizing outputs info on frame: {input_frame_path}")
    frame = cv2.imread(input_frame_path)
    if args.save_csv_result:
        csv_save_path = os.path.join(frame_save_path, "outputs_features.csv")
        csv_file = open(csv_save_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["level", "bbox", "obj_score", "cls_score", "conf_score", "class_label", "class_label_name", "feat_num", "IoU"])
    height, width = frame.shape[:2]
    if exp is not None:
        ratio = min(exp.test_size[0] / height, exp.test_size[1] / width)
    else:
        ratio = 1.0
    logger.info(f"height: {height}, width: {width}, ratio: {ratio}")
    for i, output_info in enumerate(outputs_info):
        if args.draw_outputs_feat_info:
            frame = cv2.imread(input_frame_path)

        logger.info(f"    output_info -{i}th - len: {len(output_info)}")
        logger.info(f"    output_info -{i}th - content: {output_info}")
        batch_set_output = output_info[0]
        batch_item_output = output_info[1]
        if batch_set_output != batch_set or batch_item_output != batch_item:
            logger.info(f"Skipping output info -{i}th because batch_set_output: {batch_set_output}, batch_item_output: {batch_item_output} do not match input frame's batch_set: {batch_set}, batch_item: {batch_item}")
            continue
        bbox = np.array([v / ratio for v in output_info[2:6]])
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], a_min=0, a_max=None)  # clip x1, x2
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], a_min=0, a_max=None)  # clip y1, y2
        obj_score = float(output_info[6])
        cls_score = float(output_info[7])
        class_label = int(output_info[8])
        feat_num = int(output_info[9]) if output_info[9] is not None else -1
        conf_score = cls_score * obj_score
        class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
        label_text = f"{class_label_name} {conf_score:.3f}"
        logger.info(f"    output_info -{i}th - batch_set: {batch_set_output}, batch_item: {batch_item_output}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                    obj_score: {obj_score:.3f}, \
                    cls_score: {cls_score:.3f}, \
                    conf_score: {conf_score:.6f}, \
                    class_label: {class_label}, \
                    class_label_name: {class_label_name}, \
                    feat_num: {feat_num}")

        if args.draw_outputs_feat_info:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            file_name = args.input_frame_name + f'_Output_Info_{i}.JPEG'


            if args.save_result_with_gt:
                xml_path = input_frame_path.replace("Data", "Annotations").replace(".JPEG", ".xml").replace(".jpeg", ".xml").replace(".jpg", ".xml")
                if os.path.exists(xml_path):
                    xml_doc = minidom.parse(xml_path)
                    iou_max = 0.0
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
                        iou = calculate_iou_gt_img((xmin, ymin, xmax, ymax), (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                        iou_max = max(iou_max, iou)
                else:
                    logger.warning("GT xml not found: {}".format(xml_path))

            cv2.imwrite(os.path.join(frame_save_path, file_name), frame)

        if feat_num >= 0 and feat_num < 6400:  # Assuming feat_num corresponds to P3, P4, P5 respectively
            level = "P3"
        elif feat_num >= 6400 and feat_num < 6400 + 1600:
            level = "P4"
        else:
            level = "P5"
        if args.save_csv_result:
            if args.draw_outputs_feat_info:
                csv_writer.writerow([level, f"[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]",
                                         f"{obj_score:.3f}", f"{cls_score:.3f}", f"{conf_score:.6f}", class_label, class_label_name, feat_num, f"{iou_max:.3f}"])
            else:
                csv_writer.writerow([level, f"[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]",
                                         f"{obj_score:.3f}", f"{cls_score:.3f}", f"{conf_score:.6f}", class_label, class_label_name, feat_num])

    if args.save_csv_result:
        csv_file.close()
        logger.info(f"Saved outputs feature info to CSV: {csv_save_path}")

def calculate_mAP(args, outputs_info, save_folder, iou_threshold=0.5):
    input_frame_path = os.path.join(args.input_image_path, args.input_frame_name)
    frame = cv2.imread(input_frame_path)
    if frame is None:
        logger.warning(f"calculate_mAP: cannot read image {input_frame_path}")
        return
    height, width = frame.shape[:2]

    # Derive ratio the same way visualize_outputs_info_on_frame does (exp not available here, use tsize)
    tsize = getattr(args, 'tsize', 640)
    ratio = min(tsize / height, tsize / width)

    # --- Build predictions per class: {cls: [(conf, x1, y1, x2, y2), ...]} ---
    preds_by_class = {}
    for output_info in outputs_info:
        bbox_scaled = output_info[2:6]
        bbox = [v / ratio for v in bbox_scaled]
        obj_score  = float(output_info[6])
        cls_score  = float(output_info[7])
        class_label = int(output_info[8])
        conf = obj_score * cls_score
        preds_by_class.setdefault(class_label, []).append(
            (conf, bbox[0], bbox[1], bbox[2], bbox[3])
        )

    # --- Load GT from XML ---
    xml_path = (input_frame_path
                .replace("Data", "Annotations")
                .replace(".JPEG", ".xml")
                .replace(".jpeg", ".xml")
                .replace(".jpg", ".xml"))
    gts_by_class = {}
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
            if cls_idx < 0:
                continue
            gts_by_class.setdefault(cls_idx, []).append((xmin, ymin, xmax, ymax))
    else:
        logger.warning(f"calculate_mAP: GT xml not found: {xml_path}")
        return

    # --- Compute AP per class ---
    all_classes = set(preds_by_class.keys()) | set(gts_by_class.keys())
    ap_list = []
    tp_iou_all = []  # IoU of every true positive across all classes
    for cls in sorted(all_classes):
        gt_boxes = gts_by_class.get(cls, [])
        pred_list = sorted(preds_by_class.get(cls, []), key=lambda x: -x[0])  # sort by conf desc
        n_gt = len(gt_boxes)
        if n_gt == 0:
            continue

        matched = [False] * n_gt
        tp = np.zeros(len(pred_list))
        fp = np.zeros(len(pred_list))
        tp_ious = []

        for pi, (conf, px1, py1, px2, py2) in enumerate(pred_list):
            best_iou = 0.0
            best_gi = -1
            for gi, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                iou = calculate_iou_gt_img((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_threshold and not matched[best_gi]:
                tp[pi] = 1
                matched[best_gi] = True
                tp_ious.append(best_iou)
                logger.info(f"  TP: pred ({px1:.1f},{py1:.1f},{px2:.1f},{py2:.1f}) "
                            f"conf={conf:.3f} GT=({gt_boxes[best_gi]}) IoU={best_iou:.4f}")
            else:
                fp[pi] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall    = tp_cum / (n_gt + 1e-12)
        precision = tp_cum / (tp_cum + fp_cum + 1e-12)

        # AP via 11-point interpolation
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            prec_at_rec = precision[recall >= thr]
            ap += (np.max(prec_at_rec) if len(prec_at_rec) > 0 else 0.0)
        ap /= 11.0

        cls_avg_tp_iou = float(np.mean(tp_ious)) if tp_ious else 0.0
        cls_name = VID_classes[cls] if cls < len(VID_classes) else str(cls)
        logger.info(f"  AP[{cls_name}]: {ap:.4f}  avg_TP_IoU={cls_avg_tp_iou:.4f}  "
                    f"(n_gt={n_gt}, n_pred={len(pred_list)}, n_tp={len(tp_ious)})")
        ap_list.append(ap)
        tp_iou_all.extend(tp_ious)

    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    avg_tp_iou = float(np.mean(tp_iou_all)) if tp_iou_all else 0.0
    logger.info(f"mAP@{iou_threshold}: {mAP:.4f}  (over {len(ap_list)} classes)")
    logger.info(f"Avg TP IoU: {avg_tp_iou:.4f}  (n_tp={len(tp_iou_all)})")

    mAP_save_path = os.path.join(save_folder, "mAP_result.txt")
    with open(mAP_save_path, "w") as f:
        f.write(f"mAP@{iou_threshold}: {mAP:.4f}\n")
        f.write(f"Avg TP IoU: {avg_tp_iou:.4f}  (n_tp={len(tp_iou_all)})\n")
        for cls, ap in zip(sorted(all_classes & set(gts_by_class.keys())), ap_list):
            cls_name = VID_classes[cls] if cls < len(VID_classes) else str(cls)
            f.write(f"  AP[{cls_name}]: {ap:.4f}\n")
    logger.info(f"Saved mAP result to {mAP_save_path}")
    return mAP, avg_tp_iou


def calculate_AverageIou(args, outputs_info, save_folder):
    input_frame_path = os.path.join(args.input_image_path, args.input_frame_name)
    frame = cv2.imread(input_frame_path)
    if frame is None:
        logger.warning(f"calculate_AverageIou: cannot read image {input_frame_path}")
        return
    height, width = frame.shape[:2]
    tsize = getattr(args, 'tsize', 640)
    ratio = min(tsize / height, tsize / width)

    # --- Build predictions: [(conf, cls, x1, y1, x2, y2), ...] ---
    predictions = []
    for output_info in outputs_info:
        bbox = [v / ratio for v in output_info[2:6]]
        obj_score   = float(output_info[6])
        cls_score   = float(output_info[7])
        class_label = int(output_info[8])
        conf = obj_score * cls_score
        predictions.append((conf, class_label, bbox[0], bbox[1], bbox[2], bbox[3]))

    # --- Load GT from XML ---
    xml_path = (input_frame_path
                .replace("Data", "Annotations")
                .replace(".JPEG", ".xml")
                .replace(".jpeg", ".xml")
                .replace(".jpg", ".xml"))
    gt_boxes = []  # [(cls, x1, y1, x2, y2), ...]
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
            if cls_idx >= 0:
                gt_boxes.append((cls_idx, xmin, ymin, xmax, ymax))
    else:
        logger.warning(f"calculate_AverageIou: GT xml not found: {xml_path}")
        return

    if not gt_boxes:
        logger.warning("calculate_AverageIou: no GT boxes found")
        return

    # --- For each prediction, best-matching GT IoU ---
    pred_max_ious = []     # max IoU over all GT for each prediction
    pred_cls_ious = {}     # per-class: list of max IoUs

    for conf, cls, px1, py1, px2, py2 in predictions:
        best_iou = 0.0
        for gt_cls, gx1, gy1, gx2, gy2 in gt_boxes:
            iou = calculate_iou_gt_img((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
            if iou > best_iou:
                best_iou = iou
        pred_max_ious.append(best_iou)
        pred_cls_ious.setdefault(cls, []).append(best_iou)
        cls_name = VID_classes[cls] if cls < len(VID_classes) else str(cls)
        logger.info(f"  pred cls={cls_name} conf={conf:.4f} bbox=({int(px1)},{int(py1)},{int(px2)},{int(py2)}) best_IoU={best_iou:.4f}")

    # --- For each GT, best-matching prediction IoU ---
    gt_max_ious = []
    gt_cls_ious = {}

    for gt_cls, gx1, gy1, gx2, gy2 in gt_boxes:
        best_iou = 0.0
        for conf, cls, px1, py1, px2, py2 in predictions:
            iou = calculate_iou_gt_img((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
            if iou > best_iou:
                best_iou = iou
        gt_max_ious.append(best_iou)
        gt_cls_ious.setdefault(gt_cls, []).append(best_iou)
        cls_name = VID_classes[gt_cls] if gt_cls < len(VID_classes) else str(gt_cls)
        logger.info(f"  GT   cls={cls_name} bbox=({gx1},{gy1},{gx2},{gy2}) best_IoU={best_iou:.4f}")

    avg_iou_pred = float(np.mean(pred_max_ious)) if pred_max_ious else 0.0
    avg_iou_gt   = float(np.mean(gt_max_ious))   if gt_max_ious   else 0.0

    logger.info(f"Average IoU (pred→GT): {avg_iou_pred:.4f}  (n_pred={len(predictions)})")
    logger.info(f"Average IoU (GT→pred): {avg_iou_gt:.4f}  (n_gt={len(gt_boxes)})")

    for cls in sorted(set(pred_cls_ious) | set(gt_cls_ious)):
        cls_name = VID_classes[cls] if cls < len(VID_classes) else str(cls)
        p_ious = pred_cls_ious.get(cls, [])
        g_ious = gt_cls_ious.get(cls, [])
        logger.info(f"  [{cls_name}] avg pred→GT IoU={np.mean(p_ious):.4f} (n={len(p_ious)})  "
                    f"avg GT→pred IoU={np.mean(g_ious):.4f} (n={len(g_ious)})")

    save_path = os.path.join(save_folder, "average_iou_result.txt")
    with open(save_path, "w") as f:
        f.write(f"Average IoU (pred->GT): {avg_iou_pred:.4f}  n_pred={len(predictions)}\n")
        f.write(f"Average IoU (GT->pred): {avg_iou_gt:.4f}  n_gt={len(gt_boxes)}\n\n")
        f.write("Per-class:\n")
        for cls in sorted(set(pred_cls_ious) | set(gt_cls_ious)):
            cls_name = VID_classes[cls] if cls < len(VID_classes) else str(cls)
            p_ious = pred_cls_ious.get(cls, [])
            g_ious = gt_cls_ious.get(cls, [])
            f.write(f"  [{cls_name}] pred->GT={np.mean(p_ious):.4f} (n={len(p_ious)})  "
                    f"GT->pred={np.mean(g_ious):.4f} (n={len(g_ious)})\n")
    logger.info(f"Saved average IoU result to {save_path}")
    return avg_iou_pred, avg_iou_gt


def calculate_meanIoU(args, outputs_info, save_folder):
    input_frame_path = os.path.join(args.input_image_path, args.input_frame_name)
    frame = cv2.imread(input_frame_path)
    if frame is None:
        logger.warning(f"calculate_meanIoU: cannot read image {input_frame_path}")
        return
    height, width = frame.shape[:2]
    tsize = getattr(args, 'tsize', 640)
    ratio = min(tsize / height, tsize / width)

    pred_boxes = [[v / ratio for v in output_info[2:6]] for output_info in outputs_info]

    xml_path = (input_frame_path
                .replace("Data", "Annotations")
                .replace(".JPEG", ".xml")
                .replace(".jpeg", ".xml")
                .replace(".jpg", ".xml"))
    gt_boxes = []
    if os.path.exists(xml_path):
        xml_doc = minidom.parse(xml_path)
        root = xml_doc.documentElement
        for obj in root.getElementsByTagName("object"):
            xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
            ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
            xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
            ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
            gt_boxes.append((xmin, ymin, xmax, ymax))
    else:
        logger.warning(f"calculate_meanIoU: GT xml not found: {xml_path}")
        return

    if not gt_boxes:
        logger.warning("calculate_meanIoU: no GT boxes found")
        return
    if not pred_boxes:
        logger.warning("calculate_meanIoU: no predictions found")
        return

    # Compute IoU between every (pred, GT) pair and sum all values
    total_iou = 0.0
    for pi, (px1, py1, px2, py2) in enumerate(pred_boxes):
        for gj, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            iou = calculate_iou_gt_img((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
            logger.info(f"  pred[{pi}] bbox=({int(px1)},{int(py1)},{int(px2)},{int(py2)})  "
                        f"GT[{gj}] bbox=({gx1},{gy1},{gx2},{gy2})  IoU={iou:.4f}")
            total_iou += iou

    mean_iou = total_iou / len(pred_boxes)
    logger.info(f"meanIoU: {total_iou:.4f} / {len(pred_boxes)} preds = {mean_iou:.4f}  "
                f"(n_gt={len(gt_boxes)}, n_pred={len(pred_boxes)})")

    save_path = os.path.join(save_folder, "mean_iou_result.txt")
    with open(save_path, "w") as f:
        f.write(f"total_iou={total_iou:.4f}  n_pred={len(pred_boxes)}  n_gt={len(gt_boxes)}\n")
        f.write(f"meanIoU (total_iou / n_pred): {mean_iou:.4f}\n")
    logger.info(f"Saved meanIoU result to {save_path}")
    return mean_iou


def calculate_localization_accuracy(args, outputs_info, save_folder, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    input_frame_path = os.path.join(args.input_image_path, args.input_frame_name)
    frame = cv2.imread(input_frame_path)
    if frame is None:
        logger.warning(f"calculate_localization_accuracy: cannot read image {input_frame_path}")
        return
    height, width = frame.shape[:2]
    tsize = getattr(args, 'tsize', 640)
    ratio = min(tsize / height, tsize / width)

    pred_boxes = [[v / ratio for v in output_info[2:6]] for output_info in outputs_info]

    xml_path = (input_frame_path
                .replace("Data", "Annotations")
                .replace(".JPEG", ".xml")
                .replace(".jpeg", ".xml")
                .replace(".jpg", ".xml"))
    gt_boxes = []
    if os.path.exists(xml_path):
        xml_doc = minidom.parse(xml_path)
        root = xml_doc.documentElement
        for obj in root.getElementsByTagName("object"):
            xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
            ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
            xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
            ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
            gt_boxes.append((xmin, ymin, xmax, ymax))
    else:
        logger.warning(f"calculate_localization_accuracy: GT xml not found: {xml_path}")
        return

    if not gt_boxes or not pred_boxes:
        logger.warning("calculate_localization_accuracy: empty GT or predictions")
        return

    # --- Mean Best IoU (GT-anchored) ---
    # For each GT box, find the prediction with the highest IoU.
    # Invariant to n_pred: extra predictions do not affect this score.
    best_ious = []
    for gi, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
        best_iou = max(
            calculate_iou_gt_img((gx1, gy1, gx2, gy2), p) for p in pred_boxes
        )
        best_ious.append(best_iou)
        logger.info(f"  GT[{gi}] bbox=({gx1},{gy1},{gx2},{gy2})  best_IoU={best_iou:.4f}")

    mean_best_iou = float(np.mean(best_ious))
    logger.info(f"Mean Best IoU (GT-anchored): {mean_best_iou:.4f}  "
                f"(n_gt={len(gt_boxes)}, n_pred={len(pred_boxes)})")

    # --- Matched IoU (prediction-anchored, robust to n_pred) ---
    # Build all (pred, GT) IoU pairs, sort descending.
    # Greedily assign each prediction to its best unmatched GT (each GT matched once).
    # Average IoUs of matched pairs only — unmatched predictions are ignored,
    # so adding extra low-quality predictions does NOT lower this score.
    iou_pairs = []
    for pi, (px1, py1, px2, py2) in enumerate(pred_boxes):
        for gi, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            iou = calculate_iou_gt_img((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
            iou_pairs.append((iou, pi, gi))
    iou_pairs.sort(key=lambda x: -x[0])

    matched_pred = set()
    matched_gt   = set()
    matched_ious = []
    for iou, pi, gi in iou_pairs:
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
            matched_ious.append(iou)
            logger.info(f"  Matched pred[{pi}] GT[{gi}]  IoU={iou:.4f}")

    matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    logger.info(f"Matched IoU (pred-anchored): {matched_iou:.4f}  "
                f"(n_matched={len(matched_ious)}/{min(len(gt_boxes), len(pred_boxes))})")

    # --- Recall @ each IoU threshold ---
    recalls = {}
    for thr in iou_thresholds:
        n_detected = sum(1 for iou in best_ious if iou >= thr)
        recall = n_detected / len(gt_boxes)
        recalls[thr] = recall
        logger.info(f"Recall@{thr}: {recall:.4f}  ({n_detected}/{len(gt_boxes)} GT matched)")

    save_path = os.path.join(save_folder, "localization_accuracy.txt")
    with open(save_path, "w") as f:
        f.write(f"n_gt={len(gt_boxes)}  n_pred={len(pred_boxes)}\n\n")
        f.write(f"Mean Best IoU (GT-anchored):    {mean_best_iou:.4f}\n")
        f.write(f"Matched IoU   (pred-anchored):  {matched_iou:.4f}  "
                f"(n_matched={len(matched_ious)})\n")
        for thr, recall in recalls.items():
            n_detected = sum(1 for iou in best_ious if iou >= thr)
            f.write(f"Recall@{thr}: {recall:.4f}  ({n_detected}/{len(gt_boxes)} GT matched)\n")
        f.write("\nPer-GT best IoU:\n")
        for (gx1, gy1, gx2, gy2), iou in zip(gt_boxes, best_ious):
            f.write(f"  GT ({gx1},{gy1},{gx2},{gy2})  best_IoU={iou:.4f}\n")
    logger.info(f"Saved localization accuracy to {save_path}")
    return mean_best_iou, matched_iou, recalls


def main(exp, args):

    logger.remove()  # Remove default logger to avoid duplicate logs
    logger.add(sys.stderr, level="INFO")  # Add logger to print to console
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
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    os.makedirs(save_folder, exist_ok=True)
    result_save_path = save_folder
    logger.add(os.path.join(result_save_path, "run.log"), mode="w")
    if args.gframe != 0:
        #logger.info(f"Input frame: {args.input_frame_name}, reference frame: {args.gframe}")
        ref_frame_batch_set = read_ref_frame_list(args, args.gframe)
    elif args.lframe != 0:
        #logger.info(f"Input frame: {args.input_frame_name}, reference frame: {args.lframe}")
        ref_frame_batch_set = read_ref_frame_list(args, args.lframe)

    batch_set, batch_item = find_batch_set_and_item_for_input_frame(ref_frame_batch_set, args.input_frame_name)
    input_feat_info_item = read_input_feature_info_list(args, batch_set, batch_item, save_folder)



    visualize_feature_info_on_frame(args, "Input", [input_feat_info_item], save_folder, exp)

    if batch_set == 0:
        logger.info(f"Batch set for input frame is 0, which may not have memory features.")
    else:
        mem_feat_info = read_mem_feature_info_list(args, (batch_set-1),save_folder)

        visualize_mem_info_on_frame(args, "Memory", mem_feat_info, save_folder, exp, ref_frame_batch_set)
        sampled_mem_feat_info = read_sampled_mem_feature_info_list(args, (batch_set-1),save_folder)

        visualize_mem_info_on_frame(args, "Sampled_Memory", sampled_mem_feat_info, save_folder, exp, ref_frame_batch_set)

    updated_feat_info = read_updated_feature_info_list(args, batch_set, batch_item, save_folder)

    visualize_feature_info_on_frame(args, "Updated", [updated_feat_info], save_folder, exp)
    agg_feat_info = read_agg_feature_info_list(args, batch_set, batch_item, save_folder)

    visualize_feature_info_on_frame(args, "Agg", [agg_feat_info], save_folder, exp)

    result_info = read_result_info_list(args)
    if args.draw_result_feat_info:
        visualize_result_info_on_frame(args,  [result_info], save_folder)

    outputs_info = read_outputs_info_list(args, batch_set, batch_item, save_folder)


    visualize_outputs_info_on_frame(args,  outputs_info, save_folder, exp, batch_set, batch_item)

    calculate_mAP(args, outputs_info, save_folder, iou_threshold=0)
    calculate_AverageIou(args, outputs_info, save_folder)
    calculate_meanIoU(args, outputs_info, save_folder)
    calculate_localization_accuracy(args, outputs_info, save_folder, iou_thresholds=[0.5, 0.75])
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
