#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
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
    parser.add_argument('--draw_input_feat_info', default=True)
    parser.add_argument('--draw_mem_feat_info', default=True)
    parser.add_argument('--draw_sampled_mem_feat_info', default=True)
    parser.add_argument('--draw_updated_feat_info', default=True)
    parser.add_argument('--draw_result_info', default=True)
    parser.add_argument('--gframe', default=0, help='global frame num')
    parser.add_argument('--lframe', default=16, help='local frame num')
    parser.add_argument('--test_conf', type=float, default=0.01, help='test confidence threshold')
    parser.add_argument('--tsize', default=640, type=int, help='test image size')
    parser.add_argument('--input_frame_name', default='000000.JPEG', help='input frame name to check')
    parser.add_argument('--input_image_path', default='/home/kssong/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00118007', help='input frame name to check')
    parser.add_argument('--save_result_with_gt', default=False)
    return parser

def read_ref_frame_list(args, n_frames):
    ref_frame_list_path = os.path.join(args.input_dir, "my_model_ref_frame_list.npy")
    ref_frame_list = np.load(ref_frame_list_path, allow_pickle=True)
    ref_frame_batch_set = []
    ref_frame_batch = []
    #logger.info(f"len(ref_frame_list): {len(ref_frame_list)}")
    #logger.info(f"n_frames: {n_frames}")
    for i, ref_frame in enumerate(ref_frame_list):
        ref_frame_batch.append(ref_frame)
        if (i + 1) % n_frames == 0:
            #logger.info(f"Ref frame {i}: {ref_frame}")
            ref_frame_batch_set.append(ref_frame_batch)
            ref_frame_batch = []
    return ref_frame_batch_set

def read_input_feature_info_list(args, check_batch_set, check_batch_item):
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
                                conf_score: {conf_score:.3f}, \
                                class_label: {class_label}, \
                                class_label_name: {class_label_name}\
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
                                conf_score: {conf_score:.3f}, \
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
                                conf_score: {conf_score:.3f}, \
                                class_label: {class_label}, \
                                class_label_name: {class_label_name}, \
                                feat_num: {feat_num}")
            if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                    break
        if batch_set == check_batch_set_num:
            break
    return feat_info_item

def read_outputs_info_list(args, check_batch_set, check_batch_item):
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
                                 conf_score: {conf_score:.3f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}, \
                                 feat_num: {feat_num}")
                return selected_outputs_info_list

    return selected_outputs_info_list

def read_mem_feature_info_list(args, check_batch_set):
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
                            conf_score: {conf_score:.3f}, \
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
                            conf_score: {conf_score:.3f}, \
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
                            conf_score: {conf_score:.3f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name}, \
                            feat_num: {feat_num}")
             break

    return mem_feat_info

def read_sampled_mem_feature_info_list(args, check_batch_set):
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
                            conf_score: {conf_score:.3f}, \
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
                            conf_score: {conf_score:.3f}, \
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
                            conf_score: {conf_score:.3f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name}, \
                            feat_num: {feat_num}")
             break
    return sampled_mem_feat_info
def read_updated_feature_info_list(args,check_batch_set, check_batch_item):
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
                                 conf_score: {conf_score:.3f}, \
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
                                 conf_score: {conf_score:.3f}, \
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
                                 conf_score: {conf_score:.3f}, \
                                 class_label: {class_label}, \
                                 class_label_name: {class_label_name}, \
                                 feat_num: {feat_num}")
                if batch_set == check_batch_set_num and batch_item == check_batch_item_num:
                    break
        if batch_set == check_batch_set_num:
            break
    return updated_feat_info_item

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


    for i, p3 in enumerate(p3_mem_info):
         if torch.all(p3[2] == 0):
             continue
         input_frame_path = ref_frame_batch_set[p3[0]][p3[1]]
         input_frame_name = os.path.basename(input_frame_path)
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
         logger.info(f"    p3 -{i}th batch_set: {p3[0]}, batch_item: {p3[1]}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                    obj_score: {obj_score:.3f}, \
                    cls_score: {cls_score:.3f}, \
                    conf_score: {conf_score:.3f}, \
                    class_label: {class_label}, \
                    class_label_name: {class_label_name}, \
                    feat_num: {feat_num}")
         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
         cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         file_name = input_frame_name + f'_{type_name}_P3_{i}.JPEG'
         cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
         logger.info(f"Saved visualized frame with p3 feature info: {os.path.join(frame_save_path, file_name)}")
         logger.info(f"Finished visualizing p3 feature info for {input_frame_path}")

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
         logger.info(f"    p4 -{i}th batch_set: {p4[0]}, batch_item: {p4[1]}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                    obj_score: {obj_score:.3f}, \
                    cls_score: {cls_score:.3f}, \
                    conf_score: {conf_score:.3f}, \
                    class_label: {class_label}, \
                    class_label_name: {class_label_name}, \
                    feat_num: {feat_num}")
         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
         cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         file_name = input_frame_name + f'_{type_name}_P4_{i}.JPEG'
         cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
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
         logger.info(f"    p5 -{i}th batch_set: {p5[0]}, batch_item: {p5[1]}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                    obj_score: {obj_score:.3f}, \
                    cls_score: {cls_score:.3f}, \
                    conf_score: {conf_score:.3f}, \
                    class_label: {class_label}, \
                    class_label_name: {class_label_name}, \
                    feat_num: {feat_num}")
         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
         cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         file_name = input_frame_name + f'_{type_name}_P5_{i}.JPEG'
         cv2.imwrite(os.path.join(frame_save_path, file_name), frame)


def visualize_feature_info_on_frame(args, type_name, feat_info_list, frame_save_path,exp=None):
    input_frame_path = os.path.join(args.input_image_path, args.input_frame_name)
    logger.info(f"Visualizing feature info on frame: {input_frame_path}")
    _frame = cv2.imread(input_frame_path)
    height, width = _frame.shape[:2]
    if exp is not None:
        ratio = min(exp.test_size[0] / height, exp.test_size[1] / width)
    else:
        ratio = 1.0

    for feat_info in feat_info_list:
            p3_list, p4_list, p5_list = feat_info
            for i, p3 in enumerate(p3_list):
                 if np.all(p3 == 0):
                     continue
                 frame = cv2.imread(input_frame_path)
                 bbox = p3[:4] / ratio
                 obj_score = p3[4]
                 cls_score = p3[5]
                 class_pred = p3[6: 6+len(VID_classes)]
                 class_label = int(np.argmax(class_pred))
                 feat_num = int(p3[6 + len(VID_classes)])
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 label_text = f"{class_label_name} {conf_score:.3f}"
                 logger.info(f"    p3 -{i}th  bbox: ({int(p3[0])}, {int(p3[1])}, {int(p3[2])}, {int(p3[3])}), \
                            obj_score: {obj_score:.3f}, \
                            cls_score: {cls_score:.3f}, \
                            conf_score: {conf_score:.3f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name},\
                            feat_num: {feat_num}")
                 cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                 cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                 file_name = args.input_frame_name + f'_{type_name}_P3_{i}.JPEG'
                 cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
            for i, p4 in enumerate(p4_list):
                 if np.all(p4 == 0):
                     continue
                 frame = cv2.imread(input_frame_path)
                 bbox = p4[:4] / ratio
                 obj_score = p4[4]
                 cls_score = p4[5]
                 class_pred = p4[6: 6 + len(VID_classes)]
                 class_label = int(np.argmax(class_pred))
                 feat_num = int(p4[6 + len(VID_classes)])
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 label_text = f"{class_label_name} {conf_score:.31f}"
                 cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                 cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                 file_name = args.input_frame_name + f'_{type_name}_P4_{i}.JPEG'
                 cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
                 logger.info(f"    p4 -{i}th  bbox: ({int(p4[0])}, {int(p4[1])}, {int(p4[2])}, {int(p4[3])}), \
                            obj_score: {obj_score:.3f}, \
                            cls_score: {cls_score:.3f}, \
                            conf_score: {conf_score:.3f}, \
                            class_label: {class_label}, \
                            class_label_name: {class_label_name},\
                            feat_num: {feat_num}")
            for i, p5 in enumerate(p5_list):
                 if np.all(p5 == 0):
                     continue
                 frame = cv2.imread(input_frame_path)
                 bbox = p5[:4] / ratio
                 obj_score = p5[4]
                 cls_score = p5[5]
                 class_pred = p5[6: 6+len(VID_classes)]
                 class_label = int(np.argmax(class_pred))
                 feat_num = int(p5[6 + len(VID_classes)])
                 conf_score = cls_score * obj_score
                 class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
                 label_text = f"{class_label_name} {conf_score:.3f}"
                 cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                 cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                 file_name = args.input_frame_name + f'_{type_name}_P5_{i}.JPEG'
                 cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
                 logger.info(f"    p5 -{i}th bbox: ({int(p5[0])}, {int(p5[1])}, {int(p5[2])}, {int(p5[3])}), \
                                obj_score: {obj_score:.3f}, \
                                cls_score: {cls_score:.3f}, \
                                conf_score: {conf_score:.3f}, \
                                class_label: {class_label}, \
                                class_label_name: {class_label_name},\
                                feat_num: {feat_num}")

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
    for i, output_info in enumerate(outputs_info):
        frame = cv2.imread(input_frame_path)
        logger.info(f"    output_info -{i}th - len: {len(output_info)}")
        logger.info(f"    output_info -{i}th - content: {output_info}")
        batch_set_output = output_info[0]
        batch_item_output = output_info[1]
        if batch_set_output != batch_set or batch_item_output != batch_item:
            logger.info(f"Skipping output info -{i}th because batch_set_output: {batch_set_output}, batch_item_output: {batch_item_output} do not match input frame's batch_set: {batch_set}, batch_item: {batch_item}")
            continue
        bbox = output_info[2:6]
        obj_score = float(output_info[6])
        cls_score = float(output_info[7])
        class_label = int(output_info[8])
        feat_num = int(output_info[9]) if output_info[9] is not None else -1
        conf_score = cls_score * obj_score
        class_label_name = VID_classes[class_label] if class_label < len(VID_classes) else "Unknown"
        label_text = f"{class_label_name} {conf_score:.3f}"
        height, width = frame.shape[:2]
        if exp is not None:
            ratio = min(exp.test_size[0] / height, exp.test_size[1] / width)
        else:
            ratio = 1.0
        bbox = [v / ratio for v in bbox]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        file_name = args.input_frame_name + f'_Output_Info_{i}.JPEG'
        cv2.imwrite(os.path.join(frame_save_path, file_name), frame)
        logger.info(f"    output_info -{i}th - batch_set: {batch_set_output}, batch_item: {batch_item_output}, bbox: ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}), \
                    obj_score: {obj_score:.3f}, \
                    cls_score: {cls_score:.3f}, \
                    conf_score: {conf_score:.3f}, \
                    class_label: {class_label}, \
                    class_label_name: {class_label_name}, \
                    feat_num: {feat_num}")


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
    input_feat_info_item = read_input_feature_info_list(args, batch_set, batch_item)

    visualize_feature_info_on_frame(args, "Input", [input_feat_info_item], save_folder, exp)

    if batch_set == 0:
        logger.info(f"Batch set for input frame is 0, which may not have memory features.")
    else:
        mem_feat_info = read_mem_feature_info_list(args, (batch_set-1))
        visualize_mem_info_on_frame(args, "Memory", mem_feat_info, save_folder, exp, ref_frame_batch_set)
        sampled_mem_feat_info = read_sampled_mem_feature_info_list(args, (batch_set-1))
        visualize_mem_info_on_frame(args, "Sampled_Memory", sampled_mem_feat_info, save_folder, exp, ref_frame_batch_set)

    updated_feat_info = read_updated_feature_info_list(args, batch_set, batch_item)
    visualize_feature_info_on_frame(args, "Updated", [updated_feat_info], save_folder, exp)

    result_info = read_result_info_list(args)
    visualize_result_info_on_frame(args,  [result_info], save_folder)

    outputs_info = read_outputs_info_list(args, batch_set, batch_item)
    visualize_outputs_info_on_frame(args,  outputs_info, save_folder, exp, batch_set, batch_item)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
