#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
from email import parser
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets.vid_classes import VID_classes
#from yolox.data.datasets.vid_classes import OVIS_classes as VID_classes
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from val_to_imdb import Predictor
from yolox.models.post_process import post_linking
import random
import json
import REPP
import numpy as np
import pickle
from collections import OrderedDict
from xml.dom import minidom

# synset → class index mapping (matches yolox/data/datasets/vid.py)
_VID_SYNSET_LIST = [
    'n02691156','n02419796','n02131653','n02834778','n01503061','n02924116',
    'n02958343','n02402425','n02084071','n02121808','n02503517','n02118333',
    'n02510455','n02342885','n02374451','n02129165','n01674464','n02484322',
    'n03790512','n02324045','n02509815','n02411705','n01726692','n02355227',
    'n02129604','n04468005','n01662784','n04530566','n02062744','n02391049',
]
_VID_SYNSET_TO_IDX = {s: i for i, s in enumerate(_VID_SYNSET_LIST)}

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png",".JPEG"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOV Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="/mnt/weka/scratch/yuheng.shi/dataset/VID/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00130000.mp4", help="path to images or video"
    )

    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='./exps/yolov++/v++_SwinBaseX_decoupleReg.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='./V++_outputs/v++_SwinBaseX_decoupleReg/best_ckpt.pth', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--dataset",
        default='vid',
        type = str,
        help = "Decide pred classes"
    )
    parser.add_argument("--conf", default=0.05, type=float, help="test conf")
    parser.add_argument("--nms", default=0.5, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--gframe', default=32, help='global frame num')
    parser.add_argument('--lframe', default=0, help='local frame num')
    parser.add_argument('--save_result', default=True)
    parser.add_argument('--post', default=False,action="store_true")
    parser.add_argument('--repp_cfg', default='./tools/yolo_repp_cfg.json' ,help='repp cfg filename', type=str)
    parser.add_argument("--format", default="video", type=str, help="input format files or video")
    parser.add_argument('--save_annotation', default=True)
    parser.add_argument('--save_features_info', default=True)
    parser.add_argument('--m_conf', default=0, type=float,help='select reference features minimum conf score')
    parser.add_argument('--reproduced_list', default="None", type=str, help="input image list")
    parser.add_argument('--save_result_with_gt', default=False)
    return parser


def get_image_list(path):
    image_names = []
    file_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
                file_names.append(filename)
    return file_names, image_names

def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name,[image_name])
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
def imagedir_demo(predictor, vis_folder, current_time, args,exp):
    gframe = exp.gframe_val
    lframe = exp.lframe_val
    traj_linking = exp.traj_linking
    logger.info(f"gframe: {gframe}, lframe: {lframe}, traj_linking: {traj_linking}")
    P, Cls = exp.defualt_p, exp.num_classes

    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )

    os.makedirs(save_folder, exist_ok=True)
    img_save_path = save_folder
    logger.add(os.path.join(img_save_path, "run.log"), mode="w")
    reproduced_step = True
    if args.reproduced_list == "None":
        if os.path.isdir(args.path):
            file_names, files = get_image_list(args.path)
        else:
            raise ValueError(f"{args.path} is invalid!")
        reproduced_step = False
    else:
        logger.info(f'reproduced_list: {args.reproduced_list}')
        with open(args.reproduced_list, 'r') as reproduced_list_file:
            files = [line.strip() for line in reproduced_list_file.readlines() if line.strip()]

    if gframe == 0:
        files, file_names = zip(*sorted(zip(files, file_names)))
        files, file_names = list(files), list(file_names)

    frames = []
    outputs = []
    ori_frames = []
    logger.info(f'file list len: {len(files)}')
    if reproduced_step == False:
        rep_file = open('./reproduced_list.txt', 'w')

    if gframe != 0 and reproduced_step == False:
        random.seed(41)
        random.shuffle(files)
    for file in files:
        #logger.info(f"read file {file}")
        if reproduced_step == False:
            rep_file.write(file+'\n')
        frame = cv2.imread(file)
        height, width = frame.shape[:2]
        ori_frames.append(frame)
        frame, _ = predictor.preproc(frame, None, exp.test_size)
        frames.append(torch.tensor(frame))
    if reproduced_step == False:
        rep_file.close()

    res = []
    frame_len = len(frames)
    img_path_list = []


    if gframe != 0:
        #random.seed(41)
        #random.shuffle(index_list)
        #random.seed(41)
        #random.shuffle(frames)
        split_num = int(frame_len / (gframe))
        for i in range(split_num):
            res.append(frames[i * gframe:(i + 1) * gframe])
            img_path_list.append(files[i*gframe:(i+1)*gframe])
        res.append(frames[split_num * gframe:])
        img_path_list.append(files[split_num * gframe:])

    else:
        split_num = int(frame_len / (lframe))
        for i in range(split_num):
            if traj_linking and i != 0:
                res.append(frames[i * lframe-1:(i + 1) * lframe])
            else:
                res.append(frames[i * lframe:(i + 1) * lframe])
        if traj_linking:
            tail = frames[split_num * lframe - 1:]
        else:
            tail = frames[split_num * lframe:]

        res.append(tail)
    ref_frame_list_file = open(os.path.join(img_save_path, "ref_frame_list_file_name.txt"), "w")
    for file in files:
        ref_frame_list_file.write(file+'\n')
    ref_frame_list_file.close()
    outputs, adj_lists, fc_outputs, names = [], [], [], []
    updated_feat_info_list = []
    mem_feat_info_list = []
    sampled_mem_feat_info_list = []
    input_feat_info_list = []
    outputs_info_list = []
    first_frame = True
    for ele_id,ele in enumerate(res):
        if ele == []: continue
        frame_num = len(ele)
        ele = torch.stack(ele)
        if traj_linking:
            pred_result, adj_list, fc_output = predictor.inference(ele, first_frame, lframe=frame_num, gframe=0,  img_path=img_path_list[ele_id])
            if first_frame:
                first_frame = False
            if len(outputs) != 0:  # skip the connection frame
                pred_result = pred_result[1:]
                fc_output = fc_output[1:]
            outputs.extend(pred_result)
            adj_lists.extend(adj_list)
            fc_outputs.append(fc_output)
        else:
            logger.info(f"ele_id: {ele_id}, ele.shape: {ele.shape}")
            outputs.extend(predictor.inference(ele, first_frame, lframe=lframe,gframe=gframe, img_path=img_path_list[ele_id]))
            if first_frame:
                first_frame = False
        head = predictor.model.head
        original_ref_pred_info = head._original_ref_pred_info  # Get the original reference prediction info from the head
        logger.info(f"After inference of set {ele_id}, got original_ref_pred_info length: {len(original_ref_pred_info)}")
        original_ref_pred_info_list = [[[t.cpu().numpy() for t in pX] for pX in batch_item] for batch_item in original_ref_pred_info]
        input_feat_info_list.append(original_ref_pred_info_list)
        ref_pred_info = head._ref_pred_info  # Get the reference prediction info from the head
        logger.info(f"After inference of set {ele_id}, got ref_pred_info with length: {len(ref_pred_info)}")
        updated_feat_info_list.append([[[t.cpu().numpy() for t in pX] for pX in batch_item] for batch_item in ref_pred_info])
        outputs_info = head._outputs_info  # Get the outputs info from the head
        logger.info(f"After inference of set {ele_id}, got outputs_info with length: {len(outputs_info)}")
        def _to_np(v):
            return v.cpu().numpy() if isinstance(v, torch.Tensor) else v
        converted_outputs_info = []
        for det_list in outputs_info:
            if det_list is None:
                converted_outputs_info.append([])
                continue
            item_result = []
            for det in det_list:
                bboxes = _to_np(det['bboxes'])
                item_result.append([
                    det['batch_set'], det['batch_item'],
                    bboxes[0], bboxes[1], bboxes[2], bboxes[3],
                    _to_np(det['obj_score']), _to_np(det['cls_score']),
                    _to_np(det['label']), det['feat_id'] if det['feat_id'] is not None else -1
                ])
            converted_outputs_info.append(item_result)
        outputs_info_list.append(converted_outputs_info)
        #for level_idx, level_info in enumerate(ref_pred_info):
        #    level_info_p3, level_info_p4, level_info_p5 = level_info
        #    logger.info(f"Level {level_idx} - len(level_info_p3): {len(level_info_p3)}, \
        #                len(level_info_p4): {len(level_info_p4)}, \
        #                    len(level_info_p5): {len(level_info_p5)}")
        #    for idx, items in enumerate(level_info_p3):
        #        logger.info(f"Set {ele_id} - Level {level_idx} - p3 item {idx} th - p3 item shape: {items.shape}")
        #        logger.info(f"Set {ele_id} - Level {level_idx} - p3 item {idx} th - p3 items: {items}")
        #    for idx, items in enumerate(level_info_p4):
        #        logger.info(f"Set {ele_id} - Level {level_idx} - p4 item {idx} th - p4 item shape: {items.shape}")
        #        logger.info(f"Set {ele_id} - Level {level_idx} - p4 item {idx} th - p4 items: {items}")
        #    for idx, items in enumerate(level_info_p5):
        #        logger.info(f"Set {ele_id} - Level {level_idx} - p5 item {idx} th - p5 item shape: {items.shape}")
        #        logger.info(f"Set {ele_id} - Level {level_idx} - p5 item {idx} th - p5 items: {items}")
        mem_feat_info = head._mem_info  # Get the memory feature info from the head
        logger.info(f"After inference of set {ele_id}, got mem_feat_info with keys: {mem_feat_info.keys()}")
        p3_mem_info = mem_feat_info['p3']
        logger.info(f"Set {ele_id} - mem_feat_info - len(p3_mem_info): {len(p3_mem_info)}")
        p4_mem_info = mem_feat_info['p4']
        logger.info(f"Set {ele_id} - mem_feat_info - len(p4_mem_info): {len(p4_mem_info)}")
        p5_mem_info = mem_feat_info['p5']
        logger.info(f"Set {ele_id} - mem_feat_info - len(p5_mem_info): {len(p5_mem_info)}")
        mem_feat_info_list.append({k: list(v) for k, v in mem_feat_info.items()})
        sampled_mem_feat_info = head._sampled_mem_feat_info  # Get the sampled memory feature info from the head
        logger.info(f"After inference of set {ele_id}, got sampled_mem_feat_info with keys: {sampled_mem_feat_info.keys()}")
        if sampled_mem_feat_info:
            p3_sampled_mem_info = sampled_mem_feat_info['p3']
            logger.info(f"Set {ele_id} - sampled_mem_feat_info - len(p3_sampled_mem_info): {len(p3_sampled_mem_info)}")
            p4_sampled_mem_info = sampled_mem_feat_info['p4']
            logger.info(f"Set {ele_id} - sampled_mem_feat_info - len(p4_sampled_mem_info): {len(p4_sampled_mem_info)}")
            p5_sampled_mem_info = sampled_mem_feat_info['p5']
            logger.info(f"Set {ele_id} - sampled_mem_feat_info - len(p5_sampled_mem_info): {len(p5_sampled_mem_info)}")
            sampled_mem_feat_info_list.append({k: list(v) for k, v in sampled_mem_feat_info.items()})
    if traj_linking:
        outputs = post_linking(fc_outputs, adj_lists, outputs, P, Cls, names, exp)

  #  outputs = [j for _,j in sorted(zip(index_list,outputs))]
    ratio = min(predictor.test_size[0] / height, predictor.test_size[1] / width)
    if args.post:
        logger.info("Post processing...")
        out_post_format = predictor.convert_to_post(outputs, ratio, [height, width])
        out_post = predictor.post(out_post_format)
        outputs = predictor.convert_to_ori(out_post, frame_len)

    logger.info("Saving detection image result in {}".format(img_save_path))
    img_anno_res = {}
    for (output,img, file_path) in zip(outputs,ori_frames[:len(outputs)],files):
        logger.info(f"Processing file {file_path}")
        if args.post:
            ratio = 1
        result_frame = predictor.visual(output,img,ratio,cls_conf=args.conf,color_idx=12)
        bboxes = output[:, 0:4]
        cls = output[:, 6].unsqueeze(1)
        scores = (output[:, 4] * output[:, 5]).unsqueeze(1)
        bboxes_cls = torch.cat([bboxes, scores, cls], dim=1)
        bboxes_cls_cpu = bboxes_cls.cpu().numpy()
        file_name = os.path.basename(file_path)
        img_anno_res[file_name] = bboxes_cls_cpu
        if args.save_result:
            cv2.imwrite(os.path.join(img_save_path, file_name), result_frame)
        if args.save_result_with_gt:
            file_name_with_gt = 'gt_result_' + file_name
            xml_path = file_path.replace("Data", "Annotations").replace(".JPEG", ".xml").replace(".jpeg", ".xml").replace(".jpg", ".xml")
            result_with_gt_frame = result_frame.copy()
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
                    cv2.rectangle(result_with_gt_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(result_with_gt_frame, 'GT:' + label, (xmin, max(ymin - 4, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                logger.warning("GT xml not found: {}".format(xml_path))
            cv2.imwrite(os.path.join(img_save_path, file_name_with_gt), result_with_gt_frame)
    if args.save_annotation:
        anno_save_path = os.path.join(img_save_path, "my_model_result_info.pkl")
        with open(anno_save_path, "wb") as f:
            pickle.dump(img_anno_res, f)
        logger.info("Saving detection prediction result in {}".format(anno_save_path))
    if args.save_features_info:
        input_feat_info_save_path = os.path.join(img_save_path, "my_model_input_feat_info.pkl")
        with open(input_feat_info_save_path, "wb") as f:
            pickle.dump(input_feat_info_list, f)
        logger.info("Saving detection prediction input feature info in {}".format(input_feat_info_save_path))
        updated_feat_info_save_path = os.path.join(img_save_path, "my_model_updated_feat_info.pkl")
        with open(updated_feat_info_save_path, "wb") as f:
            pickle.dump(updated_feat_info_list, f)
        logger.info("Saving detection prediction feature info in {}".format(updated_feat_info_save_path))
        ref_frame_save_path = os.path.join(img_save_path, "my_model_ref_frame_list.npy")
        np.save(ref_frame_save_path, np.array(files, dtype=object))
        logger.info("Saving detection prediction reference frame info in {}".format(ref_frame_save_path))
        mem_feat_info_save_path = os.path.join(img_save_path, "my_model_mem_feat_info.pkl")
        with open(mem_feat_info_save_path, "wb") as f:
            pickle.dump(mem_feat_info_list, f)
        logger.info("Saving detection prediction memory feature info in {}".format(mem_feat_info_save_path))
        sampled_mem_feat_info_save_path = os.path.join(img_save_path, "my_model_sampled_mem_feat_info.pkl")
        with open(sampled_mem_feat_info_save_path, "wb") as f:
            pickle.dump(sampled_mem_feat_info_list, f)
        logger.info("Saving detection prediction sampled memory feature info in {}".format(sampled_mem_feat_info_save_path))
        outputs_info_save_path = os.path.join(img_save_path, "my_model_outputs_info.pkl")
        with open(outputs_info_save_path, "wb") as f:
            pickle.dump(outputs_info_list, f)
        logger.info("Saving detection prediction outputs info in {}".format(outputs_info_save_path))

def imageflow_demo(predictor, vis_folder, current_time, args,exp):
    gframe = exp.gframe_val
    lframe = exp.lframe_val
    traj_linking = exp.traj_linking
    P, Cls = exp.defualt_p, exp.num_classes

    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )

    os.makedirs(save_folder, exist_ok=True)
    ratio = min(predictor.test_size[0] / height, predictor.test_size[1] / width)
    vid_save_path = os.path.join(save_folder, args.path.split("/")[-1])
    img_save_path = save_folder
    logger.info(f"video save_path is {vid_save_path}")
    vid_writer = cv2.VideoWriter(
        vid_save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    frames = []
    outputs = []
    ori_frames = []
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            ori_frames.append(frame)
            frame, _ = predictor.preproc(frame, None, exp.test_size)
            frames.append(torch.tensor(frame))
        else:
            break
    res = []
    frame_len = len(frames)
    index_list = list(range(frame_len))
    if gframe != 0:
        random.seed(41)
        random.shuffle(index_list)
        random.seed(41)
        random.shuffle(frames)
        split_num = int(frame_len / (gframe))#
        for i in range(split_num):
            res.append(frames[i * gframe:(i + 1) * gframe])
        res.append(frames[(i + 1) * gframe:])
    else:
        split_num = int(frame_len / (lframe))
        for i in range(split_num):
            if traj_linking and i != 0:
                res.append(frames[i * lframe-1:(i + 1) * lframe])
            else:
                res.append(frames[i * lframe:(i + 1) * lframe])
        if traj_linking:
            tail = frames[split_num * lframe - 1:]
        else:
            tail = frames[split_num * lframe:]
        res.append(tail)

    outputs, adj_lists, fc_outputs, names = [], [], [], []
    first_frame = True
    for ele_id,ele in enumerate(res):
        if ele == []: continue
        frame_num = len(ele)
        ele = torch.stack(ele)
        t0 = time.time()
        if traj_linking:
            pred_result, adj_list, fc_output = predictor.inference(ele, first_frame, lframe=frame_num, gframe=0, img_path=None)
            if first_frame:
                first_frame = False

            if len(outputs) != 0:  # skip the connection frame
                pred_result = pred_result[1:]
                fc_output = fc_output[1:]
            outputs.extend(pred_result)
            adj_lists.extend(adj_list)
            fc_outputs.append(fc_output)
        else:
            outputs.extend(predictor.inference(ele, first_frame, lframe=lframe,gframe=gframe, img_path=None))
            if first_frame:
                first_frame = False
    if traj_linking:
        outputs = post_linking(fc_outputs, adj_lists, outputs, P, Cls, names, exp)

    outputs = [j for _,j in sorted(zip(index_list,outputs))]
    if args.post:
        logger.info("Post processing...")
        out_post_format = predictor.convert_to_post(outputs, ratio, [height, width])
        out_post = predictor.post(out_post_format)
        outputs = predictor.convert_to_ori(out_post, frame_len)

    logger.info("Saving detection result in {}".format(img_save_path))
    for img_idx,(output,img) in enumerate(zip(outputs,ori_frames[:len(outputs)])):
        if args.post:
            ratio = 1
        result_frame = predictor.visual(output,img,ratio,cls_conf=args.conf)
        if args.save_result:
            vid_writer.write(result_frame)
            cv2.imwrite(os.path.join(img_save_path, str(img_idx) + '.jpg'), result_frame)

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(args.output_dir,file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    if args.dataset=='vid':
        repp_params = json.load(open(args.repp_cfg, 'r'))
        post = REPP.REPP(**repp_params)
        predictor = Predictor(model, exp, VID_classes, trt_file, decoder, args.device, args.legacy,post=post)
    else:
        predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device, args.legacy)
    current_time = time.localtime()

    if args.format == "video":
        imageflow_demo(predictor, vis_folder, current_time, args,exp)
    elif args.format == "files":
        imagedir_demo(predictor, vis_folder,  current_time, args,exp)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.traj_linking = True and exp.lmode
    exp.lframe_val = int(args.lframe)
    exp.gframe_val = int(args.gframe)
    exp.m_conf = args.m_conf
    main(exp, args)
