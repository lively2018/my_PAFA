#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import copy
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tqdm import tqdm
from yolox.evaluators.coco_evaluator import per_class_AR_table, per_class_AP_table
import torch
import pycocotools.coco
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

vid_classes = (
    'airplane', 'antelope', 'bear', 'bicycle',
    'bird', 'bus', 'car', 'cattle',
    'dog', 'domestic_cat', 'elephant', 'fox',
    'giant_panda', 'hamster', 'horse', 'lion',
    'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake', 'squirrel',
    'tiger', 'train', 'turtle', 'watercraft',
    'whale', 'zebra'
)


# from yolox.data.datasets.vid_classes import Arg_classes as  vid_classes

class VIDEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
            self, dataloader, img_size, confthre, nmsthre,
            num_classes, testdev=False, gl_mode=False,
            lframe=0, gframe=32,**kwargs
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.id = 0
        self.box_id = 0
        self.id_ori = 0
        self.box_id_ori = 0
        self.gl_mode = gl_mode
        self.lframe = lframe
        self.gframe = gframe
        self.kwargs = kwargs
        self.vid_to_coco = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': [{"supercategorie": "", "id": 0, "name": "airplane"},
                           {"supercategorie": "", "id": 1, "name": "antelope"},
                           {"supercategorie": "", "id": 2, "name": "bear"},
                           {"supercategorie": "", "id": 3, "name": "bicycle"},
                           {"supercategorie": "", "id": 4, "name": "bird"},
                           {"supercategorie": "", "id": 5, "name": "bus"},
                           {"supercategorie": "", "id": 6, "name": "car"},
                           {"supercategorie": "", "id": 7, "name": "cattle"},
                           {"supercategorie": "", "id": 8, "name": "dog"},
                           {"supercategorie": "", "id": 9, "name": "domestic_cat"},
                           {"supercategorie": "", "id": 10, "name": "elephant"},
                           {"supercategorie": "", "id": 11, "name": "fox"},
                           {"supercategorie": "", "id": 12, "name": "giant_panda"},
                           {"supercategorie": "", "id": 13, "name": "hamster"},
                           {"supercategorie": "", "id": 14, "name": "horse"},
                           {"supercategorie": "", "id": 15, "name": "lion"},
                           {"supercategorie": "", "id": 16, "name": "lizard"},
                           {"supercategorie": "", "id": 17, "name": "monkey"},
                           {"supercategorie": "", "id": 18, "name": "motorcycle"},
                           {"supercategorie": "", "id": 19, "name": "rabbit"},
                           {"supercategorie": "", "id": 20, "name": "red_panda"},
                           {"supercategorie": "", "id": 21, "name": "sheep"},
                           {"supercategorie": "", "id": 22, "name": "snake"},
                           {"supercategorie": "", "id": 23, "name": "squirrel"},
                           {"supercategorie": "", "id": 24, "name": "tiger"},
                           {"supercategorie": "", "id": 25, "name": "train"},
                           {"supercategorie": "", "id": 26, "name": "turtle"},
                           {"supercategorie": "", "id": 27, "name": "watercraft"},
                           {"supercategorie": "", "id": 28, "name": "whale"},
                           {"supercategorie": "", "id": 29, "name": "zebra"}],
            'images': [],
            'licenses': []
        }
        self.vid_to_coco_ori = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': [{"supercategorie": "", "id": 0, "name": "airplane"},
                           {"supercategorie": "", "id": 1, "name": "antelope"},
                           {"supercategorie": "", "id": 2, "name": "bear"},
                           {"supercategorie": "", "id": 3, "name": "bicycle"},
                           {"supercategorie": "", "id": 4, "name": "bird"},
                           {"supercategorie": "", "id": 5, "name": "bus"},
                           {"supercategorie": "", "id": 6, "name": "car"},
                           {"supercategorie": "", "id": 7, "name": "cattle"},
                           {"supercategorie": "", "id": 8, "name": "dog"},
                           {"supercategorie": "", "id": 9, "name": "domestic_cat"},
                           {"supercategorie": "", "id": 10, "name": "elephant"},
                           {"supercategorie": "", "id": 11, "name": "fox"},
                           {"supercategorie": "", "id": 12, "name": "giant_panda"},
                           {"supercategorie": "", "id": 13, "name": "hamster"},
                           {"supercategorie": "", "id": 14, "name": "horse"},
                           {"supercategorie": "", "id": 15, "name": "lion"},
                           {"supercategorie": "", "id": 16, "name": "lizard"},
                           {"supercategorie": "", "id": 17, "name": "monkey"},
                           {"supercategorie": "", "id": 18, "name": "motorcycle"},
                           {"supercategorie": "", "id": 19, "name": "rabbit"},
                           {"supercategorie": "", "id": 20, "name": "red_panda"},
                           {"supercategorie": "", "id": 21, "name": "sheep"},
                           {"supercategorie": "", "id": 22, "name": "snake"},
                           {"supercategorie": "", "id": 23, "name": "squirrel"},
                           {"supercategorie": "", "id": 24, "name": "tiger"},
                           {"supercategorie": "", "id": 25, "name": "train"},
                           {"supercategorie": "", "id": 26, "name": "turtle"},
                           {"supercategorie": "", "id": 27, "name": "watercraft"},
                           {"supercategorie": "", "id": 28, "name": "whale"},
                           {"supercategorie": "", "id": 29, "name": "zebra"}],
            'images': [],
            'licenses': []
        }
        self.testdev = testdev
        self.tmp_name_ori = './ori_pred.json'
        self.tmp_name_refined = './refined_pred.json'
        self.gt_ori = './gt_ori.json'
        self.gt_refined = './gt_refined.json'
    def calculate_mean_iou(self, predictions, ground_truths):
        """
        Calcuate the mean IoU for all matched prediction/GP pairs.

        :param self: Description
        :param predictions: Description
        :param ground_truths: Description
        """
        if not predictions or not ground_truths:
            return 0.0

        def compute_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
            yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
            #logger.info(f"xA: {xA} yA: {yA} xB: {xB} yB: {yB}")
            interArea = max(0, xB-xA) * max(0, yB - yA)
            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]
            #logger.info(f"interArea: {interArea}, boxAArea: {boxAArea}, boxBArea: {boxBArea}")
            iou = interArea/float(boxAArea + boxBArea - interArea + 1e-6)
            #logger.info(f"iou: {iou}")
            return iou


        gt_dict = {}
        for gt in ground_truths:
            img_id = gt['image_id']
            #logger.info(f"img_id: {img_id}")
            if img_id not in gt_dict:
                gt_dict[img_id] = []
            gt_dict[img_id].append(gt)

        total_iou = 0.0
        match_count = 0
        for pred in predictions:
            img_id = pred['image_id']
            #logger.info(f"img_id: {img_id}")
            if img_id in gt_dict:
                best_iou = 0
                #logger.info(f"best_iou: {best_iou}")
                for gt in gt_dict[img_id]:
                    if pred['category_id'] == gt['category_id']:
                        #logger.info(f"pred['category_id']: {pred['category_id']}")
                        #logger.info(f"gt['category_id']: {gt['category_id']}")
                        #logger.info(f"pred['bbox']: {pred['bbox']}")
                        #logger.info(f"gt['bbox']: {gt['bbox']}")
                        iou = compute_iou(pred['bbox'], gt['bbox'])
                        #logger.info(f"iou: {iou}, best_iou: {best_iou}")
                        best_iou = max(best_iou, iou)
                if best_iou > 0:
                    total_iou += best_iou
                    match_count += 1
                    #logger.info(f"total_iou: {total_iou}")
                    #logger.info(f"mathc_iou: {match_count}")
        return total_iou / match_count if match_count > 0  else 0.0
    def calculate_all_prediction_iou(self, predictions, ground_truths):
        """
        Calcuate the mean IoU for all matched prediction/GP pairs.

        :param self: Description
        :param predictions: Description
        :param ground_truths: Description
        """
        if not predictions or not ground_truths:
            return 0.0

        def compute_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
            yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
            #logger.info(f"xA: {xA} yA: {yA} xB: {xB} yB: {yB}")
            interArea = max(0, xB-xA) * max(0, yB - yA)
            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]
            #logger.info(f"interArea: {interArea}, boxAArea: {boxAArea}, boxBArea: {boxBArea}")
            iou = interArea/float(boxAArea + boxBArea - interArea + 1e-6)
            #logger.info(f"iou: {iou}")
            return iou


        gt_dict = {}
        gt_hit_status = {}
        for i, gt in enumerate(ground_truths):
            img_id = gt['image_id']
            #logger.info(f"img_id: {img_id}")
            if img_id not in gt_dict:
                gt_dict[img_id] = []
            gt_dict[img_id].append((i, gt))
            gt_hit_status[i] = False

        total_all_iou = 0.0
        all_pair_count = 0
        total_correct_iou = 0.0
        correct_pair_count = 0
        total_incorrect_iou = 0.0
        incorrect_pair_count = 0
        high_score_total_iou = 0.0
        high_score_pair_count = 0
        background_fp_count = 0
        for pred in predictions:
            img_id = pred['image_id']
            has_any_overlap = False
            #logger.info(f"img_id: {img_id}")
            if img_id in gt_dict:
                for gt_idx, gt in gt_dict[img_id]:
                    #logger.info(f"pred['category_id']: {pred['category_id']}")
                    #logger.info(f"gt['category_id']: {gt['category_id']}")
                    #logger.info(f"pred['bbox']: {pred['bbox']}")
                    #logger.info(f"gt['bbox']: {gt['bbox']}")

                    all_iou = compute_iou(pred['bbox'], gt['bbox'])
                    if all_iou > 0:
                        has_any_overlap = True
                        gt_hit_status[gt_idx] = True
                        total_all_iou += all_iou
                        all_pair_count += 1
                        if pred.get('score',0) > 0.25:
                            high_score_total_iou += all_iou
                            high_score_pair_count += 1

                    if pred['category_id'] == gt['category_id']:
                        correct_iou = all_iou
                        #logger.info(f"iou: {iou}, best_iou: {best_iou}")
                        if correct_iou > 0:
                            total_correct_iou += correct_iou
                            correct_pair_count += 1
                    else:
                        incorrect_iou = all_iou
                        if incorrect_iou > 0:
                            total_incorrect_iou += incorrect_iou
                            incorrect_pair_count += 1
            if not has_any_overlap:
                background_fp_count += 1
        false_negative_count = 0
        for hit in gt_hit_status.values():
            if not hit:
                false_negative_count +=1

        logger.info(f"total_all_iou: {total_all_iou:.4f}")
        logger.info(f"all_pair_count: {all_pair_count}")
        if all_pair_count > 0:
            avg_total_all_iou = total_all_iou/all_pair_count
        else:
            avg_total_all_iou = 0.0
        logger.info(f"avg_total_all_iou: {avg_total_all_iou:.4f}")
        logger.info(f"total_all_iou: {total_correct_iou:.4f}")
        logger.info(f"correct_pair_count(The same objects, True Positive): {correct_pair_count}")
        if correct_pair_count > 0:
            avg_total_correct_iou = total_correct_iou/correct_pair_count
        else:
            avg_total_correct_iou = 0.0
        logger.info(f"avg_total_correct_iou: {avg_total_correct_iou:.4f}")
        logger.info(f"total_incorrect_iou: {total_incorrect_iou:.4f}")
        logger.info(f"incorrect_pair_count(Other objects, False Positive): {incorrect_pair_count}")
        if incorrect_pair_count > 0:
            avg_total_incorrect_iou = total_incorrect_iou/incorrect_pair_count
        else:
            avg_total_incorrect_iou = 0.0
        logger.info(f"avg_total_incorrect_iou: {avg_total_incorrect_iou:.4f}")
        logger.info(f"high_score_total_iou(0.25): {high_score_total_iou:.4f}")
        logger.info(f"high_score_pair_count(0.25): {high_score_pair_count}")
        if high_score_pair_count > 0:
            avg_total_high_score_iou = high_score_total_iou/high_score_pair_count
        else:
            avg_total_high_score_iou = 0.0
        logger.info(f"avg_total_incorrect_iou: {avg_total_high_score_iou:.4f}")

        logger.info(f"background_fp_count(Ghost Detections, False Positive): {background_fp_count}")
        logger.info(f"false_negative_count(Missing objects, False Negative): {false_negative_count}")
        return

    def calculate_high_score_iou(self, predictions, ground_truths, score_threshold=0.25):
        """
        Calcuate the mean IoU for all matched prediction/GP pairs.

        :param self: Description
        :param predictions: Description
        :param ground_truths: Description
        """
        if not predictions or not ground_truths:
            return 0.0

        def compute_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
            yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
            #logger.info(f"xA: {xA} yA: {yA} xB: {xB} yB: {yB}")
            interArea = max(0, xB-xA) * max(0, yB - yA)
            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]
            #logger.info(f"interArea: {interArea}, boxAArea: {boxAArea}, boxBArea: {boxBArea}")
            iou = interArea/float(boxAArea + boxBArea - interArea + 1e-6)
            #logger.info(f"iou: {iou}")
            return iou


        gt_dict = {}
        for gt in ground_truths:
            img_id = gt['image_id']
            #logger.info(f"img_id: {img_id}")
            if img_id not in gt_dict:
                gt_dict[img_id] = []
            gt_dict[img_id].append(gt)

        total_iou = 0.0
        match_count = 0
        for pred in predictions:
            if pred.get('score', 0) < score_threshold:
                continue
            img_id = pred['image_id']
            #logger.info(f"img_id: {img_id}")
            if img_id in gt_dict:
                best_iou = 0
                #logger.info(f"best_iou: {best_iou}")
                for gt in gt_dict[img_id]:
                    if pred['category_id'] == gt['category_id']:
                        #logger.info(f"pred['category_id']: {pred['category_id']}")
                        #logger.info(f"gt['category_id']: {gt['category_id']}")
                        #logger.info(f"pred['bbox']: {pred['bbox']}")
                        #logger.info(f"gt['bbox']: {gt['bbox']}")
                        iou = compute_iou(pred['bbox'], gt['bbox'])
                        #logger.info(f"iou: {iou}, best_iou: {best_iou}")
                        best_iou = max(best_iou, iou)
                if best_iou > 0:
                    total_iou += best_iou
                    match_count += 1
                    #logger.info(f"total_iou: {total_iou}")
                    #logger.info(f"mathc_iou: {match_count}")
        logger.info(f"total_iou: {total_iou}")
        logger.info(f"match_count: {match_count}")
        return total_iou / match_count if match_count > 0  else 0.0


    def generate_detection_confusion_matrix(self, preds, gts, num_classes, iou_thresh=0.5, class_names=None):
        """
        preds: [{'image_id': 1, 'category_id': 1, 'bbox': [x,y,w,h], 'score': 0.5}, ...]
        gts: [{'image_id': 1, 'category_id': 1, 'bbox': [x,y,w,h]}, ...]
        num_classes: 배경을 제외한 클래스 개수
        """
        # 행렬 크기: (num_classes + 1) x (num_classes + 1) -> 마지막 인덱스는 'Background'
        matrix = np.zeros((num_classes + 1, num_classes + 1))

        # 1. 이미지별로 데이터 그룹화
        from collections import defaultdict
        img_preds = defaultdict(list)
        img_gts = defaultdict(list)
        for p in preds: img_preds[p['image_id']].append(p)
        for g in gts: img_gts[g['image_id']].append(g)

        # IoU 계산 함수
        def compute_iou(boxA, boxB):
            xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
            xB, yB = min(boxA[0]+boxA[2], boxB[0]+boxB[2]), min(boxA[1]+boxA[3], boxB[1]+boxB[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
            return inter / (union + 1e-6)

        # 2. 매칭 루프
        for img_id in img_gts.keys():
            current_gts = img_gts[img_id]
            current_preds = img_preds.get(img_id, [])

            matched_preds = set()
            for gt in current_gts:
                best_iou = -1
                best_pred_idx = -1

                for j, pred in enumerate(current_preds):
                    if j in matched_preds: continue
                    iou = compute_iou(gt['bbox'], pred['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = j

                # 클래스 인덱스 (0 ~ num_classes-1 가정, 배경은 num_classes)
                gt_cls = gt['category_id']

                if best_iou >= iou_thresh:
                    pred_cls = current_preds[best_pred_idx]['category_id']
                    matrix[gt_cls, pred_cls] += 1
                    matched_preds.add(best_pred_idx)
                else:
                    # False Negative: 물체가 있는데 못 찾음 (배경으로 예측됨)
                    matrix[gt_cls, num_classes] += 1

            # False Positives: 정답이 없는데 억지로 찾아낸 박스들
            for j, pred in enumerate(current_preds):
                if j not in matched_preds:
                    pred_cls = pred['category_id']
                    matrix[num_classes, pred_cls] += 1

        # 시각화
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)] + ("Background",)
        else:
            class_names = class_names +("Background",)

        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, fmt='.0f', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.xlabel('Predicted Class')
        plt.ylabel('Ground Truth Class')
        plt.title('Object Detection Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        logger.info("Confusion Matrix saved as confusion_matrix.png")

    def evaluate(
            self,
            model,
            distributed=False,
            half=True,
            trt_file=None,
            decoder=None,
            test_size=None,
            img_path=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        labels_list = []
        ori_data_list = []
        ori_label_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        #kssong
#        for cur_iter, (imgs, _, info_imgs, label, path, time_embedding) in enumerate(
#                progress_bar(self.dataloader)
#        ):
        prev_video_path = "None"

        for cur_iter, (imgs, _, info_imgs, label, path, time_embedding) in enumerate(
                progress_bar(self.dataloader)
        ):

            video_path = os.path.basename(os.path.dirname(path[0]))
            if prev_video_path == video_path:
                first_frame = False
            else:
                first_frame = True
                prev_video_path = video_path
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                #kssong
                outputs, ori_res = model(imgs, first_frame, nms_thresh=self.nmsthre,
                                         lframe=self.lframe,
                                         gframe = self.gframe)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
            if self.gl_mode:
                local_num = int(imgs.shape[0] / 2)
                info_imgs = info_imgs[:local_num]
                label = label[:local_num]
            if self.kwargs.get('first_only',False):
                info_imgs = [info_imgs[0]]
                label = [label[0]]
            #logger.info("label: {}".format(label))
            temp_data_list, temp_label_list = self.convert_to_coco_format(outputs, info_imgs, copy.deepcopy(label))
            data_list.extend(temp_data_list)
            labels_list.extend(temp_label_list)
            #logger.info("temp_labels_list: {}".format(temp_label_list))

        self.vid_to_coco['annotations'].extend(labels_list)
        #mean_iou = self.calculate_mean_iou(data_list, labels_list)
        #logger.info(f"Avearage IoU of detections: {mean_iou:.4f}")
        self.calculate_all_prediction_iou(data_list, labels_list)
        self.generate_detection_confusion_matrix(data_list, labels_list, 30, class_names=vid_classes)
        #logger.info(f"Avearage IoU of detections(All IoU): {all_iou:.4f}")
        #highscore_iou = self.calculate_high_score_iou(data_list, labels_list, 0.25)
        #logger.info(f"Avearage IoU of detections(highscore 0.25): {highscore_iou:.4f}")
        #highscore_iou = self.calculate_high_score_iou(data_list, labels_list, 0.50)
        #logger.info(f"Avearage IoU of detections(highscore 0.50): {highscore_iou:.4f}")
        #highscore_iou = self.calculate_high_score_iou(data_list, labels_list, 0.70)
        #logger.info(f"Avearage IoU of detections(highscore 0.70): {highscore_iou:.4f}")

        #json.dump(self.vid_to_coco, open("gt_annotations.json", 'w'))
        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        del labels_list
        eval_results = self.evaluate_prediction(data_list, statistics)
        del data_list
        self.vid_to_coco['annotations'] = []

        #kssong
        synchronize() #results in deadlock
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, labels):
        data_list = []
        label_list = []
        frame_now = 0

        for (output, info_img, _label) in zip(
                outputs, info_imgs, labels
        ):
            # if frame_now>=self.lframe: break
            scale = min(
                self.img_size[0] / float(info_img[0]), self.img_size[1] / float(info_img[1])
            )
            bboxes_label = _label[:, 1:]
            bboxes_label /= scale
            bboxes_label = xyxy2xywh(bboxes_label)
            cls_label = _label[:, 0]
            for ind in range(bboxes_label.shape[0]):
                label_pred_data = {
                    "image_id": int(self.id),
                    "category_id": int(cls_label[ind]),
                    "bbox": bboxes_label[ind].numpy().tolist(),
                    "segmentation": [],
                    'id': self.box_id,
                    "iscrowd": 0,
                    'area': int(bboxes_label[ind][2] * bboxes_label[ind][3])
                }  # COCO json format
                self.box_id = self.box_id + 1
                label_list.append(label_pred_data)
            self.vid_to_coco['images'].append({'id': self.id})

            if output is None:
                self.id = self.id + 1
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]
            # preprocessing: resize
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])
                pred_data = {
                    "image_id": int(self.id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
            self.id = self.id + 1
            frame_now = frame_now + 1

        return data_list, label_list

    def convert_to_coco_format_ori(self, outputs, info_imgs, labels):

        data_list = []
        label_list = []
        frame_now = 0
        for (output, info_img, _label) in zip(
                outputs, info_imgs, labels
        ):
            scale = min(
                self.img_size[0] / float(info_img[0]), self.img_size[1] / float(info_img[1])
            )
            bboxes_label = _label[:, 1:]
            bboxes_label /= scale
            bboxes_label = xyxy2xywh(bboxes_label)
            cls_label = _label[:, 0]
            for ind in range(bboxes_label.shape[0]):
                label_pred_data = {
                    "image_id": int(self.id_ori),
                    "category_id": int(cls_label[ind]),
                    "bbox": bboxes_label[ind].numpy().tolist(),
                    "segmentation": [],
                    'id': self.box_id_ori,
                    "iscrowd": 0,
                    'area': int(bboxes_label[ind][2] * bboxes_label[ind][3])
                }  # COCO json format
                self.box_id_ori = self.box_id_ori + 1
                label_list.append(label_pred_data)

                # print('label:',label_pred_data)

            self.vid_to_coco_ori['images'].append({'id': self.id_ori})

            if output is None:
                self.id_ori = self.id_ori + 1
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            # print(cls.shape)
            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])
                pred_data = {
                    "image_id": int(self.id_ori),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

            self.id_ori = self.id_ori + 1
            frame_now = frame_now + 1
        return data_list, label_list

    def evaluate_prediction(self, data_dict, statistics, ori=False):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_sampler.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_sampler.batch_size)
        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                ["forward", "NMS", "inference"],
                [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
            )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:

            _, tmp = tempfile.mkstemp()
            if ori:
                json.dump(self.vid_to_coco_ori, open(self.gt_ori, 'w'))
                json.dump(data_dict, open(self.tmp_name_ori, 'w'))
                json.dump(self.vid_to_coco_ori, open(tmp, "w"))
            else:
                json.dump(self.vid_to_coco, open(self.gt_refined, 'w'))
                json.dump(data_dict, open(self.tmp_name_refined, 'w'))
                json.dump(self.vid_to_coco, open(tmp, "w"))

            cocoGt = pycocotools.coco.COCO(tmp)
            # TODO: since pycocotools can't process dict in py36, write data to json file.

            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()

            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
            info += "per class AP:\n" + AP_table + "\n"

            AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
            info += "per class AR:\n" + AR_table + "\n"
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
