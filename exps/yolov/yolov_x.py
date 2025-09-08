import os
import torch.nn as nn
import sys
sys.path.append("..")
from exps.yolov.yolov_base import Exp as MyExp
from yolox.data.datasets import vid
from loguru import logger
import torch


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path

        self.num_classes = 30
        self.data_dir = "/home/kssong"
        self.train_ann = "vid_train_coco.json"
        self.val_ann = "vid_val10000_coco.json"
        self.max_epoch = 20
        self.no_aug_epochs = 2
        self.pre_no_aug = 2
        self.warmup_epochs = 1
        self.eval_interval = 1
        self.min_lr_ratio = 0.5
        self.basic_lr_per_img = 0.002 / 64.0
        self.test_size = (640, 640)
        self.input_size = (640, 640)
        self.nmsthre = 0.5
        self.m_conf = 0
        #global frames for training
        self.gframe = 16
        #globale frames for validation
        #self.gframe_val = 32
        self.gframe_val = 16
        #sequence number for validation,-1 denote all
        self.tnum = -1
        #sequence number for train,-1 denote all
        self.tseq = -1
        self.pre_nms = 0.75
        self.multiscale_range = 0

    def get_evaluator(self, val_loader):
        from yolox.evaluators.vid_evaluator_v2 import VIDEvaluator

        # val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VIDEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        
        return evaluator

    def get_data_loader(
            self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import TrainTransform
        from yolox.data.datasets.mosaicdetection import MosaicDetection_VID

        #dataset = vid.VIDDataset(file_path='./yolox/data/datasets/train_seq.npy',
        #                         img_size=self.input_size,
        #                         preproc=TrainTransform(
        #                             max_labels=50,
        #                             flip_prob=self.flip_prob,
        #                             hsv_prob=self.hsv_prob),
        #                         lframe=0,  # batch_size,
        #                         gframe=batch_size,
        #                         dataset_pth=self.data_dir)
        dataset = vid.VIDRefDataset(file_path=self.vid_train_path,
                                img_size=self.input_size,
                                preproc=TrainTransform(
                                     max_labels=50,
                                     flip_prob=self.flip_prob,
                                     hsv_prob=self.hsv_prob),
                                lframe=self.lframe,  # batch_size,
                                gframe=self.gframe,                               
                                dataset_pth=self.data_dir,
                                mode=self.mode,
                                tseq=self.tseq,
                                local_stride=self.local_stride,
                                )
        dataset = MosaicDetection_VID(
            dataset,
            mosaic=False,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
            dataset_path=self.data_dir
        )
        dataset = vid.get_trans_loader(batch_size=batch_size, data_num_workers=4, dataset=dataset)
        return dataset

