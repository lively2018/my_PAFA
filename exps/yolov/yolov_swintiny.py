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
        self.backbone_name = 'Swin_Tiny'
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.pre_no_aug = 2
        #kssong
        self.multiscale_range = 0
        self.input_size = (640, 640)  # (height, width)
