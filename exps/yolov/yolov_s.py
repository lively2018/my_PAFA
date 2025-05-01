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
        self.depth = 0.33  # 1#0.67
        self.width = 0.5  # 1#0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.pre_no_aug = 2
        #self.warmup_epochs = 0
        #kssong
        self.multiscale_range = 0
        self.input_size = (640, 640)  # (height, width)


