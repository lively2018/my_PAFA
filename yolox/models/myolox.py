#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

#kssong
import torch
def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """    
    return torch.cuda.max_memory_allocated() / (1024 * 1024)

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head

    
    #kssong
    #def forward(self, x, targets=None,nms_thresh=0.5,lframe=0,gframe=32):
    def forward(self, x, first, targets=None,nms_thresh=0.5,lframe=0,gframe=32):
        #print(f"Before YOLOX forward: {gpu_mem_usage():.0f}")        
        # fpn output content features of [dark3, dark4, dark5]        
        fpn_outs = self.backbone(x)
        #print(f"After backbone in YOLOX forward: {gpu_mem_usage():.0f}")        
        if self.training:
            assert targets is not None
            #kssong
            #loss, iou_loss, conf_loss, cls_loss, l1_loss, rconf_loss,num_fg = self.head(
            #    fpn_outs, targets, x, lframe=lframe,gframe=gframe
            #)
            loss, iou_loss, conf_loss, cls_loss, l1_loss, rconf_loss,num_fg = self.head(
                fpn_outs, first, targets, x, lframe=lframe,gframe=gframe
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "rconf_loss":rconf_loss,
                "num_fg": num_fg,
            }
        else:
            #kssong
            #outputs = self.head(fpn_outs,targets,x,nms_thresh=nms_thresh, lframe=lframe,gframe=gframe)
            outputs = self.head(fpn_outs,first,targets,x,nms_thresh=nms_thresh, lframe=lframe,gframe=gframe)
        #print(f"After YOLOX forward: {gpu_mem_usage():.0f}")            
        return outputs
