# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from .memory_bank import MemoryBank
from loguru import logger
#kssong
import csv
import os

def log_stats_to_csv(filename, data):
    # data format: [frame_idx, level_name, feature_count, gpu_mem_mb]
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['frame_idx', 'level', 'count', 'gpu_mem_mb']) # Header
        writer.writerow(data)

class MambaAggregator(nn.Module):
    """Selsa aggregator module.

    This module is proposed in "Sequence Level Semantics Aggregation for Video
    Object Detection". `SELSA <https://arxiv.org/abs/1907.06390>`_.

    Args:
        in_channels (int): The number of channels of the features of
            proposal.
        num_attention_blocks (int): The number of attention blocks used in
            selsa aggregator module. Defaults to 16
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, in_channels, num_attention_blocks=16, memory_cfg=dict()):
        super(MambaAggregator, self).__init__()
        self.fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc_embed = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.num_attention_blocks = num_attention_blocks

        # instance-level memory bank
        self.memory_bank_p3 = MemoryBank(**memory_cfg)
        self.memory_bank_p4 = MemoryBank(**memory_cfg)
        self.memory_bank_p5 = MemoryBank(**memory_cfg)
        self.frame_count = 0
        self.video_path = None
        self.memory_bank_info = []
        self.video_count = 0

    def forward(self, x, ref_x, type):
        #kssong
        if type == 0:
            ref_x = self.memory_bank_p3.sample()
        elif type == 1:
            ref_x = self.memory_bank_p4.sample()
        elif type == 2:
            ref_x = self.memory_bank_p5.sample()
        #logger.info(f"ref_x shape: {ref_x.shape} x shape: {x.shape}")
        #print(f"After sampling: {gpu_mem_usage():.0f}")
        # fort he rest frames
        if len(ref_x) != 0:
            aggregated_x = self.forward_with_ref_x(x, ref_x)
        else:
            #logger.info(f"ref_x shape: {ref_x.shape}")
            aggregated_x = torch.zeros_like(x)
        return aggregated_x


    def reset_memory_bank(self, type, video_path=None):
        #logger.info("reset_memory_bank")
        previous_memory_bank_info = self.memory_bank_info
        if previous_memory_bank_info:
            logger.info(f"Previous memory bank info: {previous_memory_bank_info}")
            log_stats_to_csv(f'memory_stats.csv', previous_memory_bank_info)
        else:
            logger.info("No previous memory bank info to log.")
        self.frame_count = 0
        self.video_path = video_path
        self.memory_bank_info = []
        self.video_count += 1
        if type == 0:
            self.memory_bank_p3.reset()
        elif type == 1:
            self.memory_bank_p4.reset()
        elif type == 2:
            self.memory_bank_p5.reset()


    def update_memory_bank(self, x, type):
        #logger.info("update_memory_bank")
        self.frame_count += 1
        if type == 0:
            self.memory_bank_p3.update(x)
            self.memory_bank_info.append([self.video_path, self.frame_count, 'P3', x.shape[0], self.memory_bank_p3.len()])
        elif type == 1:
            self.memory_bank_p4.update(x)
            self.memory_bank_info.append([self.video_path, self.frame_count, 'P4', x.shape[0], self.memory_bank_p4.len()])
        elif type == 2:
            self.memory_bank_p5.update(x)
            self.memory_bank_info.append([self.video_path, self.frame_count, 'P5', x.shape[0], self.memory_bank_p5.len()])

    def init_memory_bank(self, x, type):
        #logger.info("init_memory_bank")
        #logger.info("x.shape: {}".format(x.shape))
        self.frame_count += 1
        if type == 0:
            self.memory_bank_p3.init_memory(x)
            self.memory_bank_info.append([self.video_path, self.frame_count, 'P3', x.shape[0], self.memory_bank_p3.len()])
        elif type == 1:
            self.memory_bank_p4.init_memory(x)
            self.memory_bank_info.append([self.video_path, self.frame_count, 'P4', x.shape[0], self.memory_bank_p4.len()])
        elif type == 2:
            self.memory_bank_p5.init_memory(x)
            self.memory_bank_info.append([self.video_path, self.frame_count, 'P5', x.shape[0], self.memory_bank_p5.len()])

    def forward_with_ref_x(self, x, ref_x):
        """Aggregate the features `ref_x` of reference proposals.

        The aggregation mainly contains two steps:
        1. Use multi-head attention to computing the weight between `x` and
        `ref_x`.
        2. Use the normlized (i.e. softmax) weight to weightedly sum `ref_x`.

        Args:
            x (Tensor): of shape [N, C]. N is the number of key frame
                proposals.
            ref_x (Tensor): of shape [M, C]. M is the number of reference frame
                proposals.

        Returns:
            Tensor: The aggregated features of key frame proposals with shape
            [N, C].    for first in firsts:
                feat_firsts.append(first)
                feat_firsts.append(first)
                feat_firsts.append(firsts)
        """
        roi_n = x.shape[0]
        ref_roi_n = ref_x.shape[0]

        x = x.half()
        #logger.info(f"roi_n: {roi_n} ref_roi_n: {ref_roi_n}")
        x_embed = self.fc_embed(x)
        # [num_attention_blocks, roi_n, C / num_attention_blocks]
        x_embed = x_embed.view(roi_n, self.num_attention_blocks,
                               -1).permute(1, 0, 2)
        #logger.info(f"x_embed shpae: {x_embed.shape}")
        ref_x = ref_x.half()
        ref_x_embed = self.ref_fc_embed(ref_x)
        # [num_attention_blocks, C / num_attention_blocks, ref_roi_n]
        ref_x_embed = ref_x_embed.view(ref_roi_n, self.num_attention_blocks,
                                       -1).permute(1, 2, 0)
        #logger.info(f"ref_x_embed shpae: {ref_x_embed.shape}")
        # [num_attention_blocks, roi_n, ref_roi_n]
        weights = torch.bmm(x_embed, ref_x_embed) / (x_embed.shape[-1]**0.5)
        weights = weights.softmax(dim=2)

        ref_x_new = self.ref_fc(ref_x)
        # [num_attention_blocks, ref_roi_n, C / num_attention_blocks]
        ref_x_new = ref_x_new.view(ref_roi_n, self.num_attention_blocks,
                                   -1).permute(1, 0, 2)

        # [roi_n, num_attention_blocks, C / num_attention_blocks]
        x_new = torch.bmm(weights, ref_x_new).permute(1, 0, 2).contiguous()
        # [roi_n, C]
        x_new = self.fc(x_new.view(roi_n, -1))

        return x_new
