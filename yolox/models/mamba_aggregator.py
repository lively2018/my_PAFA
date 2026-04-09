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
            writer.writerow(['video_path', 'batch_set_count', 'level', 'count', 'mem_len']) # Header
        writer.writerow(data)

def log_stats_to_file_csv(filename, data):
    # data format: [frame_idx, level_name, feature_count, gpu_mem_mb]
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['frame_idx', 'level', 'count', 'mem_len']) # Header
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

    def __init__(self, in_channels, num_attention_blocks=16, memory_length=4800, key_length=480, **memory_cfg):
        super(MambaAggregator, self).__init__()
        self.fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc_embed = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.num_attention_blocks = num_attention_blocks

        # instance-level memory bank
        self.memory_bank_p3 = MemoryBank(max_length=memory_length, key_length=key_length, **memory_cfg)
        self.memory_bank_p4 = MemoryBank(max_length=memory_length, key_length=key_length, **memory_cfg)
        self.memory_bank_p5 = MemoryBank(max_length=memory_length, key_length=key_length, **memory_cfg)
        self.batchset_count = 0
        self.video_count = 0
        self.video_path = None
        self.memory_bank_info = []
        self.count = 0
        self.max_memory_bank_length = memory_length
        self.key_length = key_length

    def forward(self, x, ref_x, k_type):
        #kssong
        if k_type == 0:
            ref_x = self.memory_bank_p3.sample()
        elif k_type == 1:
            ref_x = self.memory_bank_p4.sample()
        elif k_type == 2:
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


    def reset_memory_bank(self, k_type, video_path=None):
        #logger.info("reset_memory_bank")
        self.batchset_count = 0
        self.video_path = video_path
        self.memory_bank_info = []
        self.count += 1
        self.video_count += 1
        memory_bank_p3_length, memory_bank_p4_length, memory_bank_p5_length = 0, 0, 0
        update_length = 0
        if k_type == 0:
            memory_bank_p3_length, update_length = self.memory_bank_p3.reset()
            if update_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank reset, but update_length {update_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            log_stats_to_file_csv('memory_stats.csv', [self.count, 'P3', update_length,  memory_bank_p3_length])
        elif k_type == 1:
            memory_bank_p4_length, update_length = self.memory_bank_p4.reset()
            if update_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank reset, but update_length {update_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            log_stats_to_file_csv('memory_stats.csv', [self.count, 'P4', update_length,  memory_bank_p4_length])
        elif k_type == 2:
            memory_bank_p5_length, update_length = self.memory_bank_p5.reset()
            if update_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank reset, but update_length {update_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            log_stats_to_file_csv('memory_stats.csv', [self.count, 'P5', update_length,  memory_bank_p5_length])


    def update_memory_bank(self, x, k_type):
        #logger.info("update_memory_bank")
        self.batchset_count += 1
        self.count += 1
        memory_bank_p3_length, memory_bank_p4_length, memory_bank_p5_length = 0, 0, 0
        update_length = 0
        if k_type == 0:
            memory_bank_p3_length, update_length = self.memory_bank_p3.update(x)
            if update_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but update_length {update_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p3_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but memory_bank_p3_length {memory_bank_p3_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p3_length == self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but memory_bank_p3_length {memory_bank_p3_length} reaches max_memory_bank_length {self.max_memory_bank_length}")
            #if update_length == 0:
            #    logger.warning(f"Memory bank update, but update_length is {update_length} x: {x.shape[0]}, video_path: {self.video_path}, batchset_count: {self.batchset_count}, level: P3, memory_bank_p3_length: {memory_bank_p3_length}")
            #self.memory_bank_info.append([self.video_path, self.batchset_count, 'P3', x.shape[0], self.memory_bank_p3.len()])
            #logger.info(f"update_memory_bank, video_path: {self.video_path}, batchset_count: {self.batchset_count}, level: P3, update_length: {update_length}, memory_bank_p3_length: {memory_bank_p3_length}")
            log_stats_to_csv(f'memory_bank_stats_video.csv', [self.video_path, self.batchset_count, 'P3', update_length, memory_bank_p3_length])
            log_stats_to_file_csv('memory_stats.csv', [self.count, 'P3', update_length, memory_bank_p3_length])
        elif k_type == 1:
            memory_bank_p4_length, update_length = self.memory_bank_p4.update(x)
            if update_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but update_length {update_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p4_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but memory_bank_p4_length {memory_bank_p4_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p4_length == self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but memory_bank_p4_length {memory_bank_p4_length} reaches max_memory_bank_length {self.max_memory_bank_length}")

            #if update_length == 0:
            #    logger.warning(f"Memory bank update, but update_length is {update_length} x: {x.shape[0]}, video_path: {self.video_path}, batchset_count: {self.batchset_count}, level: P4, memory_bank_p4_length: {memory_bank_p4_length}")
            #self.memory_bank_info.append([self.video_path, self.batchset_count, 'P4', x.shape[0], self.memory_bank_p4.len()])
            #logger.info(f"update_memory_bank, video_path: {self.video_path}, batchset_count: {self.batchset_count}, level: P4, update_length: {update_length}, memory_bank_p4_length: {memory_bank_p4_length}")
            log_stats_to_csv(f'memory_bank_stats_video.csv', [self.video_path, self.batchset_count, 'P4', update_length, memory_bank_p4_length])
            log_stats_to_file_csv('memory_stats.csv', [self.count, 'P4', update_length, memory_bank_p4_length])
        elif k_type == 2:
            memory_bank_p5_length, update_length = self.memory_bank_p5.update(x)
            if update_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but update_length {update_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p5_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but memory_bank_p5_length {memory_bank_p5_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p5_length == self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but memory_bank_p5_length {memory_bank_p5_length} reaches max_memory_bank_length {self.max_memory_bank_length}")
            #if update_length == 0:
            #    logger.warning(f"Memory bank update, but update_length is {update_length} x: {x.shape[0]}, video_path: {self.video_path}, batchset_count: {self.batchset_count}, level: P5, memory_bank_p5_length: {memory_bank_p5_length}")
            #self.memory_bank_info.append([self.video_path, self.batchset_count, 'P5', x.shape[0], self.memory_bank_p5.len()])
            #logger.info(f"update_memory_bank, video_path: {self.video_path}, batchset_count: {self.batchset_count}, level: P5, update_length: {update_length}, memory_bank_p5_length: {memory_bank_p5_length}")
            log_stats_to_csv(f'memory_bank_stats_video.csv', [self.video_path, self.batchset_count, 'P5', update_length, memory_bank_p5_length])
            log_stats_to_file_csv('memory_stats.csv', [self.count, 'P5', update_length, memory_bank_p5_length])

    def init_memory_bank(self, x, k_type):
        #logger.info("init_memory_bank")
        #logger.info("x.shape: {}".format(x.shape))
        self.batchset_count += 1
        self.count += 1
        memory_bank_p3_length, memory_bank_p4_length, memory_bank_p5_length = 0, 0, 0
        update_length = 0
        if k_type == 0:
            memory_bank_p3_length, update_length = self.memory_bank_p3.init_memory(x)
            if update_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank init, but update_length {update_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p3_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank init, but memory_bank_p3_length {memory_bank_p3_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p3_length == self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but memory_bank_p3_length {memory_bank_p3_length} reaches max_memory_bank_length {self.max_memory_bank_length}")

            #self.memory_bank_info.append([self.video_path, self.batchset_count, 'P3', x.shape[0], self.memory_bank_p3.len()])
            #logger.info(f"init_memory_bank, video_path: {self.video_path}, batchset_count: {self.batchset_count}, level: P3, update_length: {update_length}, memory_bank_p3_length: {memory_bank_p3_length}")
            log_stats_to_csv(f'memory_bank_stats_video.csv', [self.video_path, self.batchset_count, 'P3', update_length, memory_bank_p3_length])
            log_stats_to_file_csv('memory_stats.csv', [self.count, 'P3', update_length, memory_bank_p3_length])
        elif k_type == 1:
            memory_bank_p4_length, update_length = self.memory_bank_p4.init_memory(x)
            if update_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank init, but update_length {update_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p4_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank init, but memory_bank_p4_length {memory_bank_p4_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p4_length == self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but memory_bank_p4_length {memory_bank_p4_length} reaches max_memory_bank_length {self.max_memory_bank_length}")
            #self.memory_bank_info.append([self.video_path, self.batchset_count, 'P4', x.shape[0], self.memory_bank_p4.len()])
            #logger.info(f"init_memory_bank, video_path: {self.video_path}, batchset_count: {self.batchset_count}, level: P4, update_length: {update_length}, memory_bank_p4_length: {memory_bank_p4_length}")
            log_stats_to_csv(f'memory_bank_stats_video.csv', [self.video_path, self.batchset_count, 'P4', update_length, memory_bank_p4_length])
            log_stats_to_file_csv('memory_stats.csv', [self.count, 'P4', update_length, memory_bank_p4_length])
        elif k_type == 2:
            memory_bank_p5_length, update_length = self.memory_bank_p5.init_memory(x)
            if update_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank init, but update_length {update_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p5_length > self.max_memory_bank_length:
                logger.warning(f"Memory bank init, but memory_bank_p5_length {memory_bank_p5_length} exceeds max_memory_bank_length {self.max_memory_bank_length}")
            if memory_bank_p5_length == self.max_memory_bank_length:
                logger.warning(f"Memory bank update, but memory_bank_p5_length {memory_bank_p5_length} reaches max_memory_bank_length {self.max_memory_bank_length}")
            #self.memory_bank_info.append([self.video_path, self.batchset_count, 'P5', x.shape[0], self.memory_bank_p5.len()])
            #logger.info(f"init_memory_bank, video_path: {self.video_path}, batchset_count: {self.batchset_count}, level: P5, update_length: {update_length}, memory_bank_p5_length: {memory_bank_p5_length}")
            log_stats_to_csv(f'memory_bank_stats_video.csv', [self.video_path, self.batchset_count, 'P5', update_length, memory_bank_p5_length])
            log_stats_to_file_csv('memory_stats.csv', [self.count, 'P5', update_length, memory_bank_p5_length])

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
