# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from .memory_bank import MemoryBank
from loguru import logger
#kssong
def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """    
    return torch.cuda.max_memory_allocated() / (1024 * 1024)

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
        self.memory_bank = MemoryBank(**memory_cfg)        


    def forward(self, x, ref_x):
        #kssong
        ref_x = self.memory_bank.sample()
        #logger.info(f"ref_x shape: {ref_x.shape} x shape: {x.shape}")        
        #print(f"After sampling: {gpu_mem_usage():.0f}")
        # fort he rest frames
        aggregated_x = self.forward_with_ref_x(x, ref_x)            
        return aggregated_x       
        
    
    def reset_memory_bank(self):
        #logger.info("reset_memory_bank")
        self.memory_bank.reset()

    def update_memory_bank(self, x):
        #logger.info("update_memory_bank")
        self.memory_bank.update(x)
    
    def init_memory_bank(self, x):
        #logger.info("init_memory_bank")
        self.memory_bank.init_memory(x)

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
        
        #logger.info(f"roi_n: {roi_n} ref_roi_n: {ref_roi_n}")        
        x_embed = self.fc_embed(x)
        # [num_attention_blocks, roi_n, C / num_attention_blocks]
        x_embed = x_embed.view(roi_n, self.num_attention_blocks,
                               -1).permute(1, 0, 2)
        
        ref_x_embed = self.ref_fc_embed(ref_x)
        # [num_attention_blocks, C / num_attention_blocks, ref_roi_n]
        ref_x_embed = ref_x_embed.view(ref_roi_n, self.num_attention_blocks,
                                       -1).permute(1, 2, 0)

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
