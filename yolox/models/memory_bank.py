from loguru import logger
import torch
import torch.nn as nn
from yolox.data.datasets.vid_classes import VID_classes
#kssong
#from mmcv.runner import BaseModule
# from ..aggregators.selsa_aggregator import SelsaAggregator
import gc
def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    return torch.cuda.max_memory_allocated() / (1024 * 1024)

class MemoryBank(nn.Module):
    #def __init__(self,
    #             max_length=20000, key_length=2000,
    #             sampling_policy='random', updating_policy='random',
    #             ):
    def __init__(self,
                 max_length=4800, key_length=480,
                 sampling_policy='random', updating_policy='random',
                 ):
        super().__init__()
        #kssong
        self.max_length = max_length
        self.key_length = key_length
        self.sampling_policy = sampling_policy
        self.updating_policy = updating_policy
        #kssong
        self.feat = None
        self.feat_info = []
        self.class_num = len(VID_classes)
        self.mem_type = None
        self.sampled_ind = None
        # self.aggregator = SelsaAggregator(in_channels)

    def reset(self):
        #kssong
        if self.feat is not None:
            del self.feat  # Explicitly delete the tensor
            torch.cuda.empty_cache()  # Free GPU memory
        self.feat = None
        #print(f"reset_memory")
    def init_memory_features_info(self, feat_info, batch_set, batch_item, type_k):
        self.mem_type = type_k
        #kssong
        #logger.info(f"init_memory_features_info, mem_type: {self.mem_type}, \
        #            batch_set: {batch_set}, batch_item: {batch_item}, len(feat_info): {len(feat_info)}")
        non_zero_feat_info = [feat_info_item for feat_info_item in feat_info if not torch.all(feat_info_item == torch.zeros(6 + self.class_num, device=feat_info_item.device))]
        self.feat_info.extend([[batch_set, batch_item, feat_info_item] for feat_info_item in non_zero_feat_info])
        #logger.info(f"After filtering zero feat info, mem_type: {self.mem_type}, \
        #            batch_set: {batch_set}, batch_item: {batch_item}, len(non_zero_feat_info): {len(non_zero_feat_info)}, \
        #            total feat info in memory bank: {len(self.feat_info)}")
    def update_memory_features_info(self, feat_info, batch_set, batch_item, type_k):
        self.mem_type = type_k
        #kssong
        #logger.info(f"update_memory_features_info, mem_type: {self.mem_type}, \
        #            batch_set: {batch_set}, batch_item: {batch_item}, len(feat_info): {len(feat_info)}")
        non_zero_feat_info = [feat_info_item for feat_info_item in feat_info if not torch.all(feat_info_item == torch.zeros(6 + self.class_num, device=feat_info_item.device))]
        self.feat_info.extend([[batch_set, batch_item, feat_info_item] for feat_info_item in non_zero_feat_info])
        #logger.info(f"After filtering zero feat info, mem_type: {self.mem_type}, \
        #            batch_set: {batch_set}, batch_item: {batch_item}, len(non_zero_feat_info): {len(non_zero_feat_info)}, \
        #            total feat info in memory bank: {len(self.feat_info)}")
    def init_memory(self, feat):
        """
        init memory
        Args:
            feat: tensor [m, c]

        Returns:

        """
        #kssong
        #self.feat = feat
        #self.feat_num, self.feat_dim, self.feat_channel  = feat.shape
        # reshape [ n*m, c]
        #reshaped_feat = feat.view(-1, self.feat_channel)

        if self.feat is None:
            #self.feat = reshaped_feat
            self.feat = feat.detach().clone().to('cuda')
        else:
            new_feat = torch.cat([self.feat, feat], dim=0).detach().clone().to('cuda')
            del self.feat
            torch.cuda.empty_cache()
            self.feat = new_feat
        #print(f"init_memory, memory bank size: {len(self.feat)}, gpu memory usage: {gpu_mem_usage():.0f}")
    def get_sampled_features(self):
        if self.feat_info is None:
            return []
        if self.sampled_ind is None:
            return []
        return [self.feat_info[i] for i in self.sampled_ind.long().tolist()]

    def sample(self):
        #kssong
        if self.feat is None:
            # write first
            return []

        #if len(self.feat) < self.key_length:
        #    return self.feat
        feat_length = len(self.feat)
        if feat_length < self.key_length:
            #print(f"sample, memory bank size: {len(self.feat)}, gpu memory usage: {gpu_mem_usage():.0f}")
            self.sampled_ind = torch.arange(len(self.feat), device=self.feat.device)
            return self.feat.detach().clone().to('cuda')

        if self.sampling_policy == 'random':
            sampled_ind = torch.randperm(len(self.feat), device=self.feat.device)[:self.key_length]
            self.sampled_ind = sampled_ind
            #print(f"sample, memory bank size: {len(self.feat)}, gpu memory usage: {gpu_mem_usage():.0f}")
            return self.feat[sampled_ind].detach().clone().to('cuda')
        else:
            raise NotImplementedError

    def update(self, new_feat):
        #kssong
        #if self.feat is None:
            # first time
            #self.feat = new_feat
            #return
        #print(f"Before update: {torch.cuda.memory_allocated()} / {torch.cuda.memory_reserved()}")
        new_feat = new_feat.to('cuda')
        if self.feat is None:
            self.feat = new_feat.detach().clone()
            return

        if len(self.feat) < self.max_length:
            new_feat_combined = torch.cat([self.feat, new_feat], dim=0).detach().clone()

        elif self.updating_policy == "random":
            new_num = len(new_feat)
            reserved_ind = torch.randperm(len(self.feat), device=self.feat.device)[:-new_num]
            new_feat_combined = torch.cat([self.feat[reserved_ind], new_feat], dim=0).detach().clone()

        else:
            raise NotImplementedError("not implemented")


        del self.feat
        torch.cuda.empty_cache()
        self.feat = new_feat_combined

        gc.collect()
        torch.cuda.empty_cache()
        #print(f"memory bank update, memory bank size: {len(self.feat)} gpu memory usage: {gpu_mem_usage():.0f}")

    def __len__(self):
        if self.feat is None:
            return 0
        return len(self.feat)

    # def forward(self, x, x_support=None):
    #     # inference
    #     if x_support is None:
    #         raise NotImplementedError
    #     # training
    #     else:
    #         x = x + self.aggregator(x, x_support)
    #         return x
