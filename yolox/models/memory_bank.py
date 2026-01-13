import torch
import torch.nn as nn
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
                 max_length=4800, key_length=480,feature_dim=256, device='cuda',
                 sampling_policy='random', updating_policy='random',
                 ):
        super().__init__()
        #kssong
        self.max_length = max_length
        self.key_length = key_length
        self.sampling_policy = sampling_policy
        self.updating_policy = updating_policy
        #kssong
        self.feat = torch.zeros((max_length, feature_dim), device=device)
        self.current_size = 0
        self.device = device
        

    def reset(self):
        #kssong
        self.current_size = 0

    def init_memory(self, feat):
        """
        init memory
        Args:
            feat: tensor [m, c]

        Returns:

        """
        #kssong
        print(f"Bank Dim: {self.feat.shape[1]}, Input Feat Dim: {feat.shape[1]}")
        feat = feat.to('cuda', non_blocking=True)
        num_new = feat.size(0)
        if self.feat is None:
            self.feat = torch.zero((self.max_length, feat.size(1)), device='cuda')
            self.current_size = 0

        if num_new > self.max_length:
            feat = feat[-self.max_length:]
            num_new = self.max_length
        self.feat[:num_new] = feat        
        self.current_size = num_new

    def sample(self):
        #kssong
        if self.current_size == 0:
            # write first
            return []

        actual_limit = self.current_size
        if actual_limit < self.key_length:
            #print(f"sample, memory bank size: {len(self.feat)}, gpu memory usage: {gpu_mem_usage():.0f}")
            return self.feat[:actual_limit]
        if self.sampling_policy == "random":
            inds = torch.randint(0, actual_limit, (self.key_length,), device=self.device)
        else:
            raise NotImplementedError("not implemented")
        return self.feat[inds]
    
    def update(self, new_feat):
        new_feat = new_feat.to(self.device, non_blocking=True)
        num_new = new_feat.size(0)

        if self.feat is None:
            self.feat = torch.zeros((self.max_length, new_feat.size(1)), device='cuda')

        if self.current_size + num_new <= self.max_length:
            self.feat[self.current_size : self.current_size + num_new] = new_feat
            self.current_size += num_new            
        else:
            if self.updating_policy == "random":
                replace_ids = torch.randomperm(self.current_size, device='cuda')[:num_new]
                self.feat[replace_ids] = new_feat
                self.current_size = self.max_length
            else:
                raise NotImplementedError("not implemented")

