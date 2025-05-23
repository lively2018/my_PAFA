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

        # self.aggregator = SelsaAggregator(in_channels)

    def reset(self):
        #kssong
        if self.feat is not None:
            del self.feat  # Explicitly delete the tensor
            torch.cuda.empty_cache()  # Free GPU memory        
        self.feat = None
        #print(f"reset_memory")            
    
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
            self.feat = feat.detach().clone()
        else:
            new_feat = torch.cat([self.feat, feat], dim=0).detach().clone()
            del self.feat
            torch.cuda.empty_cache()
            self.feat = new_feat
        #print(f"init_memory, memory bank size: {len(self.feat)}, gpu memory usage: {gpu_mem_usage():.0f}")                    

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
            return self.feat.detach().clone()
                       
        if self.sampling_policy == 'random':
            sampled_ind = torch.randperm(len(self.feat), device=self.feat.device)[:self.key_length]
            #print(f"sample, memory bank size: {len(self.feat)}, gpu memory usage: {gpu_mem_usage():.0f}") 
            return self.feat[sampled_ind].detach().clone()
        else:
            raise NotImplementedError

    def update(self, new_feat):
        #kssong
        #if self.feat is None:
            # first time
            #self.feat = new_feat
            #return       
        #print(f"Before update: {torch.cuda.memory_allocated()} / {torch.cuda.memory_reserved()}")    
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
