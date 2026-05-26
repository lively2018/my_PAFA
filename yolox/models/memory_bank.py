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
        self.feat = None
        #print(f"reset_memory")
        return 0, 0

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


        #self.feat = reshaped_feat
        if len(feat) >= self.max_length:
            self.feat = feat[:self.max_length].detach().clone().to('cuda')
            update_length = self.max_length
        else:
            self.feat = feat.detach().clone().to('cuda')
            update_length = len(feat)
        #print(f"init_memory, memory bank max size: {self.max_length}, memory_bank key size: {self.key_length}")
        return len(self.feat), update_length

    def sample(self):
        #kssong
        if self.feat is None:
            # write first
            return []

        #if len(self.feat) < self.key_length:
        #    return self.feat
        feat_length = len(self.feat)
        if feat_length <= self.key_length:
            #print(f"sample, memory bank size: {len(self.feat)}, gpu memory usage: {gpu_mem_usage():.0f}")
            return self.feat.detach().clone().to('cuda')

        if self.sampling_policy == 'random':
            sampled_ind = torch.randperm(len(self.feat), device=self.feat.device)[:self.key_length]
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
        update_length = 0
        if self.feat is None:
            if len(new_feat) >= self.max_length:
                new_feat_combined = new_feat[:self.max_length].detach().clone()
                update_length = self.max_length
            else:
                new_feat_combined = new_feat.detach().clone()
                update_length = len(new_feat)

        else:
            if len(new_feat) + len(self.feat) >= self.max_length:
                if self.updating_policy == "random":
                    if len(new_feat) >= self.max_length:
                        new_feat_combined = new_feat[:self.max_length].detach().clone()
                        update_length = self.max_length
                    else:
                        slots_available = self.max_length - len(self.feat)
                        num_replace = len(new_feat) - slots_available
                        if num_replace <= 0:
                            new_feat_combined = torch.cat([self.feat, new_feat], dim=0).detach().clone()
                        else:
                            old_ind_to_keep = torch.randperm(len(self.feat), device=self.feat.device)[num_replace:]
                            kept_old = self.feat[old_ind_to_keep]
                            new_feat_combined = torch.cat([kept_old, new_feat], dim=0).detach().clone()
                        update_length = len(new_feat)
                elif self.updating_policy == "fifo":
                    if len(new_feat) >= self.max_length:
                        new_feat_combined = new_feat[:self.max_length].detach().clone()
                        update_length = self.max_length
                    else:
                        slots_available = self.max_length - len(self.feat)
                        num_replace = len(new_feat) - slots_available
                        if num_replace <= 0:
                            new_feat_combined = torch.cat([self.feat, new_feat], dim=0).detach().clone()
                        else:
                            new_feat_combined = torch.cat([self.feat[num_replace:], new_feat], dim=0).detach().clone()
                        update_length = len(new_feat)
                else:

                    NotImplementedError("Only random updating policy is implemented")
            else:
                new_feat_combined = torch.cat([self.feat, new_feat], dim=0).detach().clone()
                update_length = len(new_feat)


        del self.feat
        self.feat = new_feat_combined
        return len(self.feat), update_length

        #print(f"memory bank update, memory bank size: {len(self.feat)} gpu memory usage: {gpu_mem_usage():.0f}")

    def __len__(self):
        if self.feat is None:
            return 0
        return len(self.feat)

    def len(self):
        return self.__len__()
    # def forward(self, x, x_support=None):
    #     # inference
    #     if x_support is None:
    #         raise NotImplementedError
    #     # training
    #     else:
    #         x = x + self.aggregator(x, x_support)
    #         return x
