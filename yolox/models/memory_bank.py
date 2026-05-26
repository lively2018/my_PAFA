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
        self.feat_info = None
        self.update_lenght = 0

        # self.aggregator = SelsaAggregator(in_channels)

    def reset(self):
        #kssong
        if self.feat is not None:
            del self.feat  # Explicitly delete the tensor
        self.feat = None
        #print(f"reset_memory")
        if self.feat_info is not None:
            del self.feat_info
        self.feat_info = None
        return 0, 0

    def init_memory(self, feat, feat_info):
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
            if feat_info is not None:
                self.feat_info = feat_info[:self.max_length]
            update_length = self.max_length
        else:
            self.feat = feat.detach().clone().to('cuda')
            if feat_info is not None:
                self.feat_info = feat_info[:]
            update_length = len(feat)
        #print(f"init_memory, memory bank max size: {self.max_length}, memory_bank key size: {self.key_length}")
        self.update_length = update_length
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

    def update(self, new_feat, new_feat_info):
        new_feat = new_feat.to('cuda')
        update_length = 0
        new_feat_info_combined = None
        if self.feat is None:
            if len(new_feat) >= self.max_length:
                new_feat_combined = new_feat[:self.max_length].detach().clone()
                if new_feat_info is not None:
                    new_feat_info_combined = new_feat_info[:self.max_length]
                update_length = self.max_length
            else:
                new_feat_combined = new_feat.detach().clone()
                if new_feat_info is not None:
                    new_feat_info_combined = new_feat_info[:]
                update_length = len(new_feat)

        else:
            if len(new_feat) + len(self.feat) >= self.max_length:
                if self.updating_policy == "random":
                    if len(new_feat) >= self.max_length:
                        new_feat_combined = new_feat[:self.max_length].detach().clone()
                        if new_feat_info is not None:
                            new_feat_info_combined = new_feat_info[:self.max_length]
                        update_length = self.max_length
                    else:
                        slots_available = self.max_length - len(self.feat)
                        num_replace = len(new_feat) - slots_available
                        if num_replace <= 0:
                            new_feat_combined = torch.cat([self.feat, new_feat], dim=0).detach().clone()
                            if new_feat_info is not None:
                                new_feat_info_combined = self.feat_info + new_feat_info
                        else:
                            old_ind_to_keep = torch.randperm(len(self.feat), device=self.feat.device)[num_replace:]
                            kept_old = self.feat[old_ind_to_keep]
                            new_feat_combined = torch.cat([kept_old, new_feat], dim=0).detach().clone()
                            if new_feat_info is not None:
                                kept_info = [self.feat_info[i] for i in old_ind_to_keep.tolist()]
                                new_feat_info_combined = kept_info + new_feat_info
                        update_length = len(new_feat)
                elif self.updating_policy == "fifo":
                    if len(new_feat) >= self.max_length:
                        new_feat_combined = new_feat[:self.max_length].detach().clone()
                        if new_feat_info is not None:
                            new_feat_info_combined = new_feat_info[:self.max_length]
                        update_length = self.max_length
                    else:
                        slots_available = self.max_length - len(self.feat)
                        num_replace = len(new_feat) - slots_available
                        if num_replace <= 0:
                            new_feat_combined = torch.cat([self.feat, new_feat], dim=0).detach().clone()
                            if new_feat_info is not None:
                                new_feat_info_combined = self.feat_info + new_feat_info
                        else:
                            new_feat_combined = torch.cat([self.feat[num_replace:], new_feat], dim=0).detach().clone()
                            if new_feat_info is not None:
                                new_feat_info_combined = self.feat_info[num_replace:] + new_feat_info
                        update_length = len(new_feat)
                elif self.updating_policy == "conf":
                    new_feat_combined = torch.cat([self.feat, new_feat], dim=0).detach().clone()
                    new_feat_info_combined = self.feat_info + new_feat_info
                    update_length = len(new_feat)
                else:
                    NotImplementedError("Only random updating policy is implemented")
            else:
                new_feat_combined = torch.cat([self.feat, new_feat], dim=0).detach().clone()
                if new_feat_info is not None:
                    new_feat_info_combined = self.feat_info + new_feat_info
                update_length = len(new_feat)


        del self.feat
        del self.feat_info
        self.feat = new_feat_combined
        self.feat_info = new_feat_info_combined
        self.update_length = update_length
        return len(self.feat), update_length

        #print(f"memory bank update, memory bank size: {len(self.feat)} gpu memory usage: {gpu_mem_usage():.0f}")

    @staticmethod
    def _unique_bbox_max_conf(result_item):
        """For each unique bbox in result_item, keep the row with the highest
        confidence score (obj_score * cls_score). Returns filtered tensor."""
        if result_item is None or len(result_item) == 0:
            return result_item
        confs = result_item[:, 4] * result_item[:, 5]   # obj_score * cls_score
        bboxes = result_item[:, :4]
        best = {}   # bbox_key -> (max_conf, row_idx)
        for idx in range(len(result_item)):
            key = tuple(bboxes[idx].tolist())
            conf = confs[idx].item()
            if key not in best or conf > best[key][0]:
                best[key] = (conf, idx)
        keep = [v[1] for v in best.values()]
        return result_item[keep]

    def post_init_memory(self, result):
        if self.feat is not None and self.feat_info is not None:
            for result_item in result:
                result_item = self._unique_bbox_max_conf(result_item)
                if result_item is None or len(result_item) == 0:
                    continue
                bboxes = result_item[:, :4]
                conf_scores = result_item[:, 4] * result_item[:, 5]
                for i, bbox in enumerate(bboxes):
                    for j, info in enumerate(self.feat_info):
                        info_bbox = info['bbox']
                        if not isinstance(info_bbox, torch.Tensor):
                            info_bbox = torch.tensor(info_bbox, device=bbox.device, dtype=bbox.dtype)
                        else:
                            info_bbox = info_bbox.to(device=bbox.device, dtype=bbox.dtype)
                        if torch.allclose(info_bbox, bbox, atol=1.0):
                            #print(f"info_bbox:{info_bbox} bbox: {bbox}")
                            #print(f"before conf_score: {self.feat_info[j]['conf']}")
                            self.feat_info[j]['conf'] = conf_scores[i].item()
                            #print(f"after conf_score: {self.feat_info[j]['conf']}")
            if self.updating_policy == 'conf':
                if len(self.feat) >= self.max_length:
                    print(f"updating_policy: {self.updating_policy}")
                    print(f"self.feat length {len(self.feat)} is larger than max_length {self.max_length}")
                    sorted_indices = sorted(range(len(self.feat_info)),
                                        key=lambda j: self.feat_info[j].get('conf', 0.0),
                                        reverse=True)
                    self.feat_info = [self.feat_info[j] for j in sorted_indices]
                    self.feat = self.feat[sorted_indices]
                    new_feat_combined = self.feat[:self.max_length].detach().clone()
                    new_feat_info_combined = self.feat_info[:self.max_length]
                    del self.feat
                    self.feat = new_feat_combined
                    del self.feat_info
                    self.feat_info = new_feat_info_combined

    def post_update_memory(self, result):
        if self.feat is not None and self.feat_info is not None:
            for result_item in result:
                result_item = self._unique_bbox_max_conf(result_item)
                if result_item is None or len(result_item) == 0:
                    continue
                bboxes = result_item[:, :4]
                conf_scores = result_item[:, 4] * result_item[:, 5]
                old_end = len(self.feat_info) - self.update_length if self.update_length > 0 else len(self.feat_info)
                for i, bbox in enumerate(bboxes):
                    for info in self.feat_info[old_end:]:
                        info_bbox = info['bbox']
                        if not isinstance(info_bbox, torch.Tensor):
                            info_bbox = torch.tensor(info_bbox, device=bbox.device, dtype=bbox.dtype)
                        else:
                            info_bbox = info_bbox.to(device=bbox.device, dtype=bbox.dtype)
                        if torch.allclose(info_bbox, bbox, atol=1.0):
                            #print(f"before confidence {info['conf']}")
                            info['conf'] = conf_scores[i].item()
                            #print(f"after confidence {info['conf']}")
            if self.updating_policy == 'conf':
                if len(self.feat) >= self.max_length:
                    print(f"updating_policy: {self.updating_policy}")
                    print(f"self.feat length {len(self.feat)} is larger than max_length {self.max_length}")
                    sorted_indices = sorted(range(len(self.feat_info)),
                                        key=lambda j: self.feat_info[j].get('conf', 0.0),
                                        reverse=True)
                    self.feat_info = [self.feat_info[j] for j in sorted_indices]
                    self.feat = self.feat[sorted_indices]
                    new_feat_combined = self.feat[:self.max_length].detach().clone()
                    new_feat_info_combined = self.feat_info[:self.max_length]
                    del self.feat
                    self.feat = new_feat_combined
                    del self.feat_info
                    self.feat_info = new_feat_info_combined
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
