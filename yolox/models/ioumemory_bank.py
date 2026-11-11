import torch
import torch.nn as nn
#kssong
#from mmcv.runner import BaseModule
# from ..aggregators.selsa_aggregator import SelsaAggregator
from yolox.utils import bboxes_iou
import gc
def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    return torch.cuda.max_memory_allocated() / (1024 * 1024)

class IoUMemoryBank(nn.Module):
    #def __init__(self,
    #             max_length=20000, key_length=2000,
    #             sampling_policy='random', updating_policy='random',
    #             ):
    def __init__(self,
                 max_length=4800, key_length=480,
                 sampling_policy='random', updating_policy='random',
                 diversity_iou_thresh=0.5, quality_upgrade_margin=0.1,
                 ):
        super().__init__()
        #kssong
        self.max_length = max_length
        self.key_length = key_length
        self.sampling_policy = sampling_policy
        self.updating_policy = updating_policy
        # diversity-based candidate insertion (see post_update_candidate_memory)
        self.diversity_iou_thresh = diversity_iou_thresh
        self.quality_upgrade_margin = quality_upgrade_margin
        #kssong
        self.feat = None
        self.feat_info = None
        self.update_length = 0
        self.candidate_length = 0
        self.candidate_info = None
        self.candidate_feat = None
        # Tensors mirroring feat_info's 'cls_id' / 'bbox' / 'iou_score', kept in sync at
        # every self.feat_info mutation. Used only by the diverse-insertion fast path
        # (_try_insert_candidate / _evict_most_redundant_pair) to avoid O(len(feat_info))
        # Python loops over feat_info on every candidate/eviction.
        self.cls_ids = None
        self.bboxes = None
        self.quality = None

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
        self.cls_ids = None
        self.bboxes = None
        self.quality = None
        return 0, 0

    def _rebuild_parallel_tensors(self):
        """Rebuild cls_ids / bboxes / quality from feat_info. Called only at low-frequency
        points (video init, first-frame refinement) -- the hot path (_try_insert_candidate,
        _append_slot, _remove_slot, _evict_most_redundant_pair) updates these incrementally
        instead of calling this."""
        if self.feat is None or self.feat_info is None or len(self.feat_info) == 0:
            self.cls_ids = None
            self.bboxes = None
            self.quality = None
            return
        device, dtype = self.feat.device, self.feat.dtype
        self.cls_ids = torch.as_tensor(
            [info.get('cls_id', -1) for info in self.feat_info], device=device, dtype=torch.long
        )
        self.bboxes = self._stack_bboxes(self.feat_info, device, dtype)
        self.quality = torch.as_tensor(
            [info.get('iou_score', 0.0) for info in self.feat_info], device=device, dtype=dtype
        )

    def init_memory(self, feat, feat_info):
        """
        init memory
        Args:
            feat: tensor [m, c]

        Returns:

        """
        if self.max_length == 0:
            return 0, 0
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
        self._rebuild_parallel_tensors()
        return len(self.feat), update_length

    def sample(self):
        #kssong
        if self.max_length == 0 or self.key_length == 0:
            return []
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

    def pre_update_candidate_memory(self, new_feat, new_feat_info):
        if self.max_length == 0:
            return 0, 0
        new_feat = new_feat.to('cuda')
        self.candidate_length = len(new_feat)
        self.candidate_feat = new_feat.detach().clone()
        if new_feat_info is not None:
            self.candidate_info = new_feat_info[:]
        else:
            self.candidate_info = None
        return self.candidate_length, (len(self.feat) if self.feat is not None else 0)

    def update(self, new_feat, new_feat_info):
        if self.max_length == 0:
            return 0, 0
        new_feat = new_feat.to('cuda')
        update_length = 0
        new_feat_info_combined = None
        new_feat_combined = None
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
                elif self.updating_policy == "cls_score":
                    #print(f"self.updating_policy: {self.updating_policy}")
                    new_feat_combined = torch.cat([self.feat, new_feat], dim=0).detach().clone()
                    new_feat_info_combined = self.feat_info + new_feat_info
                    update_length = len(new_feat)
                elif self.updating_policy == "iou_score":
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
        if self.feat is not None:
            return len(self.feat), update_length
        else:
            return 0, update_length

        #print(f"memory bank update, memory bank size: {len(self.feat)} gpu memory usage: {gpu_mem_usage():.0f}")

    @staticmethod
    def _unique_bbox_max_conf(result_item):
        """For each unique bbox in result_item, keep the row with the highest
        confidence score (obj_score * cls_score). Returns filtered tensor."""
        if result_item is None or len(result_item) == 0:
            return result_item
        confs = result_item[:, 4] * result_item[:, 5]   # obj_score * cls_score
        bboxes = result_item[:, :4]
        # single device->host transfer instead of one per row
        bboxes_list = bboxes.tolist()
        confs_list = confs.tolist()
        best = {}   # bbox_key -> (max_conf, row_idx)
        for idx, (bbox_vals, conf) in enumerate(zip(bboxes_list, confs_list)):
            key = tuple(bbox_vals)
            if key not in best or conf > best[key][0]:
                best[key] = (conf, idx)
        keep = [v[1] for v in best.values()]
        return result_item[keep]

    @staticmethod
    def _stack_bboxes(feat_info, device, dtype):
        """Stack the 'bbox' entries of a feat_info list into a single [M, 4] tensor."""
        if len(feat_info) == 0:
            return torch.empty((0, 4), device=device, dtype=dtype)
        bbox_list = []
        for info in feat_info:
            b = info['bbox']
            if not isinstance(b, torch.Tensor):
                b = torch.as_tensor(b, dtype=dtype)
            bbox_list.append(b)
        return torch.stack(bbox_list).to(device=device, dtype=dtype)

    @staticmethod
    def _match_bbox_pairs(query_bboxes, ref_bboxes, atol=1.0, rtol=1e-5):
        """Vectorized equivalent of:
            [(i, j) for i, q in enumerate(query_bboxes)
                    for j, r in enumerate(ref_bboxes) if torch.allclose(r, q, atol=atol)]
        Returns a [K, 2] long tensor of (query_idx, ref_idx) pairs.
        """
        if len(query_bboxes) == 0 or len(ref_bboxes) == 0:
            return torch.empty((0, 2), dtype=torch.long, device=query_bboxes.device)
        # torch.allclose(input=ref, other=query) tolerance: atol + rtol * |query|
        diff = (ref_bboxes.unsqueeze(0) - query_bboxes.unsqueeze(1)).abs()  # [n, m, 4]
        tol = atol + rtol * query_bboxes.abs().unsqueeze(1)  # [n, 1, 4]
        match = (diff <= tol).all(dim=-1)  # [n, m]
        return match.nonzero(as_tuple=False)  # [K, 2] -> (query_idx, ref_idx)

    def post_init_memory(self, result):
        if self.max_length == 0:
            return
        if self.feat is not None and self.feat_info is not None:
            for result_item in result:
                result_item = self._unique_bbox_max_conf(result_item)
                if result_item is None or len(result_item) == 0:
                    continue
                bboxes = result_item[:, :4]
                conf_scores = result_item[:, 4] * result_item[:, 5]
                cls_scores = result_item[:, 5]
                iou_scores = result_item[:, 7]
                obj_scores = result_item[:, 4]
                cls_ids = result_item[:, 6]
                ref_bboxes = self._stack_bboxes(self.feat_info, bboxes.device, bboxes.dtype)
                pairs = self._match_bbox_pairs(bboxes, ref_bboxes, atol=1.0)
                if len(pairs):
                    conf_scores_list = conf_scores.tolist()
                    cls_scores_list = cls_scores.tolist()
                    iou_scores_list = iou_scores.tolist()
                    obj_scores_list = obj_scores.tolist()
                    cls_ids_list = cls_ids.tolist()
                    for i, j in pairs.tolist():
                        self.feat_info[j]['conf'] = conf_scores_list[i]
                        self.feat_info[j]['cls_score'] = cls_scores_list[i]
                        self.feat_info[j]['iou_score'] = iou_scores_list[i]
                        self.feat_info[j]['obj_score'] = obj_scores_list[i]
                        self.feat_info[j]['cls_id'] = cls_ids_list[i]
            if self.updating_policy == 'conf':
                if len(self.feat) >= self.max_length:
                    #print(f"updating_policy: {self.updating_policy}")
                    #print(f"self.feat length {len(self.feat)} is larger than max_length {self.max_length}")
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
            elif self.updating_policy == 'cls_score':
                if len(self.feat) >= self.max_length:
                    #print(f"updating_policy: {self.updating_policy}")
                    #print(f"self.feat length {len(self.feat)} is larger than max_length {self.max_length}")
                    sorted_indices = sorted(range(len(self.feat_info)),
                                        key=lambda j: self.feat_info[j].get('cls_score', 0.0),
                                        reverse=True)
                    self.feat_info = [self.feat_info[j] for j in sorted_indices]
                    self.feat = self.feat[sorted_indices]
                    new_feat_combined = self.feat[:self.max_length].detach().clone()
                    new_feat_info_combined = self.feat_info[:self.max_length]
                    del self.feat
                    self.feat = new_feat_combined
                    del self.feat_info
                    self.feat_info = new_feat_info_combined
            elif self.updating_policy == 'iou_score':
                if len(self.feat) >= self.max_length:
                    sorted_indices = sorted(range(len(self.feat_info)),
                                        key=lambda j: self.feat_info[j].get('iou_score', 0.0),
                                        reverse=True)
                    self.feat_info = [self.feat_info[j] for j in sorted_indices]
                    self.feat = self.feat[sorted_indices]
                    new_feat_combined = self.feat[:self.max_length].detach().clone()
                    new_feat_info_combined = self.feat_info[:self.max_length]
                    del self.feat
                    self.feat = new_feat_combined
                    del self.feat_info
                    self.feat_info = new_feat_info_combined
            elif self.updating_policy == 'obj_score':
                if len(self.feat) >= self.max_length:
                    sorted_indices = sorted(range(len(self.feat_info)),
                                        key=lambda j: self.feat_info[j].get('obj_score', 0.0),
                                        reverse=True)
                    self.feat_info = [self.feat_info[j] for j in sorted_indices]
                    self.feat = self.feat[sorted_indices]
                    new_feat_combined = self.feat[:self.max_length].detach().clone()
                    new_feat_info_combined = self.feat_info[:self.max_length]
                    del self.feat
                    self.feat = new_feat_combined
                    del self.feat_info
                    self.feat_info = new_feat_info_combined
            self._rebuild_parallel_tensors()

    def post_update_candidate_memory(self, result):
        """
        Score-refine staged candidates (from pre_update_candidate_memory) against the postprocess
        `result`, then merge the confirmed ones into the permanent memory (self.feat /
        self.feat_info) using a diversity-based insertion + redundancy-eviction policy:
          - a candidate is inserted only if its bbox is spatially novel relative to
            same-class entries already in memory (IoU_max < diversity_iou_thresh);
          - if it instead overlaps an existing same-class slot, that slot is upgraded
            in place when the candidate's predicted IoU quality is clearly higher;
          - when memory is full, room is made by evicting the lower-quality member of
            whichever same-class pair in memory currently has the highest mutual IoU.
        """
        if self.max_length == 0:
            return 0
        if self.candidate_feat is not None and self.candidate_info is not None:
            for result_item in result:
                result_item = self._unique_bbox_max_conf(result_item)
                if result_item is None or len(result_item) == 0:
                    continue
                bboxes = result_item[:, :4]
                conf_scores = result_item[:, 4] * result_item[:, 5]
                cls_scores = result_item[:, 5]
                iou_scores = result_item[:, 7]
                obj_scores = result_item[:, 4]
                cls_ids = result_item[:, 6]
                ref_bboxes = self._stack_bboxes(self.candidate_info, bboxes.device, bboxes.dtype)
                pairs = self._match_bbox_pairs(bboxes, ref_bboxes, atol=1.0)
                if len(pairs):
                    conf_scores_list = conf_scores.tolist()
                    cls_scores_list = cls_scores.tolist()
                    iou_scores_list = iou_scores.tolist()
                    obj_scores_list = obj_scores.tolist()
                    cls_ids_list = cls_ids.tolist()
                    for i, j in pairs.tolist():
                        self.candidate_info[j]['conf'] = conf_scores_list[i]
                        self.candidate_info[j]['cls_score'] = cls_scores_list[i]
                        self.candidate_info[j]['iou_score'] = iou_scores_list[i]
                        self.candidate_info[j]['obj_score'] = obj_scores_list[i]
                        self.candidate_info[j]['cls_id'] = cls_ids_list[i]

            # Only candidates confirmed by postprocess (i.e. matched above, so they
            # carry a 'cls_id') are eligible for insertion into permanent memory.
            for cand_idx in range(len(self.candidate_feat)):
                cand_info = self.candidate_info[cand_idx]
                if 'cls_id' not in cand_info:
                    continue
                self._try_insert_candidate(self.candidate_feat[cand_idx], cand_info)

        if self.candidate_feat is None:
            return 0

        return len(self.candidate_feat)

    def _try_insert_candidate(self, cand_feat, cand_info):
        """Apply the Goal-1/Goal-2 diversity-insertion policy for a single candidate."""
        device, dtype = cand_feat.device, cand_feat.dtype
        cand_bbox = cand_info['bbox']
        if not isinstance(cand_bbox, torch.Tensor):
            cand_bbox = torch.as_tensor(cand_bbox, dtype=dtype)
        cand_bbox = cand_bbox.to(device=device, dtype=dtype).view(1, 4)
        cand_cls = cand_info['cls_id']
        cand_q = cand_info.get('iou_score', 0.0)

        if self.feat is None or len(self.feat) == 0:
            # Empty memory: IoU_max is 0 by definition, so accept unconditionally.
            self._append_slot(cand_feat, cand_info)
            return

        # Vectorized equivalent of scanning feat_info in Python for same-class slots:
        # cls_ids/bboxes are tensors kept in sync with feat_info (see _append_slot,
        # _remove_slot, _rebuild_parallel_tensors), so this is a single masked index
        # instead of an O(len(feat_info)) Python loop + per-call tensor rebuild.
        same_mask = self.cls_ids == cand_cls
        if not same_mask.any():
            iou_max, best_j = 0.0, None
        else:
            same_cls_bboxes = self.bboxes[same_mask].to(device=device, dtype=dtype)
            ious = bboxes_iou(cand_bbox, same_cls_bboxes, xyxy=True)[0]
            best_iou, best_local = torch.max(ious, dim=0)
            same_cls_idx = same_mask.nonzero(as_tuple=True)[0]
            best_j = same_cls_idx[best_local].item()
            iou_max = best_iou.item()

        if best_j is None or iou_max < self.diversity_iou_thresh:
            # Goal 1: spatially novel candidate -> insert (evicting if memory is full).
            if len(self.feat) >= self.max_length:
                if not self._evict_most_redundant_pair():
                    return  # memory full and no redundant same-class pair to free up
            self._append_slot(cand_feat, cand_info)
        else:
            # Redundant with an existing slot: upgrade it in place if the candidate is
            # clearly higher quality, otherwise drop the candidate.
            best_q = self.quality[best_j].item()
            if cand_q > best_q + self.quality_upgrade_margin:
                self.feat[best_j] = cand_feat.detach().clone()
                self.feat_info[best_j] = cand_info
                self.bboxes[best_j] = cand_bbox.view(4).to(self.bboxes.dtype)
                self.quality[best_j] = cand_q
                # cls_ids[best_j] unchanged: best_j was matched by same class already.

    def _append_slot(self, feat_row, info):
        feat_row = feat_row.detach().clone().to('cuda').view(1, -1)
        cls_val = info.get('cls_id', -1)
        q_val = info.get('iou_score', 0.0)
        bbox_val = info['bbox']
        if not isinstance(bbox_val, torch.Tensor):
            bbox_val = torch.as_tensor(bbox_val)
        bbox_val = bbox_val.to(device=feat_row.device, dtype=feat_row.dtype).view(1, 4)
        if self.feat is None:
            self.feat = feat_row
            self.feat_info = [info]
            self.cls_ids = torch.as_tensor([cls_val], device=feat_row.device, dtype=torch.long)
            self.bboxes = bbox_val
            self.quality = torch.as_tensor([q_val], device=feat_row.device, dtype=feat_row.dtype)
        else:
            self.feat = torch.cat([self.feat, feat_row], dim=0)
            self.feat_info.append(info)
            self.cls_ids = torch.cat([
                self.cls_ids, torch.as_tensor([cls_val], device=self.cls_ids.device, dtype=torch.long)
            ])
            self.bboxes = torch.cat([self.bboxes, bbox_val.to(self.bboxes.dtype)], dim=0)
            self.quality = torch.cat([
                self.quality, torch.as_tensor([q_val], device=self.quality.device, dtype=self.quality.dtype)
            ])

    def _remove_slot(self, idx):
        keep_mask = torch.ones(len(self.feat), dtype=torch.bool, device=self.feat.device)
        keep_mask[idx] = False
        self.feat = self.feat[keep_mask]
        del self.feat_info[idx]
        self.cls_ids = self.cls_ids[keep_mask]
        self.bboxes = self.bboxes[keep_mask]
        self.quality = self.quality[keep_mask]

    def _evict_most_redundant_pair(self):
        """Goal 2: find the same-class pair anywhere in memory with the highest mutual
        spatial IoU and evict whichever of the two has the lower predicted IoU quality,
        freeing one slot. Returns False if no such same-class pair exists.

        Vectorized equivalent of the original per-class-bucket loop: computing one global
        [N, N] IoU matrix masked to same-class, off-diagonal pairs and taking its argmax
        gives the same (highest cross-bucket IoU) pair the original found by scanning each
        class bucket's local argmax and keeping the best across buckets.
        """
        n = len(self.feat)
        if n < 2:
            return False
        same_cls = self.cls_ids.unsqueeze(0) == self.cls_ids.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=same_cls.device)
        mask = same_cls & ~eye
        if not mask.any():
            return False

        ious = bboxes_iou(self.bboxes, self.bboxes, xyxy=True)
        ious_masked = ious.masked_fill(~mask, -1.0)
        flat_idx = torch.argmax(ious_masked).item()
        a_j, b_j = divmod(flat_idx, n)

        q_a = self.quality[a_j].item()
        q_b = self.quality[b_j].item()
        evict_j = a_j if q_a <= q_b else b_j
        self._remove_slot(evict_j)
        return True
    def post_update_memory(self, result):
        if self.max_length == 0:
            return 0
        if self.feat is not None and self.feat_info is not None:
            for result_item in result:
                result_item = self._unique_bbox_max_conf(result_item)
                if result_item is None or len(result_item) == 0:
                    continue
                bboxes = result_item[:, :4]
                conf_scores = result_item[:, 4] * result_item[:, 5]
                iou_scores = result_item[:, 7]
                cls_scores = result_item[:, 5]
                obj_scores = result_item[:, 4]
                cls_ids = result_item[:, 6]
                old_end = len(self.feat_info) - self.update_length if self.update_length > 0 else len(self.feat_info)
                new_feat_info = self.feat_info[old_end:]
                ref_bboxes = self._stack_bboxes(new_feat_info, bboxes.device, bboxes.dtype)
                pairs = self._match_bbox_pairs(bboxes, ref_bboxes, atol=1.0)
                if len(pairs):
                    conf_scores_list = conf_scores.tolist()
                    iou_scores_list = iou_scores.tolist()
                    cls_scores_list = cls_scores.tolist()
                    obj_scores_list = obj_scores.tolist()
                    cls_ids_list = cls_ids.tolist()
                    for i, j in pairs.tolist():
                        info = new_feat_info[j]
                        info['conf'] = conf_scores_list[i]
                        info['iou_score'] = iou_scores_list[i]
                        info['cls_score'] = cls_scores_list[i]
                        info['obj_score'] = obj_scores_list[i]
                        info['cls_id'] = cls_ids_list[i]
            if self.updating_policy == 'conf':
                if len(self.feat) >= self.max_length:
                    #print(f"updating_policy: {self.updating_policy}")
                    #print(f"self.feat length {len(self.feat)} is larger than max_length {self.max_length}")
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
            elif self.updating_policy == 'cls_score':
                if len(self.feat) >= self.max_length:
                    #print(f"updating_policy: {self.updating_policy}")
                    #print(f"self.feat length {len(self.feat)} is larger than max_length {self.max_length}")
                    sorted_indices = sorted(range(len(self.feat_info)),
                                        key=lambda j: self.feat_info[j].get('cls_score', 0.0),
                                        reverse=True)
                    self.feat_info = [self.feat_info[j] for j in sorted_indices]
                    self.feat = self.feat[sorted_indices]
                    new_feat_combined = self.feat[:self.max_length].detach().clone()
                    new_feat_info_combined = self.feat_info[:self.max_length]
                    del self.feat
                    self.feat = new_feat_combined
                    del self.feat_info
                    self.feat_info = new_feat_info_combined
            elif self.updating_policy == 'iou_score':
                if len(self.feat) >= self.max_length:
                    #print(f"updating_policy: {self.updating_policy}")
                    #print(f"self.feat length {len(self.feat)} is larger than max_length {self.max_length}")
                    sorted_indices = sorted(range(len(self.feat_info)),
                                        key=lambda j: self.feat_info[j].get('iou_score', 0.0),
                                        reverse=True)
                    self.feat_info = [self.feat_info[j] for j in sorted_indices]
                    self.feat = self.feat[sorted_indices]
                    new_feat_combined = self.feat[:self.max_length].detach().clone()
                    new_feat_info_combined = self.feat_info[:self.max_length]
                    del self.feat
                    self.feat = new_feat_combined
                    del self.feat_info
                    self.feat_info = new_feat_info_combined
            elif self.updating_policy == 'obj_score':
                if len(self.feat) >= self.max_length:
                    #print(f"updating_policy: {self.updating_policy}")
                    #print(f"self.feat length {len(self.feat)} is larger than max_length {self.max_length}")
                    sorted_indices = sorted(range(len(self.feat_info)),
                                        key=lambda j: self.feat_info[j].get('obj_score', 0.0),
                                        reverse=True)
                    self.feat_info = [self.feat_info[j] for j in sorted_indices]
                    self.feat = self.feat[sorted_indices]
                    new_feat_combined = self.feat[:self.max_length].detach().clone()
                    new_feat_info_combined = self.feat_info[:self.max_length]
                    del self.feat
                    self.feat = new_feat_combined
                    del self.feat_info
                    self.feat_info = new_feat_info_combined
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
