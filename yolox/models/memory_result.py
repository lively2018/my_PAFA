from loguru import logger
import torch
import torch.nn as nn
from yolox.data.datasets.vid_classes import VID_classes
#kssong
#from mmcv.runner import BaseModule
# from ..aggregators.selsa_aggregator import SelsaAggregator

class MemoryResult(nn.Module):
    #def __init__(self,
    #             max_length=20000, key_length=2000,
    #             sampling_policy='random', updating_policy='random',
    #             ):
    def __init__(self,
                 max_length=4800,
                 ):
        super().__init__()
        #kssong
        self.max_length = max_length
        self.update_memory_result_policy = 'age'
        self.results = []
        self.age = 0

    def get_memory_result_info(self):
        return self.results

    def reset_memory_result(self):
        #kssong
        if self.results is not None:
            self.results.clear()  # Clear the list to free memory
        self.age = 0
        self.results = []
        logger.info(f"reset_memory_result, memory result size: {len(self.results)}")

    def init_memory_result(self, result):
        """
        init memory result
        Args:
            result_info: list of tensors

        Returns:

        """

        logger.info(f"init_memory_result, len(result): {len(result)}")
        for result_item in result:
            #logger.info(f"result_item: {result_item}")
            self.results.append(result_item)
            self.age += 1


    def update_memory_result(self, new_result):
        logger.info(f"update_memory_result, len(new_result): {len(new_result)}, current memory result size: {len(self.results)}")
        if len(self.results) < self.max_length:
            for result_item in new_result:
                #logger.info(f"result_item: {result_item}")
                self.results.append(result_item)
                self.age += 1

        elif self.updating_policy == "age":
            for result_item in new_result:
                logger.info(f"result_item: {result_item}")
                self.results.pop(0)
                self.results.append(result_item)
                self.age += 1
        else:
            raise NotImplementedError("not implemented")

    def __len__(self):
        return len(self.results)