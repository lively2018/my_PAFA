#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "diou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            center_distance = torch.sum((pred[:, :2] - target[:, :2]) ** 2, dim=1)
            diagonal_length = torch.sum((c_br - c_tl) ** 2, dim=1) + 1e-16
            diou = iou - center_distance / diagonal_length
            loss = 1 - diou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "ciou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            center_distance = torch.sum((pred[:, :2] - target[:, :2]) ** 2, dim=1)
            diagonal_length = torch.sum((c_br - c_tl) ** 2, dim=1) + 1e-16

            v = (
                (torch.atan(target[:, 2] / target[:, 3]) - torch.atan(pred[:, 2] / pred[:, 3]))
                * 2
                / torch.pi
            ) ** 2
            with torch.no_grad():
                alpha = v / (1 - iou + v)

            ciou = iou - (center_distance / diagonal_length + alpha * v)
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "siou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            center_distance = torch.sum((pred[:, :2] - target[:, :2]) ** 2, dim=1)
            diagonal_length = torch.sum((c_br - c_tl) ** 2, dim=1) + 1e-16

            sigma = torch.sqrt(center_distance) / torch.sqrt(diagonal_length)
            alpha = sigma / (sigma + iou - sigma * iou + 1e-16)

            siou = iou - (center_distance / diagonal_length + alpha * sigma)
            loss = 1 - siou.clamp(min=-1.0, max=1.0)


        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
