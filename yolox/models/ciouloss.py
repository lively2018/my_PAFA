import math
import torch
import torch.nn as nn

class CIoULoss(nn.Module):
    def __init__(self, reduction="none", eps=1e-7):
        super(CIoULoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: (Tensor) Predicted bboxes, shape [N, 4] (format: cx, cy, w, h)
            target: (Tensor) Ground truth bboxes, shape [N, 4] (format: cx, cy, w, h)
        """
        # 1. Calculate standard IoU
        # Convert from cxcywh to x1y1x2y2
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        target_x1 = target[:, 0] - target[:, 2] / 2
        target_y1 = target[:, 1] - target[:, 3] / 2
        target_x2 = target[:, 0] + target[:, 2] / 2
        target_y2 = target[:, 1] + target[:, 3] / 2

        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        pred_area = pred[:, 2] * pred[:, 3]
        target_area = target[:, 2] * target[:, 3]
        union = pred_area + target_area - inter_area + self.eps
        iou = inter_area / union

        # 2. Calculate Distance Penalty (DIoU part)
        # Center points
        inter_diag = (pred[:, 0] - target[:, 0])**2 + (pred[:, 1] - target[:, 1])**2
        # Smallest enclosing box (convex box)
        outer_x1 = torch.min(pred_x1, target_x1)
        outer_y1 = torch.min(pred_y1, target_y1)
        outer_x2 = torch.max(pred_x2, target_x2)
        outer_y2 = torch.max(pred_y2, target_y2)
        outer_diag = (outer_x2 - outer_x1)**2 + (outer_y2 - outer_y1)**2 + self.eps
        
        diou_penalty = inter_diag / outer_diag

        # 3. Calculate Aspect Ratio Penalty (v) and Weight (alpha)
        # v = (4 / pi^2) * (arctan(w_gt/h_gt) - arctan(w/h))^2
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target[:, 2] / (target[:, 3] + self.eps)) - 
            torch.atan(pred[:, 2] / (pred[:, 3] + self.eps)), 2
        )
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        # 4. Final CIoU score and loss
        ciou = iou - (diou_penalty + alpha * v)
        loss = 1.0 - ciou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss