import torch
import torch.nn as nn

from mmdet3d.models.builder import LOSSES


@LOSSES.register_module()
class VelocityPlausibilityLoss(nn.Module):

    def __init__(self, class_max_vel, loss_weight=0.1):
        super().__init__()
        self.class_max_vel = class_max_vel
        self.loss_weight = loss_weight

    def forward(self, vel_pred, labels, avg_factor=None):
        if vel_pred.numel() == 0:
            return vel_pred.sum() * 0.0

        speed = vel_pred.norm(dim=-1)
        loss = torch.zeros_like(speed)

        for cls_idx, max_vel in self.class_max_vel.items():
            mask = labels == cls_idx
            if mask.any():
                excess = (speed[mask] - max_vel).clamp(min=0.0)
                loss[mask] = excess ** 2

        if avg_factor is not None:
            return self.loss_weight * loss.sum() / avg_factor
        return self.loss_weight * loss.mean()