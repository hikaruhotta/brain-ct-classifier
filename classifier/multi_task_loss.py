import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Wrapper for multi-task loss functions.
    Has learnable lambda parameters as described in https://arxiv.org/abs/1705.07115
    """
    def __init__(self, args):
        super().__init__()
        self.task_num = len(args.features)
        self.device = args.device
        self.eps = 1e-8
        if hasattr(args, 'focal_loss_alphas'):
            self.alphas = args.focal_loss_alphas
        
    def focal_loss(self, labels, logits, alpha, gamma=1.0):
        """
        Compute the focal loss between `logits` and the ground truth `labels`.
        """
        BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")
        weights = torch.zeros_like(BCLoss)
        weights += labels * alpha
        weights += -(labels - 1) * (1 - alpha)
        loss = BCLoss * weights
        loss = torch.sum(loss)
        return loss

    def forward(self, preds, targets):
        losses = [nn.BCEWithLogitsLoss() for i in range(self.task_num)]
        if hasattr(self, 'alphas'):
            losses = [self.focal_loss for i in range(self.task_num)]
        
        total_loss = 0
        for i in range(self.task_num):
            loss_fn = losses[i]
            loss = loss_fn(torch.unsqueeze(targets[:,i], -1), preds[i], self.alphas[i])
            total_loss += loss
        
        return total_loss