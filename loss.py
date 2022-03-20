
from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = Variable(targets, requires_grad=False)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class BCELossForDuIE(nn.Module):
    def __init__(self, args):
        super(BCELossForDuIE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')# nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        mask = mask.float()
        loss = loss * mask.unsqueeze(-1)
        loss = torch.sum(torch.mean(loss, dim=2), dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss

class BCELossForDuIE_Smooth(nn.Module):
    def __init__(self, args):
        super(BCELossForDuIE_Smooth, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss(reduction='none')# nn.BCEWithLogitsLoss(reduction='none')
        self.exponent_coef = args.exponent_coef

    def forward(self, logits, labels, mask):
        logits_sigmoid = self.sigmoid(logits)
        logits = logits_sigmoid**self.exponent_coef
        loss = self.criterion(logits, labels)
        mask = mask.float()
        loss = loss * mask.unsqueeze(-1)
        loss = torch.sum(torch.mean(loss, dim=2), dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss


class BCELossForDuIE_Smooth_prob(nn.Module):
    def __init__(self, args):
        super(BCELossForDuIE_Smooth_prob, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')# nn.BCEWithLogitsLoss(reduction='none')
        self.exponent_coef = int(args.exponent_coef)

    def forward(self, logits, labels, mask):
        if self.exponent_coef % 2 == 0:
            with torch.no_grad():
                logits_symbol = torch.where(logits > 0, 1.0, -1.0).cuda()
            logits = logits**self.exponent_coef
            logits = logits*logits_symbol
        else:
            logits = logits**self.exponent_coef
        loss = self.criterion(logits, labels)
        mask = mask.float()
        loss = loss * mask.unsqueeze(-1)
        loss = torch.sum(torch.mean(loss, dim=2), dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss

class BCELossForDuIE_BI(nn.Module):
    def __init__(self, args):
        super(BCELossForDuIE_BI, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')# nn.BCEWithLogitsLoss(reduction='none')
        self.weight_coef = args.weight_coef - 1

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        with torch.no_grad():
            weight_matrix = torch.ones_like(labels)
            weight_matrix[:, :, 1:] += labels[:, :, 1:] * self.weight_coef
        loss *= weight_matrix
        mask = mask.float()
        loss = loss * mask.unsqueeze(-1)
        loss = torch.sum(torch.mean(loss, dim=2), dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss

class BCELossForDuIE_BIO(nn.Module):
    def __init__(self, args):
        super(BCELossForDuIE_BIO, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')# nn.BCEWithLogitsLoss(reduction='none')
        self.weight_coef = args.weight_coef - 1

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        with torch.no_grad():
            weight_matrix = torch.ones_like(labels)
            weight_matrix += labels * self.weight_coef
        loss *= weight_matrix
        mask = mask.float()
        loss = loss * mask.unsqueeze(-1)
        loss = torch.sum(torch.mean(loss, dim=2), dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss
