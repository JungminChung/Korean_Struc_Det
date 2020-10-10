# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _transpose_and_gather_feat
import torch.nn.functional as F


def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt, smoothing=0):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  max_val = 1
  min_val = 0

  if smoothing > 0:
    conf = 1 - smoothing
    num_cls = gt.shape[1]

    max_val = conf
    min_val = smoothing / (num_cls - 1)

    gt = torch.where(gt > 0, torch.mul(gt, conf), torch.mul(torch.ones_like(gt), min_val))

  pos_inds = gt.eq(max_val).float()
  neg_inds = gt.lt(max_val).float()

  neg_weights = torch.pow(max_val - gt, 4)

  loss = 0
  pos_loss = torch.log(torch.clamp(1 - torch.abs(max_val - pred), min=1e-5)) \
              * torch.pow(torch.abs(max_val - pred), 2) \
              * pos_inds 
  neg_loss = torch.log(torch.clamp(1 - torch.abs(pred - min_val), min=1e-5)) \
              * torch.pow(torch.abs(pred - min_val), 2) \
              * neg_weights * neg_inds 

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self, opt):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss
    self.smoothing = opt.smoothing

  def forward(self, out, target):
    return self.neg_loss(out, target, smoothing=self.smoothing)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _transpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

class IoULoss(nn.Module):
    def __init__(self, method, opt):
        super(IoULoss, self).__init__()
        self.method = method

    def forward(self, output_wh, output_reg, target_reg_mask, target_ind, target_wh, target_reg):
        height, width = output_wh.size(2), output_wh.size(3)

        xs = torch.remainder(target_ind, width) 
        ys = target_ind // width 

        output_wh = _transpose_and_gather_feat(output_wh, target_ind)  
        output_reg = _transpose_and_gather_feat(output_reg, target_ind)  

        output_bboxes = torch.cat([(xs + output_reg[..., 0] - output_wh[..., 0] / 2).unsqueeze(-1),
                                   (ys + output_reg[..., 1] - output_wh[..., 1] / 2).unsqueeze(-1),
                                   (xs + output_reg[..., 0] + output_wh[..., 0] / 2).unsqueeze(-1),
                                   (ys + output_reg[..., 1] + output_wh[..., 1] / 2).unsqueeze(-1)], dim=2)
        output_bboxes = output_bboxes[target_reg_mask.bool()] 

        target_bboxes = torch.cat([(xs + target_reg[..., 0] - target_wh[..., 0] / 2).unsqueeze(-1),
                                   (ys + target_reg[..., 1] - target_wh[..., 1] / 2).unsqueeze(-1),
                                   (xs + target_reg[..., 0] + target_wh[..., 0] / 2).unsqueeze(-1),
                                   (ys + target_reg[..., 1] + target_wh[..., 1] / 2).unsqueeze(-1)], dim=2)
        target_bboxes = target_bboxes[target_reg_mask.bool()]  

        loss = self.compute_loss(output_bboxes, target_bboxes) / (target_reg_mask.sum() + 1e-4)
        return loss

    def compute_loss(self, output, target):
        x1, y1, x2, y2 = self.transform_bbox(output)
        x1g, y1g, x2g, y2g = self.transform_bbox(target)

        area_output = (x2 - x1) * (y2 - y1)
        area_target = (x2g - x1g) * (y2g - y1g)

        x1i = torch.max(x1, x1g)
        x2i = torch.min(x2, x2g)
        y1i = torch.max(y1, y1g)
        y2i = torch.min(y2, y2g)

        area_inter = torch.zeros_like(x1i)
        inter_mask = (x2i > x1i) * (y2i > y1i)
        area_inter[inter_mask] = (x2i[inter_mask] - x1i[inter_mask]) * (y2i[inter_mask] - y1i[inter_mask])

        area_union = area_output + area_target - area_inter + 1e-7
        iou = area_inter / area_union
        base = iou

        if self.method == 'giou':
            x1c = torch.min(x1, x1g)
            x2c = torch.max(x2, x2g)
            y1c = torch.min(y1, y1g)
            y2c = torch.max(y2, y2g)

            area_c = (x2c - x1c) * (y2c - y1c) + 1e-7

            giou = iou - (torch.abs(area_c - area_union) / torch.abs(area_c))
            base = giou

        loss = torch.ones_like(base) - base

        return loss.sum()

    def transform_bbox(self, bbox):
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        x1, y1, x2, y2 = x1.view(-1), y1.view(-1), x2.view(-1), y2.view(-1)
        return x1, y1, x2, y2