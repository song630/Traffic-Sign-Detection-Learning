from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din): # din: depth of input(深度可能是有多少个feature map)
        super(_RPN, self).__init__() # 初始化nn.Module
        
        # RPN网络的输入是1024个feature map
        # 在经过头部的rpn_net卷积层后变为512个
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES # [8, 16, 32]
        self.anchor_ratios = cfg.ANCHOR_RATIOS # [0.5, 1, 2]
        self.feat_stride = cfg.FEAT_STRIDE[0] # 卷积的stride

        # define the convrelu layers processing input feature map
        # 头部的卷积层 卷积核大小为3
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classification score layer
        # 9种anchor * 前景/背景 = 18
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        # 一个分支卷积层 卷积核大小为1
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        # 9种anchor * 4个回归系数 = 36
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        # 另一个分支卷积层 卷积核大小为1
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        """
        ===
        region-proposal-layer对应3层:
        proposal layer
        anchor target layer
        proposal target layer
        下面的代码建立了其中的前2层(再细看)
        """
        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # 和RPN网络的2个loss function有关
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0) # feature map个数??

        # return feature map after convrelu layer
        # RPN头部的卷积层RPN_Conv(输入是base_feat)经过relu输出
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        # 一个分支卷积层 算出得分(输出的是tensor)
        # =====
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        # 刚从分支出来时维度是w/16, h/16, 9, 2(前景/背景)
        # 之后要reshape成w/16 * h/16 * 9, 2来算class scores
        # 最后经过softmax -> class概率
        # 结构见pdf-page10
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)
        # 又reshape了一次产生输出 原pdf图上没有画
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        # ===== 这个分支结束
        # get rpn offsets to the anchor boxes
        # 另一个卷积分支
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)
        # =====

        cfg_key = 'TRAIN' if self.training else 'TEST' # 训练或测试

        # proposal layer(第1层)
        # === 任务是: 用回归系数来变换这些anchor并使用NMS来过滤
        # 经过这一层后 输出ROI scores, ROIs
        # === 再看这一层网络的另一个文件
        # === 对应教程第11页图
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        # 和RPN网络的2个loss function有关
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training: # 如果是训练阶段
            assert gt_boxes is not None

            # 3层中第2层
            # 输出: 较好的bbox和对应的label + 目标回归系数
            # === 看另一个文件
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # 计算RPN loss: 分两部分
            # 1.compute classification loss 预测的和实际类的交叉熵
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            # 与bbox对应的labels
            rpn_label = rpn_data[0].view(batch_size, -1)
            # 下面几行: 选出了有意义的那部分label以及分类的得分 (=== 没有细看)
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            # classification loss是交叉熵
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            # 计数: 属于前景的box
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            # inside_weights, outside_weights好像是2个mask
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # 2.compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)
            # 调用smooth函数(具体公式参照教程第12页)(=== 没有细看)
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
        # proposal target layer输出: 筛选过的ROIs和"类特定"的回归系数 (=== 不确定)
        return rois, self.rpn_loss_cls, self.rpn_loss_box
