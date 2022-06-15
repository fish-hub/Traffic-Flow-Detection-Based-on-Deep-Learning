#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        #conv用的都是baseconv，只是提特征
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        #各个head，用来预测定位目标位置+类别
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        #stem也是提取特征的baseconv
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv


        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            #分类头，输出深度为特征图每个点预测anchor数量*class数量，这里为1*1=1
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            #回归头，输出深度为4，代表特征图每个点预测目标的位置
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            #前背景分类头，输出深度为1，代表特征图每个点预测的anchor是前景or背景的概率
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # xin([1,128,80,80],[1,256,40,40],[1,512,20,20])
    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x) #pafpn输出层n用上第n个baseconv
            cls_x = x
            reg_x = x

            #得到分类输出
            cls_feat = cls_conv(cls_x)#因为这个是遍历拿到的子元素，所以不用[k]
            cls_output = self.cls_preds[k](cls_feat)#Bx1x80x80(40x40,20x20)
            #得到回归输出和前背景分类输出
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)#Bx4x80x80(40x40,20x20)
            obj_output = self.obj_preds[k](reg_feat)#Bx1x80x80(40x40,20x20)

            if self.training:
                # 拼接输出Bx6x80x80
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                # 拿到映射回原图的预测anchor和对应grid标号
                # output Bx6400x6 grid 1x6400x2(栅格x坐标和y坐标信息)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )

                # 分离出x坐标和y坐标向量
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                # 1x6400，为每个x/y坐标返回回原图的stride
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1: # anchor回归默认不用L1
                    batch_size = reg_output.shape[0]#
                    hsize, wsize = reg_output.shape[-2:]#80x80
                    reg_output = reg_output.view( #bx1x4x80x80
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    #bx1x80x80x4->bx(1*80*80)x4
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                #验证模式，拼接头返回不计算loss，返回Bx(4+1+1)x80x80
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
                #print("output.shape eval", output.shape)
            #不同尺度输出放在一个list里，3个元素
            # 训练模式:Bx6400x6..Bx1600x6..Bx400x6
            # 验证模式:Bx6x80x80..Bx6x40x40..Bx6x20x20
            outputs.append(output)
            #print(outputs[0].shape)
        if self.training:
            #如果是训练模式，计算损失值
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:# 验证模式
            # 拿到三个尺度特征图的锚点数目[[80,80],[40.40],[20,20]]
            self.hw = [x.shape[-2:] for x in outputs]
            # 拼接三个head的输出为[B, 8500, 6]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())#进行目标位置修正到原始图像
            else:
                return outputs

    # 1x6x(80x80),0/1/2,8/16/32,FloatTensor
    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]# [tensor(0),tensor(0),tensor(0)]

        batch_size = output.shape[0]
        # 输出channel个数为类别数加5
        n_ch = 5 + self.num_classes
        # 输出特征图的尺寸80x80
        hsize, wsize = output.shape[-2:]
        # 初始化grid中三个tensor
        if grid.shape[2:4] != output.shape[2:4]:
            # yv:80x80-->[[0,0,0...],[1,1,1....]],xv:80x80-->[[0,1,2,3...],[0,1,2,3...]]
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # stack->沿一个新维度对输入张量序列进行连接
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid # 1x1x80x80x2

        # output->8x1x6x80x80
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        #8x1x80x80x6->8x(1*80*80)x6
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        #1x(1*80*80)x2
        grid = grid.view(1, -1, 2)
        #取出来6中的前两个得到它在原图中对应位置，6->(x,y,w,h,obj,class)
        output[..., :2] = (output[..., :2] + grid) * stride
        #取出来w和h得到预测锚框的宽和高
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        #返回修正过锚框的输出和每个grid排列的index（修正包括anchor位置和宽高修正到原始图像大小上，grid为80x80对应的行列标号）
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []#保存每个特征图的栅格坐标
        strides = []#对于每个特征图的栅格坐标的stride
        #(80 80) 8,(40 40) 16,(20 20) 32
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)#[0,0],[0,1]...[0,79],[1,0]->[79,79]
            grids.append(grid)
            shape = grid.shape[:2]#1x6400(1600,400)
            strides.append(torch.full((*shape, 1), stride))#1x6400x1

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides #修正得到目标在原图的xy坐标
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides#修正得到目标在原图的wh坐标
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,#栅格x坐标
        y_shifts,#栅格y坐标
        expanded_strides,#栅格对应的stride
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        # 预测框位置信息 8x8400x4
        bbox_preds = outputs[:, :, :4]  # [B, n_anchors_all, 4]
        # 前背景预测信息 8x8400x1
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        # 类别预测信息 8x8400x1
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # 每个图片中GT的数目 8
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        # 预测anchor个数8400
        total_num_anchors = outputs.shape[1]
        # 三个特征图各锚点x，y坐标拼接起来，1x8400
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        # 三个特征图各锚点stride拼接起来，1x8400
        expanded_strides = torch.cat(expanded_strides, 1)# [1, n_anchors_all]
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0
        # 遍历batch中每一个图像
        for batch_idx in range(outputs.shape[0]):
            # 当前图像的标注框数目
            num_gt = int(nlabel[batch_idx])
            # 当前batch中总标注框数目
            num_gts += num_gt

            # 如果当前图片不包含标注框，所有label全是0
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            # 如果当前图片包含标注框，分配各分支label
            else:
                # 取出当前图片的边界框标注坐标 Nx4
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                # 取出当前图片边界框类别信息 N
                gt_classes = labels[batch_idx, :num_gt, 0]
                # 取出当前图片的预测坐标信息 8400x4
                bboxes_preds_per_image = bbox_preds[batch_idx]

                # simota正负样本分配
                try:
                    (
                        gt_matched_classes,  # X 精细化筛选后每个检测框匹配GT的类别
                        fg_mask,  # 8400 8400个检测框通过精细化筛选的每个检测框位置为1
                        pred_ious_this_matching,  # X 精细化筛选后每个检测框与GT的IOU
                        matched_gt_inds,  # X 精细化筛选后每个检测框匹配GT的索引
                        num_fg_img,  # 当前批次数据通过精细化筛选的高质量正样本检测框数目
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img
                # Xx1 正样本类别Onehot编码×它与GT的IOU
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                # 8400x1 所有预测框是否为正样本掩膜
                obj_target = fg_mask.unsqueeze(-1)
                # Xx4 每个正样本检测框对应GT的坐标
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg

        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + 3.0*loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
        # fg_mask->8400，通过两种正样本筛选其中一个就保留的检测框True、False矩阵
        # is_in_boxes_and_center->16xM，同时通过两种检测方式的检测框
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        # Nx4 根据正负样本矩阵fg_mask筛选拿到正样本预测框位置
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        # Nx1 根据正负样本矩阵fg_mask筛选拿到正样本预测框类别信息
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        # Nx1 根据正负样本矩阵fg_mask筛选拿到正样本预测框前背景得分
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        # N 正样本框数目
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        # GT_num x Pos_Pred_num，每个GT和预测框的IOU
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        # GT_num x Pos_Pred_num x 1,每个GT的类别重复堆叠Pos_Pred_num次
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        # GT_num x Pos_Pred_num，每个GT和预测框的IOU Loss
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            # GT_num x Pos_Pred_num x 1 SigMoid(分类得分)乘SigMoid(前背景得分)=最终得分
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            # GT_num x Pos_Pred_num 最终得分与类别标签作交叉熵
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_
        # GT_num x Pos_Pred_num Cost代价矩阵计算
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,  # X 精细化筛选后的正样本数目
            gt_matched_classes,  # X 精细化筛选后的正样本匹配到的GT所属类别
            pred_ious_this_matching,  # X 精细化筛选后的正样本与GT的IOU
            matched_gt_inds,  # X 精细化筛选后的正样本匹配到的GT索引
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        # 每个预测框映射回原图的stride 8400
        expanded_strides_per_image = expanded_strides[0]
        # 把每个锚框的初始点×stride映射回原图 8400
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        # 16x8400，16份，每个预测框映射回原图的中心点x坐标
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        # 16x8400，16份，每个预测框映射回原图的中心点y坐标
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        # 16x8400 8400份，每个真实框的左上角x坐标
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        # 16x8400 8400份，每个真实框的右下角x坐标
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        # 16x8400 8400份，每个真实框的左上角y坐标
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        # 16x8400 8400份，每个真实框的右下角y坐标
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        # 16x8400，求出预测框中心点与GT左边线的距离
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        # 16x8400，求出预测框中心点与GT右边线的距离
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        # 16x8400，求出预测框中心点与GT上边线的距离
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        # 16x8400，求出预测框中心点与GT下边线的距离
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        # 16x8400x4，堆叠出每个预测框与所有真实框的真实回归距离（真实，预测框在原图中心点与GT四个边之间的距离）
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        # 16x8400，当前检测框中心点是否在对应的GT内部
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        # 8400，每个检测框被分配到几个GT内
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5
        # 16x8400，8400份真实框中心x坐标-2.5*stride，用于中心点一定区域内正样本，大尺度特征图上预测大目标，区域也会变大点
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        # 16x8400，8400份真实框中心x坐标+2.5*stride，用于中心点一定区域内正样本
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        # 16x8400，8400份真实框中心y坐标-2.5*stride，用于中心点一定区域内正样本
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        # 16x8400，8400份真实框中心y坐标+2.5*stride，用于中心点一定区域内正样本
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        # 16x8400 计算每个预测框中心点与GT中心点一定区域内四个边的距离
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        # 16x8400x4 堆叠起来
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        # 16x8400 判断预测框中心点是否在GT中心点一定区域内
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        # 8400 计算每个检测框匹配到的真实框个数
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # 8400 两种正样本筛选中有一种满足就认为是正样本
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        # 16xn 同时满足在GT内和GT中心一定区域内的检测框
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        # GT_num x Pos_Pred_num 创建全0匹配矩阵
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # GT_num x Pos_Pred_num IOU矩阵
        ious_in_boxes_matrix = pair_wise_ious
        # 10 使用IOU最大的前十个检测框来计算
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        # GT_num x 10 取出来对应GT的最大的十个IOU
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        # 按行累加求出每个GT分配的检测框个数
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            # 对每个GT，根据他分配的检测框数目找出前几个Cost最大的检测框，要索引
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            # GT_num x Pos_Pred_num 匹配上的位置为1
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx
        # 使用Cost过滤共用候选框
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        # 匹配到目标的检测框掩膜矩阵
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        # 匹配到目标的检测框个数
        num_fg = fg_mask_inboxes.sum().item()
        # 8400个检测框中匹配到正样本的检测框为1
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        # 所有正样本检测框匹配到的GT索引矩阵
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
