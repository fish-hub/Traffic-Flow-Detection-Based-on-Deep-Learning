#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    # 遍历初始模型结构得到key和value
    for key_model, v in model_state_dict.items():
        #判断模型结构在ckpt中是否存在，不存在进入下一次循环
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        # 如果模型结构在ckpt中存在则从ckpt中取出key对应的value
        v_ckpt = ckpt[key_model]
        # 如果模型结构的shape和ckpt结构的shape不同，发出警告，直接进行下一次循环
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        # 模型和ckpt结构和shape完全相同时，在load——dict中生成对应的key和value
        load_dict[key_model] = v_ckpt
    # 模型导入预训练参数
    model.load_state_dict(load_dict, strict=False)
    #返回导入参数后的模型
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)
