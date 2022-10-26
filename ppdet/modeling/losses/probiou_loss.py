# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle
import paddle.nn.functional as F

from ppdet.core.workspace import register, serializable

__all__ = ['ProbIoULoss']


def gbb_form(boxes):
    x1, y1, x2, y2 = paddle.split(boxes, [1, 1, 1, 1], axis=-1)

    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0

    w = paddle.abs(x2 - x1)
    h = paddle.abs(y2 - y1)

    return paddle.concat([x, y, w.pow(2) / 12, h.pow(2) / 12], axis=-1)



def probiou_loss(pred, target, eps=1e-3, mode='l1'):
    """
        ***  pred    -> a matrix [N,4](x1,y1,x2,y2) containing ours predicted box
        ***  target  -> a matrix [N,4](x1,y1,x2,y2) containing ours target    box
        eps     -> threshold to avoid infinite values
        mode    -> ('l1' in [0,1] or 'l2' in [0,inf]) metrics according our paper
    """
    gbboxes1 = gbb_form(pred)
    gbboxes2 = gbb_form(target)

    x1, y1, a1, b1 = gbboxes1[:,0], gbboxes1[:,1], gbboxes1[:,2], gbboxes1[:,3]
    x2, y2, a2, b2 = gbboxes2[:,0], gbboxes2[:,1], gbboxes2[:,2], gbboxes2[:,3]
    c1 = paddle.zeros_like(a1)
    c2 = paddle.zeros_like(a2)                                                     

    t1 = 0.25 * (
                (a1 + a2) * (paddle.pow(y1 - y2, 2)) 
            +   (b1 + b2) * (paddle.pow(x1 - x2, 2))
        ) + 0.5 * ((c1+c2)*(x2-x1)*(y1-y2))

    t2 = (a1 + a2) * (b1 + b2) - paddle.pow(c1 + c2, 2)
    
    t3_ = (a1 * b1 - c1 * c1) * (a2 * b2 - c2 * c2)

    t3 = 0.5 * paddle.log(t2 / (4 * paddle.sqrt(F.relu(t3_)) + eps))

    B_1 = t1 / t2
    B_2 = t3

    B_d = B_1 + B_2
    B_d = paddle.clip(B_d, min=eps, max=100.0)

    l1 = paddle.sqrt(1.0 - paddle.exp(-B_d) + eps)

    l_i = paddle.pow(l1, 2.0)
    l2 = -paddle.log(1.0 - l_i + eps)

    if mode == 'l1':
        probiou = l1.unsqueeze(1)
    if mode == 'l2':
        probiou = l2.unsqueeze(1)

    return probiou

@serializable
@register
class ProbIoULoss(object):
    """ ProbIoU Loss, refer to https://arxiv.org/abs/2106.06072 for details """

    def __init__(self, mode='l1', eps=1e-3):
        super(ProbIoULoss, self).__init__()
        self.mode = mode
        self.eps = eps

    def __call__(self, pred_rboxes, assigned_rboxes):
        return probiou_loss(pred_rboxes, assigned_rboxes, self.eps, self.mode)
