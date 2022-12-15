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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register

from ..bbox_utils import batch_distance2bbox
from ..losses import GIoULoss, SIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ..backbones.yolov6_efficientrep import BaseConv
from ppdet.modeling.ops import get_static_shape, get_act_fn
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['EffiDeHead']


@register
class EffiDeHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'self_distill', 'distill_feat',
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(
            self,
            in_channels=[128, 256, 512],
            num_classes=80,
            act='relu',
            fpn_strides=[8, 16, 32],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            reg_max=16,  # reg_max=0 if use_dfl is False
            use_dfl=False,  # False in n/t/s version, True in m/l version
            static_assigner_epoch=3,  # warmup_epoch
            static_assigner='ATSSAssigner',
            assigner='TaskAlignedAssigner',
            eval_size=None,
            iou_type='giou',  # 'siou' in n/t version
            loss_weight={
                'cls': 1.0,
                'iou': 2.5,
                'dfl': 0.5,  # used in m/l version 
                'cwd': 10.0,  # used when self_distill=True, in m/l version
            },
            nms='MultiClassNMS',
            trt=False,
            exclude_nms=False,
            exclude_post_process=False,
            self_distill=False,
            distill_feat=False,
            distill_weight={
                'cls': 1.0,
                'dfl': 1.0,
            },
            distill_temperature=20,
            print_l1_loss=True):




        super(EffiDeHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.eval_size = eval_size
        self.iou_loss = GIoULoss()
        assert iou_type in ['giou', 'siou'], "only support giou and siou loss."
        if iou_type == 'siou':
            self.iou_loss = SIoULoss()
        self.loss_weight = loss_weight

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.print_l1_loss = print_l1_loss

        # for self-distillation
        self.self_distill = self_distill
        self.distill_feat = distill_feat
        self.distill_weight = distill_weight
        self.distill_temperature = distill_temperature

        # stem
        self.stem_conv = nn.LayerList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_conv.append(BaseConv(in_c, in_c, 1, 1))

        # pred head
        self.pred_cls = nn.LayerList()
        self.pred_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Sequential(* [
                    BaseConv(in_c, in_c, 3, 1), nn.Conv2D(
                        in_c,
                        self.num_classes,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
                ]))
            self.pred_reg.append(
                nn.Sequential(* [
                    BaseConv(in_c, in_c, 3, 1), nn.Conv2D(
                        in_c,
                        4 * (self.reg_max + 1),
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
                ]))
        # projection conv
        self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True
        self._init_weights()
        self.print_l1_loss = print_l1_loss

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            constant_(cls_[-1].weight)
            constant_(cls_[-1].bias, bias_cls)
            constant_(reg_[-1].weight)
            constant_(reg_[-1].bias, 1.0)

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor


    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            feat = self.stem_conv[i](feat)
            cls_logit = self.pred_cls[i](feat)
            reg_distri = self.pred_reg[i](feat)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)



    def forward_train_with_distill(self, feats, targets, t_outputs, epoch_num, max_epoch):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            feat = self.stem_conv[i](feat)
            cls_logit = self.pred_cls[i](feat)
            reg_distri = self.pred_reg[i](feat)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        return self.get_loss_with_distill([
            feats, cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor, 
        ], targets, t_outputs, self.distill_temperature, epoch_num, max_epoch)




    def _generate_anchors(self, feats=None, dtype='float32'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = paddle.arange(end=w) + self.grid_cell_offset
            shift_y = paddle.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(paddle.full([h * w, 1], stride, dtype=dtype))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor


    def forward_eval(self, feats):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            feat = self.stem_conv[i](feat)
            cls_logit = self.pred_cls[i](feat)
            reg_dist = self.pred_reg[i](feat)
            if self.use_dfl:  ### diff with PPYOLOEHead
                reg_dist = reg_dist.reshape(
                    [-1, 4, self.reg_max + 1, l]).transpose([0, 2, 1, 3])
                reg_dist = self.proj_conv(F.softmax(reg_dist, axis=1))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([b, 4, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=-1)

        return cls_score_list, reg_distri_list, anchor_points, stride_tensor


    def forward_raw(self, feats):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            feat = self.stem_conv[i](feat)
            cls_logit = self.pred_cls[i](feat)
            reg_distri = self.pred_reg[i](feat)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        return cls_score_list, reg_distri_list, anchor_points, stride_tensor


    def forward(self, feats, targets=None, t_outputs=None, epoch_num=None, max_epoch=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"
        if self.training:
            if self.self_distill:
                return self.forward_train_with_distill(feats, targets, t_outputs, epoch_num, max_epoch)
            else:
                return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)


    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        ### diff with PPYOLOEHead
        if self.use_dfl:
            b, l, _ = get_static_shape(pred_dist)
            pred_dist = F.softmax(
                pred_dist.reshape([b, l, 4, self.reg_max + 1])).matmul(
                    self.proj)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = paddle.cast(target, 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)


    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # iou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            # l1 loss just see the convergence, same in PPYOLOEHead
            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            # dfl loss ### diff with PPYOLOEHead
            if self.use_dfl:
                dist_mask = mask_positive.unsqueeze(-1).tile(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = paddle.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                assigned_ltrb = self._bbox2distance(anchor_points,
                                                    assigned_bboxes)
                assigned_ltrb_pos = paddle.masked_select(
                    assigned_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         assigned_ltrb_pos) * bbox_weight
                loss_dfl = loss_dfl.sum() / assigned_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl



    def _bbox_loss_with_distill(self, pred_dist, pred_bboxes, 
                t_pred_dist, t_pred_bboxes, temperature,
                anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # iou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            # t_pred_bboxes_pos = paddle.masked_select(t_pred_bboxes,
                                                    # bbox_mask).reshape([-1, 4])

            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            # l1 loss just see the convergence, same in PPYOLOEHead
            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)


            # dfl loss ### diff with PPYOLOEHead
            if self.use_dfl:

                dist_mask = mask_positive.unsqueeze(-1).tile(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = paddle.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                t_pred_dist_pos = paddle.masked_select(
                    t_pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])

                assigned_ltrb = self._bbox2distance(anchor_points,
                                                    assigned_bboxes)
                assigned_ltrb_pos = paddle.masked_select(
                    assigned_ltrb, bbox_mask).reshape([-1, 4])

                loss_dfl = self._df_loss(pred_dist_pos,
                                         assigned_ltrb_pos) * bbox_weight
                
                d_loss_dfl = self._distill_dfl_loss(pred_dist_pos, t_pred_dist_pos, temperature) * bbox_weight

                if assigned_scores_sum != 0:
                    loss_dfl = loss_dfl.sum() / assigned_scores_sum
                    d_loss_dfl = d_loss_dfl.sum() / assigned_scores_sum
                else:
                    loss_dfl = loss_dfl.sum()
                    d_loss_dfl = d_loss_dfl.sum()

            else:
                loss_dfl = pred_dist.sum() * 0.
                d_loss_dfl = pred_dist.sum() * 0.
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl, d_loss_dfl




    def _distill_cls_loss(self, logits_student, logits_teacher, num_classes, temperature=20):
        logits_student = logits_student.reshape((-1, num_classes))
        logits_teacher = logits_teacher.reshape((-1, num_classes))
        pred_student = F.softmax(logits_student / temperature, axis=1)
        pred_teacher = F.softmax(logits_teacher / temperature, axis=1)
        log_pred_student = paddle.log(pred_student)

        d_loss_cls = F.kl_div(log_pred_student, pred_teacher, reduction="sum")
        d_loss_cls *= temperature ** 2
        return d_loss_cls


    def _distill_dfl_loss(self, logits_student, logits_teacher, temperature=20):
        logits_student = logits_student.reshape((-1, self.reg_max + 1))
        logits_teacher = logits_teacher.reshape((-1, self.reg_max + 1))
        pred_student = F.softmax(logits_student / temperature, axis=1)
        pred_teacher = F.softmax(logits_teacher / temperature, axis=1)
        log_pred_student = paddle.log(pred_student)

        d_loss_dfl = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        d_loss_dfl *= temperature ** 2
        return d_loss_dfl


    def _distill_cw_loss(self, s_feats, t_feats, temperature=1):
        loss_cw = 0.0
        for s_feat, t_feat in zip(s_feats, t_feats):
            N, C, H, W = s_feat.shape
            loss_cw += F.kl_div(F.log_softmax(s_feat.reshape((N, C, H * W)) / temperature, axis=2),
                            F.log_softmax(t_feat.reshape((N, C, H * W)).detach() / temperature, axis=2),
                            reduction='sum',
                            log_target=True) * (temperature * temperature)/ (N * C)
        return loss_cw



    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
        # rescale bbox
        assigned_bboxes /= stride_tensor

        # cls loss: varifocal_loss
        one_hot_label = F.one_hot(assigned_labels,
                                  self.num_classes + 1)[..., :-1]
        loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                        one_hot_label)
        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        loss_cls /= assigned_scores_sum

        # bbox loss
        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        if self.use_dfl:
            loss = self.loss_weight['cls'] * loss_cls + \
                self.loss_weight['iou'] * loss_iou + \
                self.loss_weight['dfl'] * loss_dfl
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls,
                'loss_iou': loss_iou,
                'loss_dfl': loss_dfl,
            }
        else:
            loss = self.loss_weight['cls'] * loss_cls + \
                self.loss_weight['iou'] * loss_iou
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls,
                'loss_iou': loss_iou,
            }

        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1})
        return out_dict


    def get_loss_with_distill(self, head_outs, gt_meta, t_outputs, 
        temperature, epoch_num, max_epoch):
        neck_feats, pred_scores, pred_distri, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        # []
        t_neck_feats, t_pred_scores, t_pred_distri, \
        t_anchor_points, t_stride_tensor = t_outputs

        # print("pred_scores : ", pred_scores.shape)
        # print("pred_distri : ", pred_distri.shape)
        # print("t_pred_scores : ", t_pred_scores.shape)        
        # print("t_pred_distri : ", t_pred_distri.shape)

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        t_anchor_points_s = t_anchor_points / t_stride_tensor
        t_pred_bboxes = self._bbox_decode(t_anchor_points_s, t_pred_distri)


        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
        # rescale bbox
        assigned_bboxes /= stride_tensor

        # cls loss: varifocal_loss
        one_hot_label = F.one_hot(assigned_labels,
                                  self.num_classes + 1)[..., :-1]
        loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                        one_hot_label)
        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        loss_cls /= assigned_scores_sum

        # bbox loss
        loss_l1, loss_iou, loss_dfl, d_loss_dfl = \
            self._bbox_loss_with_distill(pred_distri, pred_bboxes, 
                            t_pred_distri, t_pred_bboxes, temperature,
                            anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        d_loss_cls = self._distill_cls_loss(pred_scores, t_pred_scores, self.num_classes, temperature)
        
        if self.distill_feat:
            d_loss_cw = self._distill_cw_loss(neck_feats, t_neck_feats)
        else:
            d_loss_cw = paddle.to_tensor(0.0, place=neck_feats[0].place)

        import math
        distill_weightdecay = ((1 - math.cos(epoch_num * math.pi / max_epoch)) / 2) * (0.01 - 1) + 1
        d_loss_dfl *= distill_weightdecay
        d_loss_cls *= distill_weightdecay
        d_loss_cw *= distill_weightdecay

        loss_cls_all = loss_cls + d_loss_cls * self.distill_weight['cls']
        loss_dfl_all = loss_dfl + d_loss_dfl * self.distill_weight['dfl']

        if self.use_dfl:
            loss = self.loss_weight['cls'] * loss_cls_all + \
                self.loss_weight['iou'] * loss_iou + \
                self.loss_weight['dfl'] * loss_dfl_all + \
                self.loss_weight['cwd'] * d_loss_cw
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls_all,
                'loss_iou': loss_iou,
                'loss_dfl': loss_dfl_all,
                'loss_cwd': d_loss_cw,
            }
        else:
            loss = self.loss_weight['cls'] * loss_cls_all + \
                self.loss_weight['iou'] * loss_iou + \
                self.loss_weight['cw'] * d_loss_cw
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls,
                'loss_iou': loss_iou,
                'loss_cwd': d_loss_cw,
            }

        # for k, v in out_dict.items():
        #     print(k, v.shape)

        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1})
        return out_dict




    def post_process(self, head_outs, im_shape, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points,
                                          pred_dist.transpose([0, 2, 1]))
        pred_bboxes *= stride_tensor
        # scale bbox to origin
        scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
        scale_factor = paddle.concat(
            [scale_x, scale_y, scale_x, scale_y], axis=-1).reshape([-1, 1, 4])
        pred_bboxes /= scale_factor

        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])],
                axis=-1), paddle.to_tensor(
                    [1], dtype='int32')
        else:
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes.sum(), pred_scores.sum()
            else:
                bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                return bbox_pred, bbox_num
