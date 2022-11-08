# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from distutils.log import debug
from os import cpu_count

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ..bbox_utils import iou_similarity, batch_iou_similarity
from ..bbox_utils import bbox_center
from .utils import (check_points_inside_bboxes, compute_max_iou_anchor,
                    compute_max_iou_gt)

__all__ = ['ATSSAssigner']





@register
class ATSSAssigner(nn.Layer):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection
     via Adaptive Training Sample Selection
    """
    __shared__ = ['num_classes']

    def __init__(self,
                 topk=9,
                 num_classes=80,
                 force_gt_matching=False,
                 eps=1e-9):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps





    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
                             pad_gt_mask):



        gt2anchor_distances_list = paddle.split(gt2anchor_distances, num_anchors_list, axis=-1)


        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]



        is_in_topk_list = []
        topk_idxs_list = []




        for distances, anchors_index in zip(gt2anchor_distances_list, 
                                            num_anchors_index):


            num_anchors = distances.shape[-1]
            
            _, topk_idxs = paddle.topk(distances, self.topk, axis=-1, largest=False)


            topk_idxs_list.append(topk_idxs + anchors_index)
            
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2).astype(gt2anchor_distances.dtype)

            is_in_topk_list.append(is_in_topk * pad_gt_mask)




        is_in_topk_list = paddle.concat(is_in_topk_list, axis=-1)
        topk_idxs_list = paddle.concat(topk_idxs_list, axis=-1)


        return is_in_topk_list, topk_idxs_list







    @paddle.no_grad()
    def forward(self,
                anchor_bboxes,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None,
                pred_bboxes=None):



        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.


        Args:

            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level


            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)


            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label


            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)


        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """



        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3



        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape


        # negative batch
        if num_max_boxes == 0:


            assigned_labels = paddle.full([batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = paddle.zeros([batch_size, num_anchors, 4])
            assigned_scores = paddle.zeros([batch_size, num_anchors, self.num_classes])


            return assigned_labels, assigned_bboxes, assigned_scores




        # 1. compute iou between gt and anchor bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        ious = ious.reshape([batch_size, -1, num_anchors])



        # 2. compute center distance between all anchors and gt, [B, n, L]
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)) \
            .norm(2, axis=-1).reshape([batch_size, -1, num_anchors])






        # 3. on each pyramid level, selecting topk closest candidates
        # based on the center distance, [B, n, L]
        is_in_topk, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask)



        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk
        iou_threshold = paddle.index_sample(iou_candidates.flatten(stop_axis=-2), topk_idxs.flatten(stop_axis=-2))



        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(axis=-1, keepdim=True) + iou_threshold.std(axis=-1, keepdim=True)





        is_in_topk = paddle.where(iou_candidates > iou_threshold, is_in_topk, 
                                  paddle.zeros_like(is_in_topk))




        # 6. check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)



        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask




        # 7. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = paddle.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)






        # 8. make sure every gt_bbox matches the anchor
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).tile(
                [1, num_max_boxes, 1])
            mask_positive = paddle.where(mask_max_iou, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)

        assigned_gt_index = mask_positive.argmax(axis=-2)




        # assigned target
        batch_ind = paddle.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = paddle.gather(
            gt_labels.flatten(), assigned_gt_index.flatten(), axis=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = paddle.where(
            mask_positive_sum > 0, assigned_labels,
            paddle.full_like(assigned_labels, bg_index))



        assigned_bboxes = paddle.gather(
            gt_bboxes.reshape([-1, 4]), assigned_gt_index.flatten(), axis=0)
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])



        assigned_scores = F.one_hot(assigned_labels, self.num_classes + 1)
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)



        assigned_scores = paddle.index_select(assigned_scores, paddle.to_tensor(ind), axis=-1)
        if pred_bboxes is not None:
            # assigned iou
            ious = batch_iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious = ious.max(axis=-2).unsqueeze(-1)
            assigned_scores *= ious







        elif gt_scores is not None:

            gather_scores = paddle.gather(gt_scores.flatten(), assigned_gt_index.flatten(), axis=0)
            gather_scores = gather_scores.reshape([batch_size, num_anchors])
            gather_scores = paddle.where(mask_positive_sum > 0, gather_scores, paddle.zeros_like(gather_scores))
            assigned_scores *= gather_scores.unsqueeze(-1)




        return assigned_labels, assigned_bboxes, assigned_scores









################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################




    @paddle.no_grad()
    def forward_debug(self,
                anchor_bboxes,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None,
                pred_bboxes=None):



        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.


        Args:

            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level


            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)


            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label


            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)


        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """

        print("ATSS Assigner Debug, forward_debug")



        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3



        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape




        # negative batch
        if num_max_boxes == 0:


            assigned_labels = paddle.full([batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = paddle.zeros([batch_size, num_anchors, 4])
            assigned_scores = paddle.zeros([batch_size, num_anchors, self.num_classes])

            return assigned_labels, assigned_bboxes, assigned_scores




        # 1. compute iou between gt and anchor bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        ious = ious.reshape([batch_size, -1, num_anchors])

        print("ious = iou_similarity(gt_bboxes.reshape([-1, 4]), anchor_bboxes)")
        print("ious : ", ious.shape)


        # print()
        # print()


        # 2. compute center distance between all anchors and gt, [B, n, L]
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)).norm(2, axis=-1).reshape([batch_size, -1, num_anchors])


        print("gt_centers : ", gt_centers.shape)
        print("anchor_centers : ", anchor_centers.shape)
        print("gt2anchor_distances : ", gt2anchor_distances.shape)




        # 3. on each pyramid level, selecting topk closest candidates
        # based on the center distance, [B, n, L]
        is_in_topk, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask)


        print("is_in_topk : ", is_in_topk.shape)
        print("topk_idxs : ", topk_idxs.shape)






        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk
        iou_threshold_candidates = paddle.index_sample(iou_candidates.flatten(stop_axis=-2), topk_idxs.flatten(stop_axis=-2))



        print("iou_candidates : ", iou_candidates.shape)
        print("iou_threshold_candidates : ", iou_threshold_candidates.shape)


        iou_threshold_candidates = iou_threshold_candidates.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold_candidates.mean(axis=-1, keepdim=True) + iou_threshold_candidates.std(axis=-1, keepdim=True)



        is_in_topk = paddle.where(iou_candidates > iou_threshold, is_in_topk, paddle.zeros_like(is_in_topk))



        # 6. check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)




        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask



        # 7. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.


        print("mask_positive : ", mask_positive.shape)
        def count(array, name):
            array = array.cpu().numpy()
            min_value = int(array.min())
            max_value = int(array.max())

            counter = {
                i : np.sum((array == i).astype(np.int)) for i in range(min_value, max_value+1)
            }

            print(name, counter)


        def get_box_center(bbox):
            return  int((bbox[0] + bbox[2])  / 2), int((bbox[1] + bbox[3]    ) / 2)

        def get_box_scale(bbox):
            return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

        def get_box_center_offset(bbox1, bbox2):
            c1 = get_box_center(bbox1)
            c2 = get_box_center(bbox2)
            return int(c1[0] - c2[0]), int(c1[1] - c2[1])


        anchor_bboxes_numpy = anchor_bboxes.cpu().numpy()




        def debug_gt(ind):

            gt_bbox = gt_bboxes[0][ind].cpu().numpy()

            print(ind, gt_bbox)
            print("\tbox center : {}".format(get_box_center(gt_bbox)))
            print("\tbox scale : {}".format(get_box_scale(gt_bbox)))


            topk_idxs_ind = topk_idxs[0,ind].cpu().numpy()
            is_in_topk_ind = np.where(is_in_topk[0, ind].cpu().numpy() != 0)[0]
            is_in_gts_ind = np.where(is_in_gts[0, ind].cpu().numpy() != 0)[0]




            print("\tis_in_topk before thres : ",  topk_idxs_ind)
            print("\tis_in_topk center before thres : ",  [get_box_center(anchor_bboxes_numpy[i]) for i in topk_idxs_ind])
            print("\tis_in_topk offset  before thres : ",  [get_box_center_offset(gt_bbox, anchor_bboxes_numpy[i]) for i in topk_idxs_ind])
            print("\tis_in_topk scale  before thres : ",  [get_box_scale(anchor_bboxes_numpy[i]) for i in topk_idxs_ind])
            
            print("\tis_in_topk before thres iou : ",  iou_threshold_candidates[0, ind].cpu().numpy())
            print("\tiou_threshold : ", iou_threshold[0, ind])


            print("\tis_in_topk : ",  is_in_topk_ind)
            print("\tis_in_topk center : ",  [get_box_center(anchor_bboxes_numpy[i]) for i in is_in_topk_ind])
            print("\tis_in_topk offset  : ",  [get_box_center_offset(gt_bbox, anchor_bboxes_numpy[i]) for i in is_in_topk_ind])
            print("\tis_in_topk scale  : ",  [get_box_scale(anchor_bboxes_numpy[i]) for i in is_in_topk_ind])
            
            print("\tis_in_gts : ",  is_in_gts_ind)
            print("\tis_in_gts center : ",  [get_box_center(anchor_bboxes_numpy[i]) for i in is_in_gts_ind])
            print("\tis_in_gts offset  : ",  [get_box_center_offset(gt_bbox, anchor_bboxes_numpy[i]) for i in is_in_gts_ind])
            print("\tis_in_gts scale  : ",  [get_box_scale(anchor_bboxes_numpy[i]) for i in is_in_gts_ind])


            print()


        mask_positive_sum1 = mask_positive.sum(axis=-1)
        print("mask_positive_sum1 : ", mask_positive_sum1.shape)
        print("mask_positive_sum1.max() : ", mask_positive_sum1.max().item())
        print("mask_positive_sum1.min() : ", mask_positive_sum1.min().item())
        print()
        print("*"*40)
        print("*"*40)
        count(mask_positive_sum1, "mask_positive_sum1")
        ious_np = ious.cpu().numpy().sum(axis=-1)
        print("ious_np : ", ious_np.shape, ious_np.max(), ious_np.min())



        mask_positive_sum1 = mask_positive_sum1.cpu().numpy()
        print(np.where(mask_positive_sum1[0] == 0.0))
        print(np.where(mask_positive_sum1[0] == 0))
        for i in np.where(mask_positive_sum1[0] == 0.0)[0]:
            # print(i, gt_bboxes[0][i].cpu().numpy())

            debug_gt(i)

        print("*"*40)
        print("*"*40)
        print()
        



        mask_positive_sum = mask_positive.sum(axis=-2)
        print("mask_positive_sum2 : ", mask_positive_sum.shape)
        print("mask_positive_sum2.max() : ", mask_positive_sum.max().item())
        print("mask_positive_sum2.min() : ", mask_positive_sum.min().item())





        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = paddle.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)




        # 8. make sure every gt_bbox matches the anchor
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask

            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).tile([1, num_max_boxes, 1])
            mask_positive = paddle.where(mask_max_iou, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)

        assigned_gt_index = mask_positive.argmax(axis=-2)





        # assigned target
        batch_ind = paddle.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = paddle.gather(
            gt_labels.flatten(), assigned_gt_index.flatten(), axis=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = paddle.where(
            mask_positive_sum > 0, assigned_labels,
            paddle.full_like(assigned_labels, bg_index))



        assigned_bboxes = paddle.gather(
            gt_bboxes.reshape([-1, 4]), assigned_gt_index.flatten(), axis=0)
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])



        assigned_scores = F.one_hot(assigned_labels, self.num_classes + 1)
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)



        assigned_scores = paddle.index_select(assigned_scores, paddle.to_tensor(ind), axis=-1)
        if pred_bboxes is not None:
            # assigned iou
            ious = batch_iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious = ious.max(axis=-2).unsqueeze(-1)
            assigned_scores *= ious






        elif gt_scores is not None:

            gather_scores = paddle.gather(gt_scores.flatten(), assigned_gt_index.flatten(), axis=0)
            gather_scores = gather_scores.reshape([batch_size, num_anchors])
            gather_scores = paddle.where(mask_positive_sum > 0, gather_scores, paddle.zeros_like(gather_scores))
            assigned_scores *= gather_scores.unsqueeze(-1)





        return assigned_labels, assigned_bboxes, assigned_scores




