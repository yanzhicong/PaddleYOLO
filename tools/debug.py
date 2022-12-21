# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob
import ast
import json
from tqdm import tqdm
import copy
import numpy as np
import typing
from PIL import Image, ImageOps, ImageFile, ImageDraw
import cv2




from scipy.optimize import linear_sum_assignment as linear_assignment
# from sklearn.utils.linear_assignment_ import linear_assignment

import paddle
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model


from ppdet.utils.visualizer import draw_bbox

from ppdet.metrics import Metric, COCOMetric, VOCMetric, get_infer_results


from ppdet.utils.logger import setup_logger
logger = setup_logger('train')





# CUDA_VISIBLE_DEVICES=1 python3.7 tools/debug.py -c sku110_configs/ppyoloe_plus_crn_s_80e_coco_large_size.yml -o weights=./sku110_output/ppyoloe_plus_crn_s_80e_coco_large_size/model_final.pdparams




def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")


    parser.add_argument(
        "--img_anno_filepath",
        type=str,
        default="/workspace/dataset/SKU110K_fixed/json_annotations/annotations_val.json",
        help="Image path, has higher priority over --infer_dir")



    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="Whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="Whether to save inference results to output_dir.")
    parser.add_argument(
        "--slice_infer",
        action='store_true',
        help="Whether to slice the image and merge the inference results for small object detection."
    )
    parser.add_argument(
        '--slice_size',
        nargs='+',
        type=int,
        default=[640, 640],
        help="Height of the sliced image.")
    parser.add_argument(
        "--overlap_ratio",
        nargs='+',
        type=float,
        default=[0.25, 0.25],
        help="Overlap height ratio of the sliced image.")
    parser.add_argument(
        "--combine_method",
        type=str,
        default='nms',
        help="Combine method of the sliced images' detection results, choose in ['nms', 'nmm', 'concat']."
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.6,
        help="Combine method matching threshold.")
    parser.add_argument(
        "--match_metric",
        type=str,
        default='ios',
        help="Combine method matching metric, choose in ['iou', 'ios'].")
    parser.add_argument(
        "--visualize",
        type=ast.literal_eval,
        default=True,
        help="Whether to save visualize results to output_dir.")
    args = parser.parse_args()
    return args








def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images





def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # assert bb1['x1'] < bb1['x2']
    # assert bb1['y1'] < bb1['y2']
    # assert bb2['x1'] < bb2['x2']
    # assert bb2['y1'] < bb2['y2']



    # determine the coordinates of the intersection rectangle
    # x_left = max(bb1['x1'], bb2['x1'])
    # y_top = max(bb1['y1'], bb2['y1'])
    # x_right = min(bb1['x2'], bb2['x2'])
    # y_bottom = min(bb1['y2'], bb2['y2'])


    bb1 = np.array(bb1, copy=True)
    bb2 = np.array(bb2, copy=True)

    bb1_area = bb1[:, 2] * bb1[:, 3]
    bb2_area = bb2[:, 2] * bb2[:, 3]

    bb1[:, 2] = bb1[:, 2] + bb1[:, 0]
    bb1[:, 3] = bb1[:, 3] + bb1[:, 1]

    bb2[:, 2] = bb2[:, 2] + bb2[:, 0]
    bb2[:, 3] = bb2[:, 3] + bb2[:, 1]


    bb1 = np.expand_dims(bb1, 1)
    bb1_area = np.expand_dims(bb1_area, 1)
    bb2 = np.expand_dims(bb2, 0)
    bb2_area = np.expand_dims(bb2_area, 0)

    left = np.maximum(bb1[:, :, 0], bb2[:, :, 0])
    top = np.maximum(bb1[:, :, 1], bb2[:, :, 1])
    right = np.minimum(bb1[:, :, 2], bb2[:, :, 2])
    bottom = np.minimum(bb1[:, :, 3], bb2[:, :, 3])


    # print("bb1 : ", bb1.shape)
    # print("bb1_area : ", bb1_area.shape)
    # print("bb2 : ", bb2.shape)
    # print("bb2_area : ", bb2_area.shape)
    # print("left : ", left.shape, left.min(), left.max())
    # print("top : ", top.shape, top.min(), top.max())
    # print("right : ", right.shape, right.min(), right.max())
    # print("bottom : ", bottom.shape, bottom.min(), bottom.max())


    # if x_right < x_left or y_bottom < y_top:
    #     return 0.0
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box


    intersection_width = np.maximum(right - left, 0.0)
    intersection_height = np.maximum(bottom - top, 0.0)
    intersection_area = intersection_width * intersection_height

    # print("intersection_area : ", intersection_area.shape)


    # compute the area of both AABBs
    # bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    # bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area

    iou = intersection_area / (bb1_area + bb2_area - intersection_area)


    # assert iou >= 0.0
    # assert iou <= 1.0


    return iou








def draw_bbox_highlight(image, name, bboxes, color, highlight=1.0):
    """
    Draw bbox on image
    """


    # print("draw_bbox_hightlight, ", len(bboxes))
    # highlight bbox areas in input image
    # image = np.asarray(image).astype(np.uint8)
    # image2 = (image.astype(np.float) * highlight).astype(np.uint8)
    # for dt in np.array(bboxes):
    #     if im_id != dt['image_id']:
    #         continue


    #     catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
    #     if score < threshold:
    #         continue

    #     # draw bbox
    #     if len(bbox) == 4:
    #         # draw bbox
    #         xmin, ymin, w, h = bbox
    #         xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
    #         xmax = xmin + w
    #         ymax = ymin + h
    #         image2[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :]
    
    # image = Image.fromarray(np.uint8(image2))




    draw = ImageDraw.Draw(image)





    # catid2color = {}
    # color_list = colormap(rgb=True)[:40]


    for dt in np.array(bboxes):

        # if im_id != dt['image_id']:
        #     continue

        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']


        # if score < threshold:
        #     continue
        # if catid not in catid2color:
        #     idx = np.random.randint(len(color_list))
        #     catid2color[catid] = color_list[idx]
        # color = tuple(catid2color[catid])


        # draw bbox
        if len(bbox) == 4:
            # draw bbox
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=2,
                fill=color)


        # elif len(bbox) == 8:
        #     x1, y1, x2, y2, x3, y3, x4, y4 = bbox
        #     draw.line(
        #         [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
        #         width=2,
        #         fill=color)
        #     xmin = min(x1, x2, x3, x4)
        #     ymin = min(y1, y2, y3, y4)

        else:
            logger.error('the shape of bbox must be [M, 4] or [M, 8]!')




        # draw label
        text = "{} {:.3f}".format(name, score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))



    return image










def get_image_annos(anno_filepath):

    images_path = "/workspace/dataset/SKU110K_fixed/images"

    anno_json = json.loads(open(anno_filepath, 'r').read())

    cates = anno_json["categories"]
    bboxes = anno_json["annotations"]
    images = anno_json["images"]

    for b in bboxes:
        b["score"] = 1.0


    catid2name = {c["id"]:c["name"] for c in cates}
    img2ind = {i["file_name"]:i["id"] for i in images}


    imgind2imgfp = {
        i["id"] : os.path.join(images_path, i['file_name']) for i in images
    }

    
    grouped_bboxes = {
        os.path.join(images_path, i['file_name']):[] for i in images
    }


    for b in bboxes:
        grouped_bboxes[imgind2imgfp[b['image_id']]].append(b)

    # grouped_bboxes = {os.path.join("/workspace/dataset/SKU110K_fixed/images", fn):[] for fn in }

    return catid2name, bboxes, img2ind, grouped_bboxes








class DebugPredictWrapper(Trainer):

    def __init__(self, cfg, mode='train'):
        super(DebugPredictWrapper, self).__init__(cfg, mode)
        self.dataset = create("InferDataset")()

    def predict(self,
                images,
                grouped_annotations,
                draw_threshold=0.1,
                output_dir='output',
                iou_threshold = 0.5,
                visualize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)



        imid2path = self.dataset.get_imid2path()


        # Run Infer 
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []


        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()


           # print('bb')
            results.append(outs)


        clsid2catid = {
            0:0,
        }


        # clsid2catid = {v: k for k, v in self.dataset.catid2clsid.items()}
        #  \
                                # if self.mode == 'eval' else None
        print("clsid2catid : ")
        print(clsid2catid)


        catid2name = {0 : "object", 1 : "object"}




        if isinstance(draw_threshold, list):
            num_matched_gt = [0 for _ in draw_threshold]
            num_missed_gt = [0 for _ in draw_threshold]
            num_false_positive = [0 for _ in draw_threshold]
        else:
            num_matched_gt = 0
            num_missed_gt = 0
            num_false_positive = 0

    
        miss_gt_severe_images = []
        false_pos_severe_images = []


        def get_matched_indices(bbox_res, gt_res, iou_threshold):

            pred_bboxes = np.array([b["bbox"] for b in bbox_res]).astype(np.float32)
            gt_bboxes = np.array([g["bbox"] for g in gt_res]).astype(np.float32)

            iou = get_iou(pred_bboxes, gt_bboxes)

            indices = np.array(linear_assignment(-iou))

            matched_ious = []
            for i in range(indices.shape[1]):
                ind1 = indices[0, i]
                ind2 = indices[1, i]
                matched_ious.append(iou[ind1, ind2])

            matched_ious = np.array(matched_ious)

            indices = indices[:, np.where(matched_ious > iou_threshold)[0]]
            matched_bbox_indices = set(list(indices[0]))
            matched_gt_indices = set(list(indices[1]))

            return matched_bbox_indices, matched_gt_indices




        for outs in tqdm(results):

            batch_res = get_infer_results(outs, clsid2catid)
            bbox_num = outs['bbox_num']

            start = 0
            for i, im_id in enumerate(outs['im_id']):

                image_path = imid2path[int(im_id)]

                image = Image.open(image_path).convert('RGB')
                image = ImageOps.exif_transpose(image)
                self.status['original_image'] = np.array(image.copy())
                end = start + bbox_num[i]

                bbox_res = batch_res['bbox'][start:end]
                gt_res = grouped_annotations[image_path]




                # pred_bboxes = np.array([b["bbox"] for b in bbox_res]).astype(np.float32)
                # bbox_self_ious = get_iou(pred_bboxes, pred_bboxes)
                # np.fill_diagonal(bbox_self_ious, 0.0)
                # print("Max of self iou : ", np.max(bbox_self_ious))



                if isinstance(draw_threshold, list):
                    for j, t in enumerate(draw_threshold):

                        bbox_res_sub = [b for b in bbox_res if b['score'] > t]
                        matched_bbox_indices, matched_gt_indices = get_matched_indices(bbox_res_sub, gt_res, iou_threshold)
                        num_matched_gt[j] += len(matched_gt_indices)
                        num_missed_gt[j] += len(gt_res) - len(matched_gt_indices)
                        num_false_positive[j] += len(bbox_res_sub) - len(matched_bbox_indices)

                        if j == 0:
                            miss_gt_severe_images.append((len(gt_res) - len(matched_gt_indices), image_path))
                            false_pos_severe_images.append((len(bbox_res_sub) - len(matched_bbox_indices), image_path))

                else:
                    bbox_res_sub = [b for b in bbox_res if b['score'] > draw_threshold]
                    matched_bbox_indices, matched_gt_indices = get_matched_indices(bbox_res_sub, gt_res, iou_threshold)
                    num_matched_gt += len(matched_gt_indices)
                    num_missed_gt += len(gt_res) - len(matched_gt_indices)
                    num_false_positive += len(bbox_res_sub) - len(matched_bbox_indices)

                    miss_gt_severe_images.append((len(gt_res) - len(matched_gt_indices), image_path))
                    false_pos_severe_images.append((len(bbox_res_sub) - len(matched_bbox_indices), image_path))



                if visualize:
                    # matched ground truth, black
                    image = draw_bbox_highlight(
                        image,
                        "mgt",
                        [gt_res[i] for i in range(len(gt_res)) if i in matched_gt_indices], 
                        color = (0, 0, 0),
                    )

                    # matched prediction, green
                    image = draw_bbox_highlight(
                        image,
                        "mp",
                        [bbox_res_sub[i] for i in range(len(bbox_res_sub)) if i in matched_bbox_indices], 
                        color = (0, 255, 0),
                    )

                    # false positive, red
                    image = draw_bbox_highlight(
                        image,
                        "fp",
                        [bbox_res_sub[i] for i in range(len(bbox_res_sub)) if i not in matched_bbox_indices], 
                        color = (255, 0, 0),
                    )

                    # missing ground truth, blue
                    image = draw_bbox_highlight(
                        image,
                        "mgt",
                        [gt_res[i] for i in range(len(gt_res)) if i not in matched_gt_indices], 
                        color = (0, 0, 255),
                    )

                    save_name = self._get_save_image_name(output_dir,
                                                            image_path)

                    logger.info("Detection bbox results save in {}".format(
                        save_name))

                    image.save(save_name, quality=95)

                start = end



        miss_gt_severe_images = sorted(miss_gt_severe_images, key=lambda x:x[0], reverse=True)
        miss_gt_severe_images = miss_gt_severe_images[0:30]
        false_pos_severe_images = sorted(false_pos_severe_images, key=lambda x:x[0], reverse=True)
        false_pos_severe_images = false_pos_severe_images[0:30]

        # print("Severe Missing GT Images : ")
        # for i in miss_gt_severe_images:
        #     print("\t", i)

        # print("Severe False Positive Images")
        # for i in false_pos_severe_images:
        #     print("\t", i)


        return num_matched_gt, num_missed_gt, num_false_positive




def run(FLAGS, cfg):
    # build trainer
    trainer = DebugPredictWrapper(cfg, mode='test')

    # load weights
    trainer.load_weights(cfg.weights)

    # get inference images
    # images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)

    print("get annos")
    catid2name, bbox_res, img2ind, grouped_bboxes = get_image_annos(FLAGS.img_anno_filepath)


    infer_dir = "/workspace/dataset/SKU110K_fixed/images"
    images = [os.path.join(infer_dir, fn) for fn in os.listdir(infer_dir) if fn.startswith('val') and fn.endswith('.jpg')]


    # num_matched_gt, num_missed_gt, num_false_positive = trainer.predict(
    #     images,
    #     grouped_bboxes,
    #     draw_threshold=0.3,
    #     output_dir=FLAGS.output_dir,
    #     visualize=True)

    # images = [
    #   (967, '/workspace/dataset/SKU110K_fixed/images/val_316.jpg'),
    #      (930, '/workspace/dataset/SKU110K_fixed/images/val_483.jpg'),
    #      (925, '/workspace/dataset/SKU110K_fixed/images/val_140.jpg'),
    #      (922, '/workspace/dataset/SKU110K_fixed/images/val_417.jpg'),
    #      (891, '/workspace/dataset/SKU110K_fixed/images/val_293.jpg'),
    #      (872, '/workspace/dataset/SKU110K_fixed/images/val_13.jpg'),
    #      (872, '/workspace/dataset/SKU110K_fixed/images/val_143.jpg'),
    #      (868, '/workspace/dataset/SKU110K_fixed/images/val_199.jpg'),
    #      (865, '/workspace/dataset/SKU110K_fixed/images/val_384.jpg'),
    #      (862, '/workspace/dataset/SKU110K_fixed/images/val_305.jpg'),
    #      (853, '/workspace/dataset/SKU110K_fixed/images/val_508.jpg'),
    #      (852, '/workspace/dataset/SKU110K_fixed/images/val_112.jpg'),
    #      (850, '/workspace/dataset/SKU110K_fixed/images/val_119.jpg'),
    #      (841, '/workspace/dataset/SKU110K_fixed/images/val_224.jpg'),
    #      (829, '/workspace/dataset/SKU110K_fixed/images/val_363.jpg'),
    #      (803, '/workspace/dataset/SKU110K_fixed/images/val_117.jpg'),
    #      (798, '/workspace/dataset/SKU110K_fixed/images/val_420.jpg'),
    #      (782, '/workspace/dataset/SKU110K_fixed/images/val_317.jpg'),
    #      (767, '/workspace/dataset/SKU110K_fixed/images/val_81.jpg'),
    #      (762, '/workspace/dataset/SKU110K_fixed/images/val_496.jpg'),
    #      (758, '/workspace/dataset/SKU110K_fixed/images/val_206.jpg'),
    #      (751, '/workspace/dataset/SKU110K_fixed/images/val_144.jpg'),
    #      (738, '/workspace/dataset/SKU110K_fixed/images/val_327.jpg'),
    #      (736, '/workspace/dataset/SKU110K_fixed/images/val_230.jpg'),
    #      (735, '/workspace/dataset/SKU110K_fixed/images/val_241.jpg'),
    #      (728, '/workspace/dataset/SKU110K_fixed/images/val_389.jpg'),
    #      (719, '/workspace/dataset/SKU110K_fixed/images/val_558.jpg'),
    #      (715, '/workspace/dataset/SKU110K_fixed/images/val_168.jpg'),
    #      (703, '/workspace/dataset/SKU110K_fixed/images/val_386.jpg'),
    #      (689, '/workspace/dataset/SKU110K_fixed/images/val_94.jpg'),
    # ]
    # images = [i[1] for i in images]


    # num_matched_gt, num_missed_gt, num_false_positive = trainer.predict(
    #     images,
    #     grouped_bboxes,
    #     draw_threshold=0.3,
    #     output_dir=os.path.join(FLAGS.output_dir, "severe_fp_0_3"),
    #     visualize=True)

    # images = [
    #      (133, '/workspace/dataset/SKU110K_fixed/images/val_577.jpg'),
    #      (108, '/workspace/dataset/SKU110K_fixed/images/val_140.jpg'),
    #      (87, '/workspace/dataset/SKU110K_fixed/images/val_204.jpg'),
    #      (85, '/workspace/dataset/SKU110K_fixed/images/val_57.jpg'),
    #      (66, '/workspace/dataset/SKU110K_fixed/images/val_150.jpg'),
    #      (63, '/workspace/dataset/SKU110K_fixed/images/val_569.jpg'),
    #      (63, '/workspace/dataset/SKU110K_fixed/images/val_284.jpg'),
    #      (62, '/workspace/dataset/SKU110K_fixed/images/val_420.jpg'),
    #      (61, '/workspace/dataset/SKU110K_fixed/images/val_346.jpg'),
    #      (59, '/workspace/dataset/SKU110K_fixed/images/val_44.jpg'),
    #      (59, '/workspace/dataset/SKU110K_fixed/images/val_473.jpg'),
    #      (56, '/workspace/dataset/SKU110K_fixed/images/val_95.jpg'),
    #      (53, '/workspace/dataset/SKU110K_fixed/images/val_229.jpg'),
    #      (50, '/workspace/dataset/SKU110K_fixed/images/val_270.jpg'),
    #      (48, '/workspace/dataset/SKU110K_fixed/images/val_504.jpg'),
    #      (46, '/workspace/dataset/SKU110K_fixed/images/val_475.jpg'),
    #      (46, '/workspace/dataset/SKU110K_fixed/images/val_398.jpg'),
    #      (42, '/workspace/dataset/SKU110K_fixed/images/val_261.jpg'),
    #      (41, '/workspace/dataset/SKU110K_fixed/images/val_42.jpg'),
    #      (40, '/workspace/dataset/SKU110K_fixed/images/val_386.jpg'),
    #      (39, '/workspace/dataset/SKU110K_fixed/images/val_257.jpg'),
    #      (39, '/workspace/dataset/SKU110K_fixed/images/val_147.jpg'),
    #      (36, '/workspace/dataset/SKU110K_fixed/images/val_447.jpg'),
    #      (35, '/workspace/dataset/SKU110K_fixed/images/val_117.jpg'),
    #      (34, '/workspace/dataset/SKU110K_fixed/images/val_212.jpg'),
    #      (32, '/workspace/dataset/SKU110K_fixed/images/val_567.jpg'),
    #      (32, '/workspace/dataset/SKU110K_fixed/images/val_279.jpg'),
    #      (31, '/workspace/dataset/SKU110K_fixed/images/val_200.jpg'),
    #      (30, '/workspace/dataset/SKU110K_fixed/images/val_494.jpg'),
    #      (29, '/workspace/dataset/SKU110K_fixed/images/val_18.jpg'),
    # ]

    # images = [i[1] for i in images]

    # num_matched_gt, num_missed_gt, num_false_positive = trainer.predict(
    #     images,
    #     grouped_bboxes,
    #     draw_threshold=0.3,
    #     output_dir=os.path.join(FLAGS.output_dir, "severe_mp_0_3"),
    #     visualize=True)



    num_matched_gt, num_missed_gt, num_false_positive = trainer.predict(
        images,
        grouped_bboxes,
        draw_threshold=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        output_dir=FLAGS.output_dir,
        visualize=False)

    print("num_matched_gt : ", num_matched_gt)
    print("num_missed_gt : ", num_missed_gt)
    print("num_false_positive : ", num_false_positive)



def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)


    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False


    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    else:
        place = paddle.set_device('cpu')

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')


    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_version()

    run(FLAGS, cfg)






if __name__ == '__main__':
    main()









# train input size 640
# test  input size 960

# PPYoloE CRN + X 
# score_threshold=              [0.01,    0.05,    0.1,     0.2,     0.3,    0.4,    0.5],
# num_matched_gt :              [87896,   87896,   87856,   87175,   85252,  83050,  80348]
# num_missed_gt :               [3072,    3072,    3112,    3793,    5716,   7918,   10620]
# num_false_positive :          [108948,  106873,  89566,   44830,   18687,  9621,   6074]

# PPYoloE CRN + S
# train input size 640
# test  input size 640
# num_matched_gt :              [86728,   86724,   86603,   85359,   83114,   79903,  75894]
# num_missed_gt :               [4240,    4244,    4365,    5609,    7854,    11065,  15074]
# num_false_positive :          [232640,  223951,  162040,  64030,   26970,   12825,  6827]

# PPYoloE CRN + S
# train input size 640
# test  input size 960
# num_matched_gt :               [87228,  87228,   87217,   86364,   84024,  80849,  77005]
# num_missed_gt :                [3740,   3740,    3751,    4604,    6944,   10119,  13963]
# num_false_positive :           [150037, 149348,  135422,  63905,   25711,  12033,  6657]



# PPYoloE CRN + S
# train input size 960
# test  input size 960
# num_matched_gt :               [87121,  87121,   87095,   86277,   83878,   80403,  76303]
# num_missed_gt :                [3847,   3847,    3873,    4691,    7090,    10565,  14665]
# num_false_positive :           [144841, 143791,  129610,  62781,   24818,   11597,  6338]



# (11, '/workspace/dataset/SKU110K_fixed/images/val_29.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_400.jpg')
# (17, '/workspace/dataset/SKU110K_fixed/images/val_300.jpg')
# (11, '/workspace/dataset/SKU110K_fixed/images/val_301.jpg')
# (18, '/workspace/dataset/SKU110K_fixed/images/val_81.jpg')
# (17, '/workspace/dataset/SKU110K_fixed/images/val_163.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_507.jpg')
# (17, '/workspace/dataset/SKU110K_fixed/images/val_554.jpg')
# (11, '/workspace/dataset/SKU110K_fixed/images/val_232.jpg')
# (19, '/workspace/dataset/SKU110K_fixed/images/val_306.jpg')
# (19, '/workspace/dataset/SKU110K_fixed/images/val_511.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_185.jpg')
# (17, '/workspace/dataset/SKU110K_fixed/images/val_426.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_177.jpg')
# (11, '/workspace/dataset/SKU110K_fixed/images/val_66.jpg')
# (15, '/workspace/dataset/SKU110K_fixed/images/val_245.jpg')
# (14, '/workspace/dataset/SKU110K_fixed/images/val_65.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_168.jpg')

# (15, '/workspace/dataset/SKU110K_fixed/images/val_274.jpg')
# (17, '/workspace/dataset/SKU110K_fixed/images/val_180.jpg')
# (14, '/workspace/dataset/SKU110K_fixed/images/val_364.jpg')
# (16, '/workspace/dataset/SKU110K_fixed/images/val_393.jpg')
# (18, '/workspace/dataset/SKU110K_fixed/images/val_316.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_459.jpg')
# (15, '/workspace/dataset/SKU110K_fixed/images/val_52.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_39.jpg')
# (15, '/workspace/dataset/SKU110K_fixed/images/val_246.jpg')
# (15, '/workspace/dataset/SKU110K_fixed/images/val_381.jpg')
# (11, '/workspace/dataset/SKU110K_fixed/images/val_36.jpg')
# (16, '/workspace/dataset/SKU110K_fixed/images/val_329.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_222.jpg')
# (18, '/workspace/dataset/SKU110K_fixed/images/val_555.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_474.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_287.jpg')
# (15, '/workspace/dataset/SKU110K_fixed/images/val_418.jpg')
# (19, '/workspace/dataset/SKU110K_fixed/images/val_581.jpg')
# (16, '/workspace/dataset/SKU110K_fixed/images/val_96.jpg')
# (11, '/workspace/dataset/SKU110K_fixed/images/val_517.jpg')
# (14, '/workspace/dataset/SKU110K_fixed/images/val_336.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_40.jpg')
# (11, '/workspace/dataset/SKU110K_fixed/images/val_362.jpg')
# (17, '/workspace/dataset/SKU110K_fixed/images/val_560.jpg')
# (19, '/workspace/dataset/SKU110K_fixed/images/val_148.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_458.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_550.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_195.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_528.jpg')
# (11, '/workspace/dataset/SKU110K_fixed/images/val_365.jpg')
# (15, '/workspace/dataset/SKU110K_fixed/images/val_297.jpg')
# (18, '/workspace/dataset/SKU110K_fixed/images/val_126.jpg')
# (19, '/workspace/dataset/SKU110K_fixed/images/val_359.jpg')
# (11, '/workspace/dataset/SKU110K_fixed/images/val_566.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_241.jpg')
# (11, '/workspace/dataset/SKU110K_fixed/images/val_242.jpg')
# (18, '/workspace/dataset/SKU110K_fixed/images/val_485.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_173.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_291.jpg')

# (19, '/workspace/dataset/SKU110K_fixed/images/val_529.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_520.jpg')
# (14, '/workspace/dataset/SKU110K_fixed/images/val_375.jpg')
# (17, '/workspace/dataset/SKU110K_fixed/images/val_369.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_94.jpg')
# (16, '/workspace/dataset/SKU110K_fixed/images/val_7.jpg')
# (19, '/workspace/dataset/SKU110K_fixed/images/val_513.jpg')
# (18, '/workspace/dataset/SKU110K_fixed/images/val_326.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_1.jpg')
# (16, '/workspace/dataset/SKU110K_fixed/images/val_115.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_268.jpg')
# (14, '/workspace/dataset/SKU110K_fixed/images/val_407.jpg')
# (11, '/workspace/dataset/SKU110K_fixed/images/val_280.jpg')
# (11, '/workspace/dataset/SKU110K_fixed/images/val_0.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_119.jpg')
# (14, '/workspace/dataset/SKU110K_fixed/images/val_542.jpg')
# (12, '/workspace/dataset/SKU110K_fixed/images/val_510.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_438.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_391.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_440.jpg')
# (15, '/workspace/dataset/SKU110K_fixed/images/val_206.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_54.jpg')
# (14, '/workspace/dataset/SKU110K_fixed/images/val_239.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_220.jpg')
# (13, '/workspace/dataset/SKU110K_fixed/images/val_267.jpg')





# (26, '/workspace/dataset/SKU110K_fixed/images/val_516.jpg')
# (23, '/workspace/dataset/SKU110K_fixed/images/val_294.jpg')
# (27, '/workspace/dataset/SKU110K_fixed/images/val_296.jpg')
# (23, '/workspace/dataset/SKU110K_fixed/images/val_564.jpg')
# (20, '/workspace/dataset/SKU110K_fixed/images/val_298.jpg')
# (28, '/workspace/dataset/SKU110K_fixed/images/val_453.jpg')
# (22, '/workspace/dataset/SKU110K_fixed/images/val_444.jpg')
# (20, '/workspace/dataset/SKU110K_fixed/images/val_475.jpg')
# (20, '/workspace/dataset/SKU110K_fixed/images/val_398.jpg')
# (21, '/workspace/dataset/SKU110K_fixed/images/val_481.jpg')
# (26, '/workspace/dataset/SKU110K_fixed/images/val_160.jpg')
# (25, '/workspace/dataset/SKU110K_fixed/images/val_285.jpg')
# (24, '/workspace/dataset/SKU110K_fixed/images/val_18.jpg')
# (29, '/workspace/dataset/SKU110K_fixed/images/val_523.jpg')
# (24, '/workspace/dataset/SKU110K_fixed/images/val_130.jpg')
# (20, '/workspace/dataset/SKU110K_fixed/images/val_169.jpg')
# (28, '/workspace/dataset/SKU110K_fixed/images/val_425.jpg')
# (22, '/workspace/dataset/SKU110K_fixed/images/val_132.jpg')
# (25, '/workspace/dataset/SKU110K_fixed/images/val_436.jpg')
# (27, '/workspace/dataset/SKU110K_fixed/images/val_38.jpg')
# (29, '/workspace/dataset/SKU110K_fixed/images/val_357.jpg')
# (22, '/workspace/dataset/SKU110K_fixed/images/val_51.jpg')
# (26, '/workspace/dataset/SKU110K_fixed/images/val_15.jpg')
# (25, '/workspace/dataset/SKU110K_fixed/images/val_376.jpg')
# (26, '/workspace/dataset/SKU110K_fixed/images/val_456.jpg')
# (20, '/workspace/dataset/SKU110K_fixed/images/val_278.jpg')
# (25, '/workspace/dataset/SKU110K_fixed/images/val_327.jpg')
# (24, '/workspace/dataset/SKU110K_fixed/images/val_44.jpg')
# (22, '/workspace/dataset/SKU110K_fixed/images/val_462.jpg')
# (26, '/workspace/dataset/SKU110K_fixed/images/val_354.jpg')
# (23, '/workspace/dataset/SKU110K_fixed/images/val_34.jpg')
# (22, '/workspace/dataset/SKU110K_fixed/images/val_22.jpg')
# (28, '/workspace/dataset/SKU110K_fixed/images/val_350.jpg')
# (25, '/workspace/dataset/SKU110K_fixed/images/val_88.jpg')
# (23, '/workspace/dataset/SKU110K_fixed/images/val_464.jpg')
# (25, '/workspace/dataset/SKU110K_fixed/images/val_482.jpg')



# (69, '/workspace/dataset/SKU110K_fixed/images/val_95.jpg',
# (53, '/workspace/dataset/SKU110K_fixed/images/val_270.jpg',
# (35, '/workspace/dataset/SKU110K_fixed/images/val_504.jpg',
# (31, '/workspace/dataset/SKU110K_fixed/images/val_229.jpg',
# (47, '/workspace/dataset/SKU110K_fixed/images/val_212.jpg',
# (72, '/workspace/dataset/SKU110K_fixed/images/val_569.jpg',
# (38, '/workspace/dataset/SKU110K_fixed/images/val_271.jpg',
# (30, '/workspace/dataset/SKU110K_fixed/images/val_236.jpg',
# (31, '/workspace/dataset/SKU110K_fixed/images/val_488.jpg',
# (39, '/workspace/dataset/SKU110K_fixed/images/val_50.jpg',
# (41, '/workspace/dataset/SKU110K_fixed/images/val_224.jpg',
# (97, '/workspace/dataset/SKU110K_fixed/images/val_567.jpg',
# (85, '/workspace/dataset/SKU110K_fixed/images/val_447.jpg',
# (31, '/workspace/dataset/SKU110K_fixed/images/val_494.jpg',
# (34, '/workspace/dataset/SKU110K_fixed/images/val_150.jpg',
# (88, '/workspace/dataset/SKU110K_fixed/images/val_204.jpg',
# (75, '/workspace/dataset/SKU110K_fixed/images/val_284.jpg',
# (40, '/workspace/dataset/SKU110K_fixed/images/val_257.jpg',
# (40, '/workspace/dataset/SKU110K_fixed/images/val_251.jpg',
# (39, '/workspace/dataset/SKU110K_fixed/images/val_496.jpg',
# (30, '/workspace/dataset/SKU110K_fixed/images/val_389.jpg',
# (31, '/workspace/dataset/SKU110K_fixed/images/val_286.jpg',
# (30, '/workspace/dataset/SKU110K_fixed/images/val_200.jpg',
# (51, '/workspace/dataset/SKU110K_fixed/images/val_42.jpg',
# (61, '/workspace/dataset/SKU110K_fixed/images/val_352.jpg',
# (76, '/workspace/dataset/SKU110K_fixed/images/val_386.jpg',
# (37, '/workspace/dataset/SKU110K_fixed/images/val_279.jpg',
# (32, '/workspace/dataset/SKU110K_fixed/images/val_283.jpg',
# (86, '/workspace/dataset/SKU110K_fixed/images/val_473.jpg',
# (66, '/workspace/dataset/SKU110K_fixed/images/val_147.jpg',
# (36, '/workspace/dataset/SKU110K_fixed/images/val_261.jpg',
# (66, '/workspace/dataset/SKU110K_fixed/images/val_117.jpg',
# (41, '/workspace/dataset/SKU110K_fixed/images/val_346.jpg',
# (86, '/workspace/dataset/SKU110K_fixed/images/val_57.jpg',
