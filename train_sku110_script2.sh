


# model_type=ppyoloe # can modify to 'yolov7'

# job_name=ppyoloe_plus_crn_l_80e_coco # can modify to 'yolov7_l_300e_coco'
# config=sku110_configs/${job_name}.yml


# log_dir=sku110_logdir/${job_name}
# # weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
# weights=sku110_output/${job_name}/model_final.pdparams
# # CUDA_VISIBLE_DEVICES=0 python3.7 tools/train.py -c ${config} --eval --amp


# python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp





# model_type=ppyoloe # can modify to 'yolov7'

# job_name=ppyoloe_plus_crn_s_20e_coco # can modify to 'yolov7_l_300e_coco'
# config=sku110_configs/ppyoloe_plus_crn_s_20e_coco.yml
# log_dir=sku110_logdir/ppyoloe_plus_crn_s_20e_coco
# # weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
# weights=sku110_output/ppyoloe_plus_crn_s_20e_coco/model_final.pdparams
# # CUDA_VISIBLE_DEVICES=0 python3.7 tools/train.py -c ${config} --eval --amp



# python3.7 -m paddle.distributed.launch --log_dir=sku110_logdir/ppyoloe_plus_crn_s_20e_coco --gpus 0,1,2,3,4,5,6,7 tools/train.py -c sku110_configs/ppyoloe_plus_crn_s_20e_coco.yml --eval --amp
# python3.7 -m paddle.distributed.launch --log_dir=sku110_logdir/ppyoloe_plus_crn_s_20e_coco_wo_amp --gpus 0,1,2,3,4,5,6,7 tools/train.py -c sku110_configs/ppyoloe_plus_crn_s_20e_coco_wo_amp.yml --eval
# python3.7 -m paddle.distributed.launch --log_dir=sku110_logdir/ppyoloe_plus_crn_x_80e_coco --gpus 4,5,6,7 tools/train.py -c sku110_configs/ppyoloe_plus_crn_x_80e_coco.yml --eval


# python3.7 -m paddle.distributed.launch --log_dir=sku110_logdir/ppyoloe_plus_crn_s_20e_coco_alpha --gpus 4,5,6,7 tools/train.py -c sku110_configs/ppyoloe_plus_crn_s_20e_coco_alpha.yml --eval
# python3.7 -m paddle.distributed.launch --log_dir=sku110_logdir/ppyoloe_plus_crn_s_20e_coco_probiou --gpus 4,5,6,7 tools/train.py -c sku110_configs/ppyoloe_plus_crn_s_20e_coco_probiou.yml --eval




CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py \
    -c sku110_configs/ppyoloe_plus_crn_s_20e_coco.yml \
    -o weights=./sku110_output/ppyoloe_plus_crn_s_20e_coco_wo_amp/model_final.pdparams --classwise > ppyoloe_plus_crn_s_20e_coco_wo_amp.txt


# weights=ppyoloe_crn_s_400e_coco.pdparams

# CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py \
#     -c sku110_configs/ppyoloe_plus_crn_s_80e_coco.yml \
#     -o weights=./sku110_output/ppyoloe_plus_crn_s_80e_coco/model_final.pdparams --classwise > ppyoloe_plus_crn_s_80e_coco.txt

CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py \
    -c sku110_configs/ppyoloe_plus_crn_s_80e_coco.yml \
    -o weights=./sku110_output/ppyoloe_plus_crn_s_80e_coco/model_final.pdparams --classwise > ppyoloe_plus_crn_s_80e_coco.txt

CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py \
    -c sku110_configs/ppyoloe_plus_crn_m_80e_coco.yml \
    -o weights=./sku110_output/ppyoloe_plus_crn_m_80e_coco/model_final.pdparams --classwise > ppyoloe_plus_crn_m_80e_coco.txt

CUDA_VISIBLE_DEVICES=6 python3.7 tools/eval.py \
    -c sku110_configs/ppyoloe_plus_crn_l_80e_coco.yml \
    -o weights=./sku110_output/ppyoloe_plus_crn_l_80e_coco/model_final.pdparams --classwise > ppyoloe_plus_crn_l_80e_coco.txt

CUDA_VISIBLE_DEVICES=6 python3.7 tools/eval.py \
    -c sku110_configs/ppyoloe_plus_crn_x_80e_coco.yml \
    -o weights=./sku110_output/ppyoloe_plus_crn_x_80e_coco/model_final.pdparams --classwise > ppyoloe_plus_crn_x_80e_coco.txt


CUDA_VISIBLE_DEVICES=6 python3.7 tools/eval.py -c sku110_configs/ppyoloe_plus_crn_s_20e_coco_alpha.yml -o weights=./sku110_output/ppyoloe_plus_crn_s_20e_coco_alpha/model_final.pdparams --classwise > ppyoloe_plus_crn_s_20e_coco_alpha.txt


# CUDA_VISIBLE_DEVICES=6 python3.7 tools/eval.py \
#     -c sku110_configs/ppyoloe_plus_crn_x_80e_coco.yml \
#     -o weights=./sku110_output/ppyoloe_plus_crn_x_80e_coco/model_final.pdparams --classwise > ppyoloe_plus_crn_x_80e_coco.txt



# CUDA_VISIBLE_DEVICES=2 python3.7 tools/infer.py \
#     -c sku110_configs/ppyoloe_plus_crn_s_80e_coco.yml \
#     -o weights=./sku110_output/ppyoloe_plus_crn_s_80e_coco/model_final.pdparams \
#     --infer_dir=/home/ws/dataset/SKU110K_fixed/val_images \
#     --output_dir ./sku110_vis_0_1/ppyoloe_plus_crn_s_80e_coco --draw_threshold=0.1

# CUDA_VISIBLE_DEVICES=2 python3.7 tools/infer.py \
#     -c sku110_configs/ppyoloe_plus_crn_m_80e_coco.yml \
#     -o weights=./sku110_output/ppyoloe_plus_crn_m_80e_coco/model_final.pdparams \
#     --infer_dir=/home/ws/dataset/SKU110K_fixed/val_images \
#     --output_dir ./sku110_vis_0_1/ppyoloe_plus_crn_m_80e_coco --draw_threshold=0.1

CUDA_VISIBLE_DEVICES=5 python3.7 tools/infer.py \
    -c sku110_configs/ppyoloe_plus_crn_s_80e_coco_nms_0_1.yml \
    -o weights=./sku110_output/ppyoloe_plus_crn_s_80e_coco/model_final.pdparams \
    --infer_dir=/home/ws/dataset/SKU110K_fixed/val_images \
    --output_dir ./sku110_vis_0_1/ppyoloe_plus_crn_s_80e_coco_nms_0_1 --draw_threshold=0.1


CUDA_VISIBLE_DEVICES=4 python3.7 tools/debug_small_target.py \
    -c sku110_configs/ppyoloe_plus_crn_x_80e_coco.yml \
    --img_anno_filepath /home/ws/dataset/SKU110K_fixed/json_annotations/annotations_val.json \
    -o weights=./sku110_output/ppyoloe_plus_crn_x_80e_coco/model_final.pdparams \
    --infer_dir=/home/ws/dataset/SKU110K_fixed/val_images \
    --output_dir ./sku110_vis_0_1/ppyoloe_plus_crn_x_80e_coco --draw_threshold=0.1





# python3.7 tools/vis_gt.py -c sku110_configs/ppyoloe_plus_crn_m_80e_coco.yml --img_dir /home/ws/dataset/SKU110K_fixed/val_images --img_anno_filepath /home/ws/dataset/SKU110K_fixed/json_annotations/annotations_val.json --output_dir /home/ws/dataset/SKU110K_fixed/val_vis





# CUDA_VISIBLE_DEVICES=0 python3.7 tools/export_model.py -c ${config} -o weights=${weights} # exclude_nms=True trt=True

# CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx
