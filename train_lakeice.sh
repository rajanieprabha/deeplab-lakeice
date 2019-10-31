#!/usr/bin/env bash

cd ..
# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
LAKEICE_FOLDER="lake"
EXP_FOLDER="exp/moritz_cam1_1718"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${LAKEICE_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${LAKEICE_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${LAKEICE_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${LAKEICE_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${LAKEICE_FOLDER}/${EXP_FOLDER}/export"

#mkdir -p "${WORK_DIR}/${DATASET_DIR}/${LAKEICE_FOLDER}/exp"
#mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
# mkdir -p "${EXPORT_DIR}"

LAKEICE_DATASET="${WORK_DIR}/${DATASET_DIR}/${LAKEICE_FOLDER}/Moritz_cam1_17-18_tfrecord"

NUM_ITERATIONS=100000

#Saving hyperparameters
cp /home/pf/pfshare/data/MA_Rajanie/models/research/deeplab/train_lakeice.sh ${TRAIN_LOGDIR}/
cp /home/pf/pfshare/data/MA_Rajanie/models/research/deeplab/train.py ${TRAIN_LOGDIR}/
cp /home/pf/pfshare/data/MA_Rajanie/models/research/deeplab/utils/train_utils.py ${TRAIN_LOGDIR}/
cp /home/pf/pfshare/data/MA_Rajanie/models/research/deeplab/eval.py ${TRAIN_LOGDIR}/

##  Comments on changes done
## --model_variant="xception_skips" if you want to use the model with skips from encoder to decoder, "xception_65 otherwise
## --skips=1 if you want 4 extra skip concat layers from encoder to decoder, 0 otherwise
## check model.py file //not skips parameter is not working yet

python3 "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="val" \
  --model_variant="xception_65" \
  --skips=0 \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size="321,321" \
  --dataset="lake" \
  --train_batch_size=8 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=false \
  --train_logdir="${TRAIN_LOGDIR}" \
  --base_learning_rate=0.0001 \
  --learning_policy="poly" \
  --tf_initial_checkpoint="/home/pf/pfshare/data/MA_Rajanie/pretrained/deeplabv3_pascal_trainval/model.ckpt" \
  --dataset_dir="${LAKEICE_DATASET}"

#  --tf_initial_checkpoint="/home/pf/pfshare/data/MA_Rajanie/models/research/deeplab/datasets/lake/init_models/deeplabv3_pascal_train_aug/model.ckpt" \
# /home/pf/pfshare/data/MA_Rajanie/models/research/deeplab/datasets/lake/exp/Train_Cam1_val_Cam0_weighted/train/model.ckpt-30000" \
#   --skips=1 \
#   --tf_initial_checkpoint="/home/pf/pfshare/data/MA_Rajanie/resnet_v1_101/model.ckpt" \
#     --top_k_percent_pixels=1.0 \
# --hard_example_mining_step=0 \

 
# python3 "${WORK_DIR}"/eval.py \
#   --logtostderr \
#   --eval_split="train" \
#   --model_variant="xception_65" \
#   --skips=0 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --atrous_rates=24 \
#   --output_stride=8 \
#   --decoder_output_stride=4 \
#   --eval_crop_size="325,1210" \
#   --dataset="lake" \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --eval_logdir="/home/pf/pfshare/data/MA_Rajanie/models/research/deeplab/datasets/lake/exp/PTZ_and_nonPTZ_cam1/Test/Moritz_cam0_16-17/eval" \
#   --dataset_dir="${LAKEICE_DATASET}" \
#   --max_number_of_evaluations=1

# # Visualize the results.
# python3 "${WORK_DIR}"/vis.py \
#   --logtostderr \
#   --vis_split="train" \
#   --model_variant="xception_65" \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --atrous_rates=24 \
#   --output_stride=8 \
#   --decoder_output_stride=4 \
#   --vis_crop_size="325,1210" \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --vis_logdir="/home/pf/pfshare/data/MA_Rajanie/models/research/deeplab/datasets/lake/exp/PTZ_and_nonPTZ_cam1/Test/Moritz_cam0_16-17/vis" \
#   --logits_file="/home/pf/pfshare/data/MA_Rajanie/models/research/deeplab/datasets/lake/exp/PTZ_and_nonPTZ_cam1/Test/Moritz_cam0_16-17/logits" \
#   --dataset="lake" \
#   --dataset_dir="${LAKEICE_DATASET}" \
#   --colormap_type="lake" \
#   --max_number_of_iterations=1

#325,1210" \
#344,421
##Export the trained checkpoint.
# CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
# EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

# python3 "${WORK_DIR}"/export_model.py \
#   --logtostderr \
#   --checkpoint_path="${CKPT_PATH}" \
#   --export_path="${EXPORT_PATH}" \
#   --model_variant="xception_65_skips" \
#   --atrous_rates=12 \
#   --atrous_rates=24 \
#   --atrous_rates=36 \
#   --output_stride=8 \
#   --decoder_output_stride=4 \
#   --num_classes=5 \
#   --inference_scales=1.0

 #######################################
#Lake Detection params
#python3 "${WORK_DIR}"/train.py \
#  --logtostderr \
#  --train_split="trainval" \
#  --model_variant="xception_65" \
#  --atrous_rates=6 \
#  --atrous_rates=12 \
#  --atrous_rates=18 \
#  --output_stride=16 \
#  --decoder_output_stride=4 \
#  --train_crop_size="321,321" \
#  --dataset="lake" \
#  --train_batch_size=8 \
#  --training_number_of_steps="${NUM_ITERATIONS}" \
#  --fine_tune_batch_norm=false \
#  --train_logdir="${TRAIN_LOGDIR}" \
#  --base_learning_rate=9e-5 \
#  --learning_policy="poly" \
#  --tf_initial_checkpoint="/home/pf/pfshare/data/MA_Rajanie/models/research/deeplab/datasets/lake/init_models/deeplabv3_pascal_train_aug/model.ckpt" \
#  --dataset_dir="${LAKEICE_DATASET}"
