EXP_NAME=$1
THRESHOLD=$2
THRESHOLD_DECAY=$3
THRESHOLD_DECAY_STEPS=$4
LR=$5
LR_DECAY=$6
LR_DECAY_STEPS=$7
BATCH_SIZE=$8

DATA_DIR=/proj/BigLearning/ahjiang/datasets/iii-buses-noloss/shards
INCEPTION_MODEL_DIR=/proj/BigLearning/ahjiang/models
MODEL_PATH="${INCEPTION_MODEL_DIR}/inception-v3/model.ckpt-157585"

TBOARD_DIR="/proj/BigLearning/ahjiang/output/tmp/"$EXP_NAME

mkdir $TBOARD_DIR
TRAIN_DIR=$TBOARD_DIR"/bus_inception_t"$THRESHOLD"_tdecay"$THRESHOLD_DECAY"_tsteps"$THRESHOLD_DECAY_STEPS"_lr"$LR"_lrdecay"$LR_DECAY"_lrsteps"$LR_DECAY_STEPS"_b"$BATCH_SIZE
mkdir $TRAIN_DIR
rm -rf $TRAIN_DIR/*

bazel build //inception:bus_train
bazel build //inception:inception_train

bazel-bin/inception/bus_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --log_gradients=False \
  --input_queue_memory_factor=1 \
  --batch_size=$BATCH_SIZE \
  --initial_learning_rate=$LR \
  --learning_rate_decay_factor=$LR_DECAY \
  --learning_rate_decay_steps=$LR_DECAY_STEPS \
  --backprop_threshold=$THRESHOLD \
  --threshold_decay_factor=$THRESHOLD_DECAY \
  --threshold_decay_steps=$THRESHOLD_DECAY_STEPS\

