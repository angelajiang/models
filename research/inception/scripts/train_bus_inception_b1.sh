EXP_NAME=$1
THRESHOLD=$2
THRESHOLD_DECAY=$3
LR=$4

DATA_DIR=/proj/BigLearning/ahjiang/datasets/iii-buses-noloss/shards
INCEPTION_MODEL_DIR=/proj/BigLearning/ahjiang/models
MODEL_PATH="${INCEPTION_MODEL_DIR}/inception-v3/model.ckpt-157585"

mkdir /proj/BigLearning/ahjiang/output/tmp/$EXP_NAME/
TRAIN_DIR=/proj/BigLearning/ahjiang/output/tmp/$EXP_NAME/bus_inception_$THRESHOLD"_tdecay"$THRESHOLD_DECAY"_lr"$LR
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
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1 \
  --batch_size=1 \
  --backprop_threshold=$THRESHOLD \
  --threshold_decay_factor=$THRESHOLD_DECAY \
  --initial_learning_rate=$LR \

