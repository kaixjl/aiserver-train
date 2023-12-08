#!/bin/bash

# IMAGE_TAG=34d769ea094a
IMAGE_TAG=aiserver-train
docker run --gpus all -it --rm \
    -e MODEL_WORKING_MODE=2 \
    -e HP_EPOCHES=80 \
    -e HP_BATCH_SIZE=2 \
    -e HP_LEARNING_RATE=0.0001 \
    -v /home/xiuhx/projects/aiserver-train/output/weight:/weight \
    -v /home/xiuhx/projects/aiserver-train/datasethelmet460/COCO:/dataset \
    -v /home/xiuhx/projects/aiserver-train/docker.local.d/pretrain_weights:/pretrain_weights \
    ${IMAGE_TAG}

#   -e ENABLE_ALGORITHMS='"fakealarm"' \
# print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)