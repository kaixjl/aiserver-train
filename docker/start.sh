#! /bin/bash

# coco
CFG=ppyoloe_plus_crn_m_80e_coco
CFG_FILE=configs/ppyoloe/${CFG}.yml
OUT_WEIGHT_DIR=/weight/${CFG}
OUT_PD_DIR=output/${CFG}
PRETRAIN_WEIGHTS=/pretrain_weights/ppyoloe_crn_m_obj365_pretrained.pdparams

# voc
# CFG=ppyoloe_plus_crn_s_30e_voc
# CFG_FILE=configs/ppyoloe/voc/${CFG}.yml
# OUT_WEIGHT_DIR=/weight/${CFG}
# OUT_PD_DIR=output/${CFG}
# PRETRAIN_WEIGHTS=/pretrain_weights/ppyoloe_plus_crn_s_80e_coco.pdparams

l_train() {
    # with env
    python tools/train.py \
        -c ${CFG_FILE} \
        --eval \
        -o epoch=${HP_EPOCHES} \
           TrainReader.batch_size=${HP_BATCH_SIZE} \
           LearningRate.base_lr=${HP_LEARNING_RATE} && \
    python tools/export_model.py \
        -c ${CFG_FILE} \
        --output_dir=/weight && \
    mv ${OUT_WEIGHT_DIR}/* /weight && \
    rmdir ${OUT_WEIGHT_DIR} && \
    python /home/PaddleDetection/post_process_infer_cfg.py && \
    cp ${OUT_PD_DIR}/model_final.pdparams /weight

    # without env
    # python -m paddle.distributed.launch tools/train.py \
    #     -c ${CFG_FILE} \
    #     -o && \
    # python tools/export_model.py \
    #     -c ${CFG_FILE} \
    #     --output_dir=/weight && \
    # mv ${OUT_WEIGHT_DIR}/* /weight && \
    # rmdir ${OUT_WEIGHT_DIR} && \
    # python /home/PaddleDetection/post_process_infer_cfg.py && \
    # cp ${OUT_PD_DIR}/model_final.pdparams /weight
}

l_eval() {
    # with env
    python tools/eval.py \
        -c ${CFG_FILE} \
        -o epoch=${HP_EPOCHES} \
           TrainReader.batch_size=${HP_BATCH_SIZE} \
           LearningRate.base_lr=${HP_LEARNING_RATE} \
           weights=/weight/model_final.pdparams

    # without env
    # python tools/eval.py \
    #     -c ${CFG_FILE} \
    #     -o weights=/weight/model_final.pdparams && \
}

if [ ${MODEL_WORKING_MODE} -eq 1 ];
then
    l_train
elif [ ${MODEL_WORKING_MODE} -eq 2 ];
then
    l_eval
fi
