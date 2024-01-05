#! /bin/bash

ANNO_PATH=$(realpath -sL --relative-to=/dataset $(ls -1 /dataset/annotations/*.json | head -1))

l_verify_anno_path() {
    # if [ ! ${ANNO_PATH##*.} == "json" ]; then
    if [ -z ${ANNO_PATH} ]; then
        echo "Cannot find annotation file."
        exit
    fi
}

l_check_int() {
    return $(expr $1 "+" 10 &> /dev/null)
}

l_print_info() {
    echo "ANNO_PATH: ${ANNO_PATH}"
}

l_train() {
    # with env
    python tools/train.py \
        -c ${CFG_FILE} \
        --eval \
        -o epoch=${HP_EPOCHES} \
           pretrain_weights=${PRETRAIN_WEIGHTS} \
           TrainReader.batch_size=${HP_BATCH_SIZE} \
           LearningRate.base_lr=${HP_LEARNING_RATE} \
           TrainDataset.anno_path=${ANNO_PATH} \
           EvalDataset.anno_path=${ANNO_PATH} \
           TestDataset.anno_path=${ANNO_PATH} && \
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

    l_check_int $HP_BATCH_SIZE
    if [ ! $? -eq 0 ]; then
        echo "Batch size is not a number. Fallback to default value 1."
        HP_BATCH_SIZE=1
    fi

    python tools/eval.py \
        -c ${CFG_FILE} \
        -o epoch=${HP_EPOCHES} \
           EvalReader.batch_size=${HP_BATCH_SIZE} \
           LearningRate.base_lr=${HP_LEARNING_RATE} \
           weights=/weight/model_final.pdparams \
           TrainDataset.anno_path=${ANNO_PATH} \
           EvalDataset.anno_path=${ANNO_PATH} \
           TestDataset.anno_path=${ANNO_PATH}

    # without env
    # python tools/eval.py \
    #     -c ${CFG_FILE} \
    #     -o weights=/weight/model_final.pdparams && \
}

l_verify_anno_path

l_print_info

if [ ${MODEL_WORKING_MODE} -eq 1 ];
then
    echo "Running in Training mode."
    echo "===="
    l_train
elif [ ${MODEL_WORKING_MODE} -eq 2 ];
then
    echo "Running in Evaluation mode."
    echo "===="
    l_eval
fi