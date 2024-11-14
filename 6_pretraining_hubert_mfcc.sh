#!/bin/bash
stage=1
stop_stage=99
path_fairseq=/home/Workspace/fairseq_w2v/fairseq
path_data=$path_fairseq/data/combined_data/
path_label=$path_fairseq/km_label_1st_mfcc
n_cluster=100

# # check arguments
# if [ $# -ne 1 ]; then
#     echo "Error: $0 requires 1 argument <n_cluster>"
#     exit 1
# fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Create a dummy dict"
    rm -f $path_label/dict.km.txt
    for x in $(seq 0 $((n_cluster - 1))); do
        echo "$x 1"
    done >> $path_label/dict.km.txt
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Pretraining Hubert"
    CUDA_VISIBLE_DEVICES=0,1,2 python fairseq_cli/hydra_train.py \
      --config-dir $path_fairseq/examples/hubert/config/pretrain \
      --config-name hubert_base_librispeech_3gpu_mfcc \
      task.data=$path_data task.label_dir=$path_label task.labels='["km"]' model.label_rate=100
fi
