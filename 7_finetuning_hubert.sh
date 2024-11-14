#!/bin/bash
stage=3
stop_stage=100

path_fairseq=/home/Workspace/fairseq_w2v/fairseq
path_data=$path_fairseq/data/combined_data_libri100_finetuning
# path_label=$path_fairseq/km_label
path_label=$path_data
dir_trainset=data/train-clean-100
dir_dev_clean=data/dev-clean
dir_dev_other=data/dev-other
data_sets="train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Prepare letter labels"
    for part in $data_sets; do
        python examples/wav2vec/libri_labels.py $path_fairseq/data/$part/train.tsv --output-dir $path_fairseq/data/${part} --output-name labels
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: prepare Libri 100h finetuning"
    mkdir -p $path_data
    cat $dir_trainset/train.tsv > $path_data/train.tsv
    cat $dir_trainset/labels.ltr > $path_data/train.ltr
    cat $dir_trainset/labels.wrd > $path_data/train.wrd

    cat $dir_dev_clean/train.tsv > $path_data/dev_clean.tsv
    cat $dir_dev_clean/labels.ltr > $path_data/dev_clean.ltr
    cat $dir_dev_clean/labels.wrd > $path_data/dev_clean.wrd

    cat $dir_dev_other/train.tsv > $path_data/dev_other.tsv
    cat $dir_dev_other/labels.ltr > $path_data/dev_other.ltr
    cat $dir_dev_other/labels.wrd > $path_data/dev_other.wrd

    cat $path_data/dev_clean.tsv | head -1 | sed s/'\/dev-clean'/''/g > $path_data/valid.tsv

    cat $path_data/dev_clean.tsv | grep -v 'LibriSpeech' | sed s/"^"/'dev-clean\/'/g >> $path_data/valid.tsv
    cat $path_data/dev_clean.ltr > $path_data/valid.ltr
    cat $path_data/dev_clean.wrd > $path_data/valid.wrd

    cat $path_data/dev_other.tsv | grep -v 'LibriSpeech' | sed s/"^"/'dev-other\/'/g >> $path_data/valid.tsv
    cat $path_data/dev_other.ltr >> $path_data/valid.ltr
    cat $path_data/dev_other.wrd >> $path_data/valid.wrd

    # make dictionary
    cat 2-1_letter_dictionary > $path_data/dict.ltr.txt
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 python fairseq_cli/hydra_train.py \
    --config-dir $path_fairseq/examples/hubert/config/finetune \
    --config-name base_100h \
    task.data=$path_data task.label_dir=$path_label \
    model.w2v_path=$path_fairseq/None_errlog068_Hubert-MFCC-1st_blur+masking_layer0_0.1_3-21kernelsize/checkpoints/checkpoint_best.pt
fi

    # model.w2v_path=$path_fairseq/None_errlog065_Hubert-MFCC-1st_blur_100kernel-size/checkpoints/checkpoint_best.pt
    # model.w2v_path=$path_fairseq/None_errlog017_2nd_from-MFCC-reweight0.3_L9-km500/checkpoints/checkpoint_best.pt
    # model.w2v_path=$path_fairseq/None_errlog012_final/checkpoints/checkpoint_best.pt
    # model.w2v_path=$path_fairseq/None_errlog014-3_final/checkpoints/checkpoint_best.pt
    # model.w2v_path=$path_fairseq/pretrained_models/hubert_base_ls960.pt
    # model.w2v_path=$path_fairseq/hubert_run/checkpoints/checkpoint_best.pt
