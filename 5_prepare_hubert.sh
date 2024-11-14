#!/bin/bash
stage=1
stop_stage=1000
datadir=/DB/LibriSpeech # dir to save data
tsvdir=data # dir to save tsv
combined_data=$tsvdir/combined_data
feat_dir=mfcc
km_path=kmeans_1st_mfcc
lab_dir=km_label_1st_mfcc
nshard=1
rank=0
n_cluster=100
simple_kemans_dir=/home/Workspace/fairseq_w2v/fairseq/examples/hubert/simple_kmeans

# name LibriSpeech datasets
data_sets_valid_test="dev-clean dev-other test-clean test-other"
data_sets_100h="train-clean-100 dev-clean dev-other test-clean test-other"
# data_sets_960h="train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other"
data_sets_960h="train valid test_other test_clean"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Prepare 960h Feature"
    for split in $data_sets_960h; do
        python ${simple_kemans_dir}/dump_mfcc_feature.py ${combined_data} ${split} ${nshard} ${rank} ${feat_dir}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Train Kmeans"
    python ${simple_kemans_dir}/learn_kmeans.py ${feat_dir} train ${nshard} ${km_path} ${n_cluster} --percent 0.1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Dump Kmeans Label"
    for split in $data_sets_960h; do
        python ${simple_kemans_dir}/dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Combine Kmeans Label"
    for split in $data_sets_960h; do
        for rank in $(seq 0 $((nshard - 1))); do
        cat $lab_dir/${split}_${rank}_${nshard}.km
        done > $lab_dir/${split}.km
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Create a dummy dict"
    for x in $(seq 0 $((n_cluster - 1))); do
        echo "$x 1"
    done >> $lab_dir/dict.km.txt
fi