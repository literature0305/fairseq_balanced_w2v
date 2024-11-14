#!/bin/bash
HYDRA_FULL_ERROR=1

path_fairseq=/home/Workspace/fairseq_w2v/fairseq
# path_data=$path_fairseq/data/combined_data_libri100_finetuning
path_data=$path_fairseq/data/combined_data_libri100
testset="test_clean test_other dev_clean dev_other"
# testset="test_clean"

cat data/test-clean/train.tsv > $path_data/test_clean.tsv
cat data/test-clean/labels.ltr > $path_data/test_clean.ltr
cat data/test-clean/labels.wrd > $path_data/test_clean.wrd

cat data/test-other/train.tsv > $path_data/test_other.tsv
cat data/test-other/labels.ltr > $path_data/test_other.ltr
cat data/test-other/labels.wrd > $path_data/test_other.wrd

cat data/dev-clean/train.tsv > $path_data/dev_clean.tsv
cat data/dev-clean/labels.ltr > $path_data/dev_clean.ltr
cat data/dev-clean/labels.wrd > $path_data/dev_clean.wrd

cat data/dev-other/train.tsv > $path_data/dev_other.tsv
cat data/dev-other/labels.ltr > $path_data/dev_other.ltr
cat data/dev-other/labels.wrd > $path_data/dev_other.wrd
mkdir -p $path_fairseq/results

for part in $testset; do
    python examples/speech_recognition/new/infer.py \
    --config-dir $path_fairseq/examples/hubert/config/decode \
    --config-name infer_viterbi \
    task.data=$path_data \
    task.normalize=true \
    common_eval.path=$path_fairseq/None/checkpoints/checkpoint_best.pt \
    dataset.gen_subset=$part
done

    # decoding.exp_dir=$path_fairseq/results \
