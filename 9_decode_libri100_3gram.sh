#!/bin/bash
stage=7
stop_stage=100

combined_data_dir=/home/Workspace/fairseq/data/combined_data
# subset=dev_clean
subsets="test_clean test_other dev_clean dev_other"
dir_checkpoint=pretrained_models/errlog040_best.pt #errlog003.pt # /home/Workspace/fairseq/outputs/2023-02-16/01-15-13/errlog003_kd/checkpoint_best.pt
results_path=/home/Workspace/fairseq/decode_results
SHELL_PATH=`pwd -P`
echo $SHELL_PATH

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: do decode (viterbi)"
    for subset in $subsets; do
        echo "Start decode $subset"
        python3 examples/speech_recognition/infer.py $combined_data_dir --task audio_finetuning \
        --nbest 5 --path $dir_checkpoint --gen-subset $subset --results-path $results_path --w2l-decoder kenlm \
        --lm-model libri_3gram.bin --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
        --post-process letter --lexicon libri_lexicon.txt --beam-threshold 99999999999 --beam-size-token 100 --diverse-beam-strength 10
    done
fi
