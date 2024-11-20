#!/bin/bash
stage=7
stop_stage=100

combined_data_dir=/home/litsub08/workspace/fairseq/data/combined_data
# subset=dev_clean
subsets="test_clean test_other dev_clean dev_other"
dir_checkpoint=/home/litsub08/workspace/fairseq/errlog048_best.pt
results_path=/home/litsub08/workspace/fairseq/decode_results
SHELL_PATH=`pwd -P`
echo $SHELL_PATH

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: do decode (4gram)"
    for subset in $subsets; do
        echo "Start decode $subset"
        python3 examples/speech_recognition/infer.py $combined_data_dir --task audio_finetuning \
        --nbest 1 --path $dir_checkpoint --gen-subset $subset --results-path $results_path --w2l-decoder kenlm \
        --lm-model libri_4gram.bin --lm-weight 2.15 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
        --post-process letter --lexicon libri_lexicon.txt --beam 1500
    done
fi # beam-size: 1500

# $subset=dev_other
# python examples/speech_recognition/infer.py /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw --task audio_finetuning \
# --nbest 1 --path /path/to/model --gen-subset $subset --results-path /path/to/save/results/for/sclite --w2l-decoder kenlm \
# --lm-model /path/to/kenlm.bin --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
# --post-process letter
