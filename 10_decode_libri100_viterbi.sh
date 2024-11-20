#!/bin/bash
stage=7
stop_stage=100

combined_data_dir=/home/litsub08/workspace/fairseq/data/combined_data
# subset=dev_clean
subsets="test_clean test_other dev_clean dev_other"
dir_checkpoint=outputs/2023-11-10/17-28-22/checkpoints/checkpoint_best.pt # outputs/2023-11-10/01-53-44/checkpoints/checkpoint_best.pt # outputs/2023-11-09/10-15-52/checkpoints/checkpoint_best.pt #outputs/2023-11-08/15-51-59/checkpoints/checkpoint_best.pt #outputs/2023-11-07/22-30-35/checkpoints/checkpoint_best.pt # outputs/2023-11-06/23-43-10/checkpoints/checkpoint_best.pt # multirun/2023-10-31/10-25-14/0/checkpoints/checkpoint_best.pt # outputs/2023-10-28/14-12-00/checkpoints/checkpoint_best.pt # outputs/2023-10-27/20-48-21/checkpoints/checkpoint_best.pt #outputs/2023-10-26/10-00-21/checkpoints/checkpoint_best.pt #
# outputs/2023-10-25/01-31-10/checkpoints/checkpoint_best.pt # outputs/2023-10-24/09-55-35/checkpoints/checkpoint_best.pt #outputs/2023-10-23/15-16-51/checkpoints/checkpoint_best.pt # outputs/2023-10-23/00-17-00/checkpoints/checkpoint_best.pt #outputs/2023-10-21/10-08-40/checkpoints/checkpoint_best.pt #outputs/2023-10-17/18-55-15/checkpoints/checkpoint_best.pt #outputs/2023-10-17/02-16-38/checkpoints/checkpoint_best.pt #outputs/2023-10-14/01-11-57/checkpoints/checkpoint_best.pt #outputs/2023-10-13/09-57-52/checkpoints/checkpoint_best.pt #outputs/2023-10-12/13-30-25/checkpoints/checkpoint_best.pt #outputs/2023-10-11/14-21-40/checkpoints/checkpoint_best.pt
# /home/litsub08/workspace/fairseq/outputs/2023-10-08/19-47-38/checkpoints/checkpoint_best.pt #errlog048_best.pt
# dir_checkpoint=outputs/2023-02-23/16-39-01/errlog005_sdt_kd/checkpoint_best.pt
results_path=/home/litsub08/workspace/fairseq/decode_results
SHELL_PATH=`pwd -P`
echo $SHELL_PATH

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: do decode (viterbi)"
    for subset in $subsets; do
        echo "Start decode $subset"
        python3 /home/litsub08/workspace/fairseq/examples/speech_recognition/infer.py $combined_data_dir --task audio_finetuning \
        --nbest 5 --path $dir_checkpoint --gen-subset $subset --results-path $results_path --w2l-decoder viterbi \
        --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
        --post-process letter
    done
fi

# if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
#     echo "stage 7: do decode (kenlm)"
#     for subset in $subsets; do
#         results_path=/home/Workspace/fairseq/decode_results_$subset
#         python /home/Workspace/fairseq/examples/speech_recognition/infer.py $combined_data_dir --task audio_finetuning \
#         --nbest 5 --path $dir_checkpoint --gen-subset $subset --results-path $results_path --w2l-decoder kenlm \
#         --lm-model libri_3gram.bin --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
#         --post-process letter
#     done
# fi
