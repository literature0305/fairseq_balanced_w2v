#!/bin/bash
stage=7
stop_stage=100

combined_data_dir=/home/Workspace/fairseq_balanced_d2v/data/combined_data
# subset=dev_clean
subsets="test_clean test_other dev_clean dev_other"
# dir_checkpoint=/home/Workspace/fairseq_balanced_d2v/outputs/2023-01-12/11-34-08/checkpoint_w2v_kd/checkpoint_best.pt # libri100 kd-sa
# dir_checkpoint=/home/Workspace/fairseq_balanced_d2v/outputs/2023-02-16/01-15-13/errlog003_kd/checkpoint_best.pt # libri100 kd-sa again
dir_checkpoint=/home/Workspace/fairseq_balanced_d2v/errlog003_finetuned-best.pt # multirun/2023-11-02/01-09-05/0/checkpoints/checkpoint_best.pt #multirun/2023-10-31/10-25-14/0/checkpoints/checkpoint_best.pt # pretrained_models/errlog005.pt

# dir_checkpoint=/home/Workspace/fairseq_balanced_d2v/pretrained_models/base_libri_960h.pt 

dir_lm_4gram=/home/Workspace/fairseq_balanced_d2v/libri_4gram.bin
dir_lm_3gram=/home/Workspace/fairseq_balanced_d2v/libri_3gram.bin
dir_lexicon=/home/Workspace/fairseq_balanced_d2v/libri_lexicon.txt
SHELL_PATH=`pwd -P`
echo $SHELL_PATH

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    num_gpus=1
    lmweight=1.45
    wordscore=-0.65
    silscore=0.0

    echo "stage 6: do decode (no lm)"
    for subset in $subsets; do
        echo "Start decode no lm $subset"
        results_path=/home/Workspace/fairseq_balanced_d2v/decode_results_${subset}_no_lm
        mkdir -p $results_path
        python3 examples/speech_recognition/new/infer.py --config-dir examples/speech_recognition/new/conf \
        --config-name infer_4gpu task=audio_finetuning task.data=$combined_data_dir common.user_dir=examples/data2vec \
        task.labels=ltr decoding.type=viterbi \
        decoding.lmweight=${lmweight} decoding.wordscore=${wordscore} decoding.silweight=${silscore} \
        decoding.lexicon=$dir_lexicon \
        decoding.unique_wer_file=True \
        dataset.gen_subset=$subset \
        decoding.results_path=$results_path \
        common_eval.path=$dir_checkpoint decoding.beam=1500 distributed_training.distributed_world_size=${num_gpus}
    done

    echo "stage 7: do decode (3gram)"
    for subset in $subsets; do
        echo "Start decode 3gram $subset"
        results_path=/home/Workspace/fairseq_balanced_d2v/decode_results_$subset
        mkdir -p $results_path
        python3 examples/speech_recognition/new/infer.py --config-dir examples/speech_recognition/new/conf \
        --config-name infer_4gpu task=audio_finetuning task.data=$combined_data_dir common.user_dir=examples/data2vec \
        task.labels=ltr decoding.type=kenlm \
        decoding.lmweight=${lmweight} decoding.wordscore=${wordscore} decoding.silweight=${silscore} \
        decoding.lexicon=$dir_lexicon \
        decoding.lmpath=$dir_lm_3gram decoding.unique_wer_file=True \
        dataset.gen_subset=$subset \
        decoding.results_path=$results_path \
        common_eval.path=$dir_checkpoint decoding.beam=1500 distributed_training.distributed_world_size=${num_gpus}
    done

    echo "stage 7: do decode (4gram)"
    for subset in $subsets; do
        echo "Start decode 4gram $subset"
        results_path=/home/Workspace/fairseq_balanced_d2v/decode_results_${subset}_4gram
        mkdir -p ${results_path}
        python3 examples/speech_recognition/new/infer.py --config-dir examples/speech_recognition/new/conf \
        --config-name infer_4gpu task=audio_finetuning task.data=$combined_data_dir common.user_dir=examples/data2vec \
        task.labels=ltr decoding.type=kenlm \
        decoding.lmweight=${lmweight} decoding.wordscore=${wordscore} decoding.silweight=${silscore} \
        decoding.lexicon=$dir_lexicon \
        decoding.lmpath=$dir_lm_4gram decoding.unique_wer_file=True \
        dataset.gen_subset=$subset \
        decoding.results_path=$results_path \
        common_eval.path=$dir_checkpoint decoding.beam=1500 distributed_training.distributed_world_size=${num_gpus}  
    done

    # echo "stage 7: do decode (4gram)"
    # for subset in $subsets; do
    # done
fi

# for subset in $subset #"dev-clean" "dev-other" "test-clean" "test-other" 
# do
#     echo "====================   $model // $subset   ===================="
#     python /Workspace/fairseq_balanced_d2v/examples/speech_recognition/new/infer.py \
#         —config-dir /Workspace/fairseq_balanced_d2v/examples/speech_recognition/new/conf \
#         —config-name infer \
#         task=audio_finetuning \
#         task.data=/Workspace/LibriSpeech/manifests \
#         task.labels=ltr \
#         decoding.type=fairseq_balanced_d2vlm \
#         decoding.lmweight=0.0 decoding.wordscore=0. decoding.silweight=0 \
#         decoding.lmpath=/Workspace/models/lm_model/lm_librispeech_word_transformer.pt \
#         decoding.lexicon=/Workspace/models/lm_model/librispeech_lexicon_lower.lst \
#         decoding.unique_wer_file=false \
#         dataset.gen_subset=$subset \
#         common_eval.path=/Workspace/models/wav2vec_model/wav2vec_small_960h.pt \
#         common_eval.quiet=false \
#         decoding.beam=5 \
#         distributed_training.distributed_world_size=1
# done
