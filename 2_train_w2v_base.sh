#!/bin/bash

fairseq-hydra-train \
    task.data=/home/Workspace/fairseq_w2v/fairseq/data/combined_data \
    --config-dir /home/Workspace/fairseq_w2v/fairseq/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech