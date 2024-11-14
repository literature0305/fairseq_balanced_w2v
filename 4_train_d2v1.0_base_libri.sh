#!/bin/bash

python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/audio/pretraining \
--config-name base_librispeech_3gpu task.data=/home/Workspace/fairseq_w2v/fairseq/data/combined_data common.user_dir=examples/data2vec