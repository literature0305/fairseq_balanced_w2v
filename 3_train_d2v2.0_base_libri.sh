#!/bin/bash

python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name base_audio_only_task_3gpu task.data=/home/Workspace/fairseq_w2v/fairseq/data/combined_data