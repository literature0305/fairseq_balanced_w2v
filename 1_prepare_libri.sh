#!/bin/bash
stage=3
stop_stage=1000
datadir=/DB/LibriSpeech # dir to save data
tsvdir=data # dir to save tsv
combined_dir=$tsvdir/combined_data

# name LibriSpeech datasets
data_sets="train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other"

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Prepare tsv"
    for part in $data_sets; do
        python examples/wav2vec/wav2vec_manifest.py $datadir/$part --dest $tsvdir/$part --ext flac --valid-percent 0
        # mv $tsvdir/$part/train.tsv $tsvdir/$part/wav_dir.tsv
    done
fi

rm -rf $combined_dir
mkdir -p $combined_dir

echo '/DB/LibriSpeech' > $combined_dir/train.tsv
cat $tsvdir/train-clean-100/train.tsv | grep -v 'LibriSpeech' | sed s/"^"/'train-clean-100\/'/g >> $combined_dir/train.tsv
cat $tsvdir/train-clean-360/train.tsv | grep -v 'LibriSpeech' | sed s/"^"/'train-clean-360\/'/g >> $combined_dir/train.tsv
cat $tsvdir/train-other-500/train.tsv | grep -v 'LibriSpeech' | sed s/"^"/'train-other-500\/'/g >> $combined_dir/train.tsv

echo '/DB/LibriSpeech' > $combined_dir/valid.tsv
cat $tsvdir/dev-clean/train.tsv | grep -v 'LibriSpeech' | sed s/"^"/'dev-clean\/'/g >> $combined_dir/valid.tsv
cat $tsvdir/dev-other/train.tsv | grep -v 'LibriSpeech' | sed s/"^"/'dev-other\/'/g >> $combined_dir/valid.tsv

cat $tsvdir/test-other/train.tsv > $combined_dir/test_other.tsv
cat $tsvdir/test-clean/train.tsv > $combined_dir/test_clean.tsv

# it works (it makes all.tsv)
# python examples/wav2vec/wav2vec_manifest.py $datadir --dest $tsvdir --ext flac --valid-percent 0

