# Linear Probing Evaluations

pretrain_model=$1
if [ $# -gt 1 ]; then
        dim=$2
        echo 'Feature dimension:' $dim
else
        dim=0
        echo 'Baseline: no need to specify dimension'
fi

# Feature Extraction

echo starting feature extractions

echo extracting feature from $pretrain_model for CirCor dataset;
python src/benchmark/processing/circor_processing.py pretrain=$pretrain_model dim=$dim

echo extracting feature from $pretrain_model for PASCAL A;
python -u src/benchmark/processing/pascal_processing.py --pretrain $pretrain_model --dim $dim

echo extracting feature from $pretrain_model for PASCAL B;
python -u src/benchmark/processing/pascal_processing.py --dataset 'B' --pretrain $pretrain_model --dim $dim

echo extracting feature from $pretrain_model for PhysioNet 2016;
python -u src/benchmark/processing/physionet16_processing.py pretrain=$pretrain_model dim=$dim

echo extracting feature from $pretrain_model for ZCHSound clean;
python -u src/benchmark/processing/zchsound_processing.py pretrain=$pretrain_model dim=$dim

echo extracting feature from $pretrain_model for ZCHSound noisy;
python -u src/benchmark/processing/zchsound_processing.py data=noisy pretrain=$pretrain_model dim=$dim

# Linear Probing

echo starting linear probing evaluations 
python src/benchmark/linear_eval.py -m \
  task=circor_murmurs,circor_outcomes,pascal_A,pascal_B,physionet16,zchsound_clean,zchsound_clean_murmurs,zchsound_noisy,zchsound_noisy_murmurs \
  pretrain=$pretrain_model \
  dim=$dim