# Fine-tuning Evaluations

pretrain_model=$1
if [ $# -gt 1 ]; then
        dim=$2
        echo 'Feature dimension:' $dim
else
        echo 'Error: Dimension must be specified'
        exit 1
fi

# Fine-tuning

echo starting fine-tuning
python src/benchmark/other_eval/finetuning.py -m \
  task=circor_murmurs,circor_outcomes,pascal_A,pascal_B,physionet16,zchsound_clean,zchsound_clean_murmurs,zchsound_noisy,zchsound_noisy_murmurs \
  pretrain=$pretrain_model \
  dim=$dim