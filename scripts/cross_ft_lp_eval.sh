# Linear Probing Evaluation of Models Fine-tuned on other tasks.

pretrain_model="operaCT"

# Pick seeds with best performance out of the 5 FT seeds.

pairs=(
  "circor_murmurs [insert_ckpt_path] 0" # Seed 0
  "circor_outcomes [insert_ckpt_path] 0" # Seed 0
  "pascal_A [insert_ckpt_path] 3" # Seed 3
  "pascal_B [insert_ckpt_path] 2" # Seed 2
  "physionet16 [insert_ckpt_path] 0" # Seed 0
  "zchsound_clean [insert_ckpt_path] 3" # Seed 3
  "zchsound_clean_murmurs [insert_ckpt_path] 2" # Seed 2
  "zchsound_noisy [insert_ckpt_path] 2" # Seed 2
  "zchsound_noisy_murmurs [insert_ckpt_path] 3" # Seed 3
)

echo starting feature extractions
# Iterate over each pair
for pair in "${pairs[@]}"; do
    # Split pair into array elements
    read -r fine_tuned ckpt_path seed <<< "$entry"

    python src/benchmark/processing/circor_processing.py pretrain=$pretrain_model dim=$dim seed="$seed" fine_tuned="$fine_tuned" ckpt_path="$ckpt_path"
    python -u src/benchmark/processing/pascal_processing.py --pretrain $pretrain_model --dim $dim --fine_tuned "$fine_tuned" --ckpt_path "$ckpt_path"
    python -u src/benchmark/processing/pascal_processing.py --dataset 'B' --pretrain $pretrain_model --dim $dim --fine_tuned "$fine_tuned"  --ckpt_path "$ckpt_path"
    python src/benchmark/processing/physionet16_processing.py pretrain=$pretrain_model dim=$dim seed="$seed" fine_tuned="$fine_tuned" ckpt_path="$ckpt_path"
    python src/benchmark/processing/zchsound_processing.py pretrain=$pretrain_model dim=$dim seed="$seed" fine_tuned="$fine_tuned" ckpt_path="$ckpt_path"
    python src/benchmark/processing/zchsound_processing.py data=noisy pretrain=$pretrain_model dim=$dim seed="$seed" fine_tuned="$fine_tuned" ckpt_path="$ckpt_path"

done


# Linear Probing

echo starting linear probing evaluations 
python src/benchmark/linear_eval.py -m \
  task=circor_murmurs,circor_outcomes,pascal_A,pascal_B,physionet16, \
       zchsound_clean,zchsound_clean_murmurs,zchsound_noisy,zchsound_noisy_murmurs \
  pretrain=operaCT768_finetuned_circor_murmurs_0,operaCT768_finetuned_circor_outcomes_0, \
           operaCT768_finetuned_pascal_A_3,operaCT768_finetuned_pascal_B_2, \
           operaCT768_finetuned_physionet16_0,operaCT768_finetuned_zchsound_clean_3, \
           operaCT768_finetuned_zchsound_clean_murmurs_2,operaCT768_finetuned_zchsound_noisy_2, \
           operaCT768_finetuned_zchsound_noisy_murmurs_3 \
  dim=768
