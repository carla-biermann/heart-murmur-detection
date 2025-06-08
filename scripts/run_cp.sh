#!/bin/bash

# Continued Pre-Training of Audio-MAE and OPERA-CT

# Prepare data
datasets=(
    "CirCor feature/circor_eval 8"
    "PASCAL_A feature/pascal_A_eval 2"
    "PASCAL_B feature/pascal_B_eval 2"
    "PhysioNet_2016 feature/physionet16_eval 8"
    "ZCHSound_clean" feature/zchsound_clean_eval 8"
    "ZCHSound_noisy" feature/zchsound_noisy_eval 8"
)

# Loop through datasets
for entry in "${datasets[@]}"; do
    read -r dataset_name feature_dir input_sec <<< "$entry"
    echo "Preparing data for $dataset_name dataset"
    python src/pretrain/prepare_data/heart_pressl.py --feature_dir "$feature_dir" --input_sec "$input_sec"
done


# # CP OPERA-CT

# # OPERA-CT CP In-corpus

# python src/pretrain/cola_training.py data=multiple\
#         circor=True\
#         pretrain=operaCT\
#         encoder=htsat\
#         title=operaCT-heart-indomain-circor\
#         epoches=250

# python src/pretrain/cola_training.py data=multiple\
#         physionet16=True\
#         pretrain=operaCT\
#         encoder=htsat\
#         title=operaCT-heart-indomain-physionet16\
#         epoches=250

# # OPERA-CT CP Cross-corpus

# python src/pretrain/cola_training.py data=multiple\
#         pascal_A=True\
#         pascal_B=True\
#         physionet16=True\
#         zchsound_clean=True\
#         zchsound_noisy=True\
#         pretrain=operaCT\
#         encoder=htsat\
#         title=operaCT-heart-nocircor\
#         epoches=250

# python src/pretrain/cola_training.py data=multiple\
#         circor=True\
#         physionet16=True\
#         zchsound_clean=True\
#         zchsound_noisy=True\
#         pretrain=operaCT\
#         encoder=htsat\
#         title=operaCT-heart-nopascal\
#         epoches=250

# python src/pretrain/cola_training.py data=multiple\
#         circor=True\
#         pascal_A=True\
#         pascal_B=True\
#         zchsound_clean=True\
#         zchsound_noisy=True\
#         pretrain=operaCT\
#         encoder=htsat\
#         title=operaCT-heart-nophysionet16\
#         epoches=250

# python src/pretrain/cola_training.py data=multiple\
#         circor=True\
#         pascal_A=True\
#         pascal_B=True\
#         physionet16=True\
#         pretrain=operaCT\
#         encoder=htsat\
#         title=operaCT-heart-nozchsound\
#         epoches=250

# # OPERA-CT CP all-corpora

# python src/pretrain/cola_training.py data=multiple\
#         circor=True\
#         pascal_A=True\
#         pascal_B=True\
#         physionet16=True\
#         zchsound_clean=True\
#         zchsound_noisy=True\
#         pretrain=operaCT\
#         encoder=htsat\
#         title=operaCT-heart-all\
#         epoches=250

# # OPERA-CT PT from scratch all-corpora

# python src/pretrain/cola_training.py data=multiple\
#         circor=True\
#         pascal_A=True\
#         pascal_B=True\
#         physionet16=True\
#         zchsound_clean=True\
#         zchsound_noisy=True\
#         encoder=htsat\
#         title=operaCT-heart-all-scratch\
#         epoches=250


# # CP Audio-MAE

# # Audio-MAE CP In-corpus

# python src/pretrain/mae_training.py data=multiple\
#         circor=True\
#         pretrain=audiomae\
#         method=audiomae\
#         encoder=vit\
#         title=audiomae-heart-indomain-circor\
#         epoches=250

# python src/pretrain/mae_training.py data=multiple\
#         physionet16=True\
#         pretrain=audiomae\
#         method=audiomae\
#         encoder=vit\
#         title=audiomae-heart-indomain-physionet16\
#         epoches=250

# # Audio-MAE CP Cross-corpus

# python src/pretrain/mae_training.py data=multiple\
#         pascal_A=True\
#         pascal_B=True\
#         physionet16=True\
#         zchsound_clean=True\
#         zchsound_noisy=True\
#         pretrain=audiomae\
#         method=audiomae\
#         encoder=vit\
#         title=audiomae-heart-nocircor\
#         epoches=250

# python src/pretrain/mae_training.py data=multiple\
#         circor=True\
#         physionet16=True\
#         zchsound_clean=True\
#         zchsound_noisy=True\
#         pretrain=audiomae\
#         method=audiomae\
#         encoder=vit\
#         title=audiomae-heart-nopascal\
#         epoches=250

# python src/pretrain/mae_training.py data=multiple\
#         circor=True\
#         pascal_A=True\
#         pascal_B=True\
#         zchsound_clean=True\
#         zchsound_noisy=True\
#         pretrain=audiomae\
#         method=audiomae\
#         encoder=vit\
#         title=audiomae-heart-nophysionet16\
#         epoches=250

# python src/pretrain/mae_training.py data=multiple\
#         circor=True\
#         pascal_A=True\
#         pascal_B=True\
#         physionet16=True\
#         pretrain=audiomae\
#         method=audiomae\
#         encoder=vit\
#         title=audiomae-heart-nozchsound\
#         epoches=250

# # Audio-MAE CP all-corpora

# python src/pretrain/mae_training.py data=multiple\
#         circor=True\
#         pascal_A=True\
#         pascal_B=True\
#         physionet16=True\
#         zchsound_clean=True\
#         zchsound_noisy=True\
#         pretrain=audiomae\
#         method=audiomae\
#         encoder=vit\
#         title=audiomae-heart-all\
#         epoches=250

# # Audio-MAE PT from scratch all-corpora

# python src/pretrain/mae_training.py data=multiple\
#         circor=True\
#         pascal_A=True\
#         pascal_B=True\
#         physionet16=True\
#         zchsound_clean=True\
#         zchsound_noisy=True\
#         method=audiomae\
#         encoder=vit\
#         title=audiomae-heart-all-scratch\
#         epoches=250