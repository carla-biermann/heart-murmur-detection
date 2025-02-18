model_name="operaCT"
dim=768
python src/benchmark/processing/zchsound_processing.py --pretrain $model_name --dim $dim
#python src/benchmark/linear_eval.py --task copd --pretrain $model_name --dim $dim

model_name="operaCE"
dim=1280
python src/benchmark/processing/zchsound_processing.py --pretrain $model_name --dim $dim
#python src/benchmark/linear_eval.py --task copd --pretrain $model_name --dim $dim

model_name="operaGT"
dim=384
python src/benchmark/processing/zchsound_processing.py --pretrain $model_name --dim $dim
#python src/benchmark/linear_eval.py --task copd --pretrain $model_name --dim $dim