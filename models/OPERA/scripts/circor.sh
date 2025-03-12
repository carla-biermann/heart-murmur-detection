model_name="operaCT"
dim=768
python src/benchmark/processing/circor_processing.py --pretrain $model_name --dim $dim
#python src/benchmark/linear_eval.py --task circor --pretrain $model_name --dim $dim --n_run 1

model_name="operaCE"
dim=1280
python src/benchmark/processing/circor_processing.py --pretrain $model_name --dim $dim
#python src/benchmark/linear_eval.py --task circor --pretrain $model_name --dim $dim --n_run 1

model_name="operaGT"
dim=384
python src/benchmark/processing/circor_processing.py --pretrain $model_name --dim $dim
#python src/benchmark/linear_eval.py --task circor --pretrain $model_name --dim $dim --n_run 1


python src/benchmark/processing/circor_processing.py --pretrain "vggish"
python src/benchmark/processing/circor_processing.py --pretrain "clap"
python src/benchmark/processing/circor_processing.py --pretrain "audiomae"