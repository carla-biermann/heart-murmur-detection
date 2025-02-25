list="opensmile vggish audiomae clap"  
for i in $list;  
do  
python src/benchmark/processing/kauh_processing.py --pretrain $i 
python src/benchmark/linear_eval.py --task kauh --pretrain $i
done 

model_name="operaCT"
dim=768
python src/benchmark/processing/kauh_processing.py --pretrain $model_name --dim $dim
python src/benchmark/linear_eval.py --task kauh --pretrain $model_name --dim $dim

model_name="operaCE"
dim=1280
python src/benchmark/processing/kauh_processing.py --pretrain $model_name --dim $dim
python src/benchmark/linear_eval.py --task kauh --pretrain $model_name --dim $dim

model_name="operaGT"
dim=384
python src/benchmark/processing/kauh_processing.py --pretrain $model_name --dim $dim
python src/benchmark/linear_eval.py --task kauh --pretrain $model_name --dim $dim


