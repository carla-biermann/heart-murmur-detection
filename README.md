# Domain Adaptation of Audio Foundation Models: A Case Study on Heart Murmurs

This is the code used for the corresponding 2025 MPhil project in Advanced Computer Science at the University of Cambridge.

## Installation

The environment with all the needed dependecies can be easily created on a Linux machine by running:
```
conda env create --file environment.yml
sh ./prepare_env.sh
source ~/.bashrc

conda init
conda activate audio
sh ./prepare_code.sh
```

*After installation, next time to run the code, you only need to acivate the audio env by `conda activate audio`.

The HeAR model requires different dependencies. When using it for feature extraction or fine-tuning, please create the corresponding environment:

```
conda env create --file hear_environment.yml
sh ./prepare_env.sh
source ~/.bashrc

conda init
conda activate hear_env
```

*After installation, next time to run the code, you only need to acivate the audio env by `conda activate hear_env`.

## Preparing data

| Dataset                                 | Access                                                       | License        |
| ---------------------------------------- | ------------------------------------------------------------ | -------------- |
| CirCor       | [https://physionet.org/content/circor-heart-sound/1.0.3/](https://physionet.org/content/circor-heart-sound/1.0.3/) | (ODC-By) v1.0         |
| PASCAL        | [https://istethoscope.peterjbentley.com/heartchallenge/index.html](https://istethoscope.peterjbentley.com/heartchallenge/index.html) |  |
| PhysioNet 2016      | [https://physionet.org/content/challenge-2016/1.0.0/](https://physionet.org/content/challenge-2016/1.0.0/) |  (ODC-By) v1.0     |
| ZCHSound         | [http://zchsound.ncrcch.org.cn/](http://zchsound.ncrcch.org.cn/)* |              |

*The ZCHSound dataset does not seem to be available through the link provided by the dataset authors anymore. The corresponding paper is the following: [https://pubmed.ncbi.nlm.nih.gov/38194403/](https://pubmed.ncbi.nlm.nih.gov/38194403/).

The datasets must be downloaded via these links for use and placed in the `datasets` folder.

## Using benchmark models

The pretrained weights for all OPERA models are available at:
__Zenodo__ or <a href="https://huggingface.co/evelyn0414/OPERA/tree/main" target="_blank"> HuggingFace </a>. This work uses the pretrained model checkpoint of [OPERA-CT](https://huggingface.co/evelyn0414/OPERA/resolve/main/encoder-operaCT.ckpt?download=true). This will be automatically downloaded before feature extraction. 

To use the HeAR model, users must create a HuggingFace token and insert it in the `extract_HeAR_feature` function in `src/benchmark/baseline/extract_feature.py`. The HeAR model generally relies on different dependencies than the other models. When using the HeAR model for feature extraction or fine-tuning please use the `hear_env` described in the Installation section.

The pretrained weights for Audio-MAE can be downloaded by following the instructions in the corresponding directory `src/benchmark/baseline/audioMAE`. 

## Continued Pretraining (CP) of OPERA-CT and Audio-MAE

Training can be found in  `cola_pretraining.py` and `mae_pretraining.py`.

The commands used for CP and pretraining from scratch are located in `scripts/run_cp.sh`. The best checkpoints are saved in the `cks/model/combined` directory.

## Benchmarking 

The linear probing and fine-tuning benchmarks for one model can be obtained using the `scripts/lp_eval.sh` and `scripts/ft_eval.sh` scripts (Corresponding to H1.1 and H1.2). Here is an example for the baseline models before any continued pretraining:

```
sh scripts/lp_eval.sh audiomae
sh scripts/lp_eval.sh clap
sh scripts/lp_eval.sh hear
sh scripts/lp_eval.sh operaCT 768

sh scripts/ft_eval.sh audiomae 768
sh scripts/ft_eval.sh clap 1024
sh scripts/ft_eval.sh hear 512
sh scripts/ft_eval.sh operaCT 768
```

Once CP or PT from scratch was run on Audio-MAE or OPERA-CT, these models can be benchmarked by using the model titles they are given in `scripts/run_cp.sh` (Corresponding to H1.3, H1.4, H2.2, H2.3, H3.1). For example,

```
sh scripts/lp_eval.sh audiomae-heart-all
sh scripts/lp_eval.sh operaCT-heart-all
```

The results concerning the generalisability of fine-tuned models can be re-created by first fine-tuning OPERA-CT on all tasks and then completing the script `scripts/cross_ft_lp_eval.sh` with the correct checkpoitns and running it (Corresponding to H2.1).
