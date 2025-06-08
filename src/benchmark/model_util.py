# Yuwei (Evelyn) Zhang
# yz798@cam.ac.uk
# Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking
# https://github.com/evelyn0414/OPERA

import os

import numpy as np
import torch
from huggingface_hub.file_download import hf_hub_download
from tqdm import tqdm

from src.model.models_cola import Cola
from src.model.models_mae import mae_vit_small
from src.util import get_entire_signal_librosa, get_split_signal_librosa

# Set device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SR = 16000

# You can add your own model path here.

ENCODER_PATH_OPERA_CE_EFFICIENTNET = "cks/model/encoder-operaCE.ckpt"
ENCODER_PATH_OPERA_CT_HT_SAT = "cks/model/encoder-operaCT.ckpt"
ENCODER_PATH_OPERA_GT_VIT = "cks/model/encoder-operaGT.ckpt"
ENCODER_PATH_OPERA_CT_HEART_PHYSIONET16 = "cks/model/combined/physionet16/encoder-operaCT-physionet16-indomain-epoch=239--valid_acc=0.98-valid_loss=0.0524.ckpt"
ENCODER_PATH_OPERA_CT_HEART_CIRCOR = "cks/model/combined/circor/encoder-operaCT-circor-indomain-epoch=209--valid_acc=0.99-valid_loss=0.0397.ckpt"
ENCODER_PATH_OPERA_CT_HEART_PRETRAINED_PHYSIONET16 = "cks/model/combined/physionet16/encoder-operaCT-physionet16-indomain-pretrained-epoch=169--valid_acc=0.99-valid_loss=0.0300.ckpt"
ENCODER_PATH_OPERA_CT_HEART_PRETRAINED_CIRCOR = "cks/model/combined/circor/encoder-operaCT-circor-indomain-pretrained-epoch=229--valid_acc=0.99-valid_loss=0.0342.ckpt"
ENCODER_PATH_OPERA_CT_HEART_NONOISY_CIRCOR = "cks/model/combined/pascal_A_pascal_B_physionet16_zchsound_clean/encoder-operaCT-nocircor-nonoisy-epoch=249--valid_acc=0.96-valid_loss=0.2138.ckpt"
ENCODER_PATH_OPERA_CT_HEART_NONOISY_PASCAL = "cks/model/combined/circor_physionet16_zchsound_clean/encoder-operaCT-nopascal-nonoisy-epoch=159--valid_acc=0.94-valid_loss=0.3256.ckpt"
ENCODER_PATH_OPERA_CT_HEART_NONOISY_PHYSIONET = "cks/model/combined/circor_pascal_A_pascal_B_zchsound_clean/encoder-operaCT-nophysionet-nonoisy-epoch=249--valid_acc=0.95-valid_loss=0.2898.ckpt"
ENCODER_PATH_OPERA_CT_HEART_NONOISY_ZCHSOUND = "cks/model/combined/circor_pascal_A_pascal_B_physionet16/encoder-operaCT-nozchsound-epoch=169--valid_acc=0.94-valid_loss=0.3174.ckpt"
ENCODER_PATH_OPERA_CT_HEART_ALL = "cks/model/combined/circor_pascal_A_pascal_B_physionet16_zchsound_clean_zchsound_noisy/encoder-operaCT-heart-all-epoch=159--valid_acc=0.94-valid_loss=0.3790.ckpt"
ENCODER_PATH_OPERA_CT_HEART_CROSS_CIRCOR = "cks/model/combined/pascal_A_pascal_B_physionet16_zchsound_clean_zchsound_noisy/model.ckpt"
ENCODER_PATH_OPERA_CT_HEART_CROSS_PASCAL = "cks/model/combined/circor_physionet16_zchsound_clean_zchsound_noisy/model.ckpt"
ENCODER_PATH_OPERA_CT_HEART_CROSS_ZCHSOUND = "cks/model/combined/circor_pascal_A_pascal_B_physionet16/model.ckpt"
ENCODER_PATH_OPERA_CT_HEART_CROSS_PHYSIONET16 = "cks/model/combined/circor_pascal_A_pascal_B_zchsound_clean_zchsound_noisy/model.ckpt"
ENCODER_PATH_OPERA_CT_HEART_ALL_SCRATCH = "cks/model/combined/circor_pascal_A_pascal_B_physionet16_zchsound_clean_zchsound_noisy/encoder-operaCT-heart-all-scratch-epoch=209--valid_acc=0.92-valid_loss=0.3899.ckpt"
OPERA_CT_CIRCOR_MURMURS_FT_PATH = "cks/finetune/circor_murmurs/finetuning_linear_operaCT_64_0.0001_64_1e-05_2_weighted-epoch=08-valid_auc=0.77.ckpt"
OPERA_CT_CIRCOR_OUTCOMES_FT_PATH = "cks/finetune/circor_outcomes/finetuning_linear_operaCT_64_0.0001_64_1e-05_3_weighted-epoch=02-valid_auc=0.61.ckpt"
OPERA_CT_ZCHSOUND_CLEAN_OUTCOMES_FT_PATH = "cks/finetune/zchsound_clean/finetuning_linear_operaCT_64_0.0001_64_1e-05_3_weighted-epoch=17-valid_auc=0.83.ckpt"
OPERA_CT_ZCHSOUND_CLEAN_MURMURS_FT_PATH = "cks/finetune/zchsound_clean_murmurs/finetuning_linear_operaCT_64_0.0001_64_1e-05_1_weighted-epoch=08-valid_auc=0.96.ckpt"
OPERA_CT_ZCHSOUND_NOISY_OUTCOMES_FT_PATH = "cks/finetune/zchsound_noisy/finetuning_linear_operaCT_64_0.0001_64_1e-05_1_weighted-epoch=11-valid_auc=0.75.ckpt"
OPERA_CT_ZCHSOUND_NOISY_MURMURS_FT_PATH = "cks/finetune/zchsound_noisy_murmurs/finetuning_linear_operaCT_64_0.0001_64_1e-05_0_weighted-epoch=05-valid_auc=0.78.ckpt"


def get_audiomae_encoder_path(pretrain):
    encoder_paths = {
        "audiomae": "src/benchmark/baseline/audioMAE/pretrained.pth",
        "audiomae-heart-all": "cks/model/combined/circor_pascal_A_pascal_B_physionet16_zchsound_clean_zchsound_noisy/encoder-audiomae-heart-all-epoch=269--valid_acc=0.00-valid_loss=0.8422.ckpt",
        "audiomae-heart-circor-indomain": "cks/model/combined/circor/encoder-audiomae-heart-circor-indomain-epoch=389--valid_acc=0.00-valid_loss=1.0124.ckpt",
        "audiomae-heart-nozchsound": "cks/model/combined/circor_pascal_A_pascal_B_physionet16/encoder-audiomae-heart-nozchsound-epoch=289--valid_acc=0.00-valid_loss=0.8262.ckpt",
        "audiomae-heart-nophysionet16": "cks/model/combined/circor_pascal_A_pascal_B_zchsound_clean_zchsound_noisy/encoder-audiomae-heart-nophysionet16-epoch=329--valid_acc=0.00-valid_loss=0.9945.ckpt",
        "audiomae-heart-nopascal": "cks/model/combined/circor_physionet16_zchsound_clean_zchsound_noisy/encoder-audiomae-heart-nopascal-epoch=329--valid_acc=0.00-valid_loss=0.8338.ckpt",
        "audiomae-heart-nocircor": "cks/model/combined/pascal_A_pascal_B_physionet16_zchsound_clean_zchsound_noisy/encoder-audiomae-heart-nocircor-epoch=429--valid_acc=0.00-valid_loss=0.6585.ckpt",
        "audiomae-heart-physionet16-indomain": "cks/model/combined/physionet16/encoder-audiomae-heart-physionet16-indomain-epoch=459--valid_acc=0.00-valid_loss=0.5994.ckpt",
        "audiomae-heart-all-scratch": "cks/model/combined/circor_pascal_A_pascal_B_physionet16_zchsound_clean_zchsound_noisy/encoder-audiomae-heart-all-scratch-epoch=389--valid_acc=0.00-valid_loss=1.1551.ckpt",
    }
    print("Pretrain:", pretrain)
    if not os.path.exists(encoder_paths[pretrain]):
        raise ValueError("Model checkpoint not found")
    return encoder_paths[pretrain]


def get_encoder_path(pretrain):
    encoder_paths = {
        "operaCT": ENCODER_PATH_OPERA_CT_HT_SAT,
        "operaCE": ENCODER_PATH_OPERA_CE_EFFICIENTNET,
        "operaGT": ENCODER_PATH_OPERA_GT_VIT,
        "operaCT-heart-indomain-physionet16": ENCODER_PATH_OPERA_CT_HEART_PHYSIONET16,
        "operaCT-heart-indomain-circor": ENCODER_PATH_OPERA_CT_HEART_CIRCOR,
        "operaCT-heart-indomain-pretrained-physionet16": ENCODER_PATH_OPERA_CT_HEART_PRETRAINED_PHYSIONET16,
        "operaCT-heart-indomain-pretrained-circor": ENCODER_PATH_OPERA_CT_HEART_PRETRAINED_CIRCOR,
        "operaCT-heart-nonoisy-physionet16": ENCODER_PATH_OPERA_CT_HEART_NONOISY_PHYSIONET,
        "operaCT-heart-nonoisy-circor": ENCODER_PATH_OPERA_CT_HEART_NONOISY_CIRCOR,
        "operaCT-heart-nonoisy-pascal": ENCODER_PATH_OPERA_CT_HEART_NONOISY_PASCAL,
        "operaCT-heart-nonoisy-zchsound": ENCODER_PATH_OPERA_CT_HEART_NONOISY_ZCHSOUND,
        "operaCT-heart-nonoisy-zchsound_clean": ENCODER_PATH_OPERA_CT_HEART_NONOISY_ZCHSOUND,
        "operaCT-heart-nonoisy-zchsound_noisy": ENCODER_PATH_OPERA_CT_HEART_NONOISY_ZCHSOUND,
        "operaCT-heart-cross-physionet16": ENCODER_PATH_OPERA_CT_HEART_CROSS_PHYSIONET16,
        "operaCT-heart-cross-circor": ENCODER_PATH_OPERA_CT_HEART_CROSS_CIRCOR,
        "operaCT-heart-cross-pascal": ENCODER_PATH_OPERA_CT_HEART_CROSS_PASCAL,
        "operaCT-heart-cross-zchsound": ENCODER_PATH_OPERA_CT_HEART_CROSS_ZCHSOUND,
        "operaCT-heart-cross-zchsound_clean": ENCODER_PATH_OPERA_CT_HEART_CROSS_ZCHSOUND,
        "operaCT-heart-cross-zchsound_noisy": ENCODER_PATH_OPERA_CT_HEART_CROSS_ZCHSOUND,
        "operaCT-heart-all": ENCODER_PATH_OPERA_CT_HEART_ALL,
        "operaCT-heart-all-scratch": ENCODER_PATH_OPERA_CT_HEART_ALL_SCRATCH,
        "operaCT-ft-circor-murmurs": OPERA_CT_CIRCOR_MURMURS_FT_PATH,
        "operaCT-ft-circor-outcomes": OPERA_CT_CIRCOR_OUTCOMES_FT_PATH,
        "operaCT-ft-zchsound-clean-outcomes": OPERA_CT_ZCHSOUND_CLEAN_OUTCOMES_FT_PATH,
        "operaCT-ft-zchsound-clean-murmurs": OPERA_CT_ZCHSOUND_CLEAN_MURMURS_FT_PATH,
        "operaCT-ft-zchsound-noisy-outcomes": OPERA_CT_ZCHSOUND_NOISY_OUTCOMES_FT_PATH, 
        "operaCT-ft-zchsound-noisy-murmurs": OPERA_CT_ZCHSOUND_NOISY_MURMURS_FT_PATH,
    }
    if not os.path.exists(encoder_paths[pretrain]):
        if pretrain in ["operaCT", "operaCE", "operaGT"]:
            print("Model checkpoint not found, downloading from Hugging Face")
            download_ckpt(pretrain)
        else:
            raise ValueError("Model checkpoint not found. Run pretraining experiments first.")
    return encoder_paths[pretrain]


def download_ckpt(pretrain):
    model_repo = "evelyn0414/OPERA"
    model_name = "encoder-" + pretrain + ".ckpt"
    hf_hub_download(model_repo, model_name, local_dir="cks/model")


def extract_opera_feature(
    sound_dir_loc,
    pretrain="operaCE",
    input_sec=8,
    from_spec=False,
    dim=1280,
    pad0=False,
    ckpt_path=None
):
    """
    extract features using OPERA models
    """

    print("extracting feature from {} model with input_sec {}".format(pretrain, input_sec))

    MAE = ("mae" in pretrain or "GT" in pretrain)

    encoder_path = get_encoder_path(pretrain) if not ckpt_path else ckpt_path
    ckpt = torch.load(encoder_path, map_location=device)
    model = initialize_pretrained_model(pretrain)
    model.eval()
    model.load_state_dict(ckpt["state_dict"], strict=False)

    opera_features = []

    for audio_file in tqdm(sound_dir_loc):

        if MAE:
            if from_spec:
                data = [audio_file[i: i+256] for i in range(0, len(audio_file), 256)]
            else:
                data = get_split_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=input_sec) ##8.18s --> T=256
            features = []
            for x in data:
                if x.shape[0]>=16: # Kernel size can't be greater than actual input size
                    x = np.expand_dims(x, axis=0)
                    x = torch.tensor(x, dtype=torch.float).to(device)
                    fea = model.forward_feature(x).detach().cpu().numpy()
                    features.append(fea)
            features_sta = np.mean(features, axis=0)
            # print('MAE ViT feature dim:', features_sta.shape)
            opera_features.append(features_sta.tolist())
        else:
            #  put entire audio into the model
            if from_spec:
                data = audio_file
            else:
                # input is filename of an audio
                max_sec = 32 if pretrain == "operaCT" else None
                if pad0:
                    data = get_entire_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=input_sec, pad=True, types='zero', max_sec=max_sec)
                else:
                    data = get_entire_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=input_sec, pad=True, max_sec=max_sec)
            
            data = np.array(data)

            # for entire audio, batchsize = 1
            data = np.expand_dims(data, axis=0)

            x = torch.tensor(data, dtype=torch.float).to(device)
            features = model.extract_feature(x, dim).detach().cpu().numpy()

            # for entire audio, batchsize = 1
            opera_features.append(features.tolist()[0])

    x_data = np.array(opera_features)
    if MAE: 
        x_data = x_data.squeeze(1) 
    #print(x_data.shape)
    return x_data


def initialize_pretrained_model(pretrain):
    if "operaCT" in pretrain:
        model = Cola(encoder="htsat").to(device).float()
    elif pretrain == "operaCE":
        model = Cola(encoder="efficientnet").to(device).float()
    elif pretrain == "operaGT":
        model = (
            mae_vit_small(
                norm_pix_loss=False,
                in_chans=1,
                audio_exp=True,
                img_size=(256, 64),
                alpha=0.0,
                mode=0,
                use_custom_patch=False,
                split_pos=False,
                pos_trainable=False,
                use_nce=False,
                decoder_mode=1,  # decoder mode 0: global attn 1: swined local attn
                mask_2d=False,
                mask_t_prob=0.7,
                mask_f_prob=0.3,
                no_shift=False,
            )
            .to(device)
            .float()
        )
    else:
        raise NotImplementedError(
            f"Model not found: {pretrain}, please check the parameter."
        )
    return model
