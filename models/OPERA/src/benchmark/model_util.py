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


def get_encoder_path(pretrain):
    encoder_paths = {
        "operaCT": ENCODER_PATH_OPERA_CT_HT_SAT,
        "operaCE": ENCODER_PATH_OPERA_CE_EFFICIENTNET,
        "operaGT": ENCODER_PATH_OPERA_GT_VIT,
    }
    if not os.path.exists(encoder_paths[pretrain]):
        print("Model checkpoint not found, downloading from Hugging Face")
        download_ckpt(pretrain)
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
    sr=None,
    butterworth_filter=None,
    lowcut=200,
    highcut=1800,
):
    """
    extract features using OPERA models
    """
    print(f"Extracting features from {pretrain} model with input_sec={input_sec}")
    MAE = "mae" in pretrain or "GT" in pretrain

    encoder_path = get_encoder_path(pretrain)
    ckpt = torch.load(encoder_path, map_location=device)
    model = initialize_pretrained_model(pretrain)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    opera_features = []

    with torch.no_grad():
        for audio_file in tqdm(sound_dir_loc):
            if MAE:
                data = (
                    get_split_signal_librosa(
                        "",
                        audio_file[:-4],
                        spectrogram=True,
                        input_sec=input_sec,
                        sample_rate=sr,
                        butterworth_filter=butterworth_filter,
                        lowcut=lowcut,
                        highcut=highcut,
                    )
                    if not from_spec
                    else [
                        audio_file[i : i + 256] for i in range(0, len(audio_file), 256)
                    ]
                )
                features = [
                    model.forward_feature(
                        torch.tensor(np.expand_dims(x, axis=0), dtype=torch.float).to(
                            device
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    for x in data
                    if x.shape[0] >= 16
                ]
                features_sta = np.mean(features, axis=0)
                opera_features.append(features_sta.tolist())
            else:
                data = get_entire_signal_librosa(
                    "",
                    audio_file[:-4],
                    spectrogram=True,
                    input_sec=input_sec,
                    pad=True,
                    types="zero" if pad0 else None,
                    sample_rate=sr,
                    butterworth_filter=butterworth_filter,
                    lowcut=lowcut,
                    highcut=highcut,
                )
                x = torch.tensor(np.expand_dims(data, axis=0), dtype=torch.float).to(
                    device
                )
                features = model.extract_feature(x, dim).detach().cpu().numpy()
                opera_features.append(features.tolist()[0])

    x_data = np.array(opera_features)
    if MAE:
        x_data = x_data.squeeze(1)
    print(f"Feature extraction completed. X shape: {x_data.shape}")
    return x_data


def initialize_pretrained_model(pretrain):
    if pretrain == "operaCT":
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
                decoder_mode=1,
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
