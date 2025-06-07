import argparse
import collections
import csv
import glob as gb
import json
import os
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
from sklearn.model_selection import train_test_split

from src.benchmark.model_util import extract_opera_feature, get_audiomae_encoder_path
from src.benchmark.baseline.extract_feature import (
    extract_audioMAE_feature,
    extract_clap_feature,
    extract_vgg_feature,
    extract_HeAR_feature,
)

# Directories
data_dir = "datasets/ZCHSound/"
int_to_murmurs = {"0": "Absent", "1": "Present"}
int_to_outcomes = {"0": "ASD", "1": "NORMAL", "2": "PDA", "3": "PFO", "4": "VSD"}
murmurs_to_int = {"NORMAL": 0, "ASD": 1, "PDA": 1, "PFO": 1, "VSD": 1}
outcomes_to_int = {"ASD": 0, "NORMAL": 1, "PDA": 2, "PFO": 3, "VSD": 4}

OPERACT_HEART_CKPT_PATH = "cks/model/combined/circor_pascal_A_pascal_B_physionet16/encoder-operaCT-nozchsound-epoch=169--valid_acc=0.94-valid_loss=0.3174.ckpt"
OPERACT_HEART_ALL_CKPT_PATH = "cks/model/combined/circor_pascal_A_pascal_B_physionet16_zchsound_clean_zchsound_noisy/encoder-operaCT-heart-all-epoch=159--valid_acc=0.94-valid_loss=0.3790.ckpt"
ENCODER_PATH_OPERA_CT_HEART_ALL_SCRATCH = "cks/model/combined/circor_pascal_A_pascal_B_physionet16_zchsound_clean_zchsound_noisy/encoder-operaCT-heart-all-scratch-epoch=209--valid_acc=0.92-valid_loss=0.3899.ckpt"

def get_labels_from_csv(path):
    """Read labels from ZCHSound CSV file and create mappings."""
    label_dict = {}
    label_set = set()  # Collect unique labels for mapping

    with open(path, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=";")
        header = next(csvreader)  # Skip header
        for row in csvreader:
            file_id, diagnosis = row[0], row[3]
            label_dict[file_id] = diagnosis
            label_set.add(diagnosis)

    # Save mappings
    with open(feature_dir + "int_to_outcomes.json", "w") as f:
        json.dump(int_to_outcomes, f)
    with open(feature_dir + "int_to_murmurs.json", "w") as f:
        json.dump(int_to_murmurs, f)

    print(f"Label Mappings: {outcomes_to_int}")
    print(f"Label Mappings: {murmurs_to_int}")
    return label_dict


def preprocess_split(csv_filename="Clean Heartsound Data Details.csv"):
    """Split dataset into train, val, and test sets, and save splits."""
    label_dict = get_labels_from_csv(data_dir + csv_filename)

    # Get patient IDs and labels (convert labels to integers)
    patient_ids = list(label_dict.keys())
    outcomes = [outcomes_to_int[label_dict[u]] for u in patient_ids]

    # Split: Train (64%), Val (16%), Test (20%)
    _x_train, x_test, _y_train, y_test = train_test_split(
        patient_ids, outcomes, test_size=0.2, random_state=42, stratify=outcomes
    )
    x_train, x_val, y_train, y_val = train_test_split(
        _x_train, _y_train, test_size=0.2, random_state=42, stratify=_y_train
    )

    print("Class distribution:")
    print(f"Train: {collections.Counter(y_train)}")
    print(f"Val: {collections.Counter(y_val)}")
    print(f"Test: {collections.Counter(y_test)}")

    # Save .wav file locations
    sound_files = np.array(gb.glob(audio_dir + "/*.wav"))
    np.save(feature_dir + "sound_dir_loc.npy", sound_files)

    # Create train/val/test splits for audio files
    audio_splits = []
    outcome_labels = []
    murmur_labels = []
    for file in sound_files:
        file_id = os.path.basename(file)
        if file_id in x_train:
            audio_splits.append("train")
        elif file_id in x_val:
            audio_splits.append("val")
        else:
            audio_splits.append("test")
        outcome_labels.append(outcomes_to_int[label_dict[file_id]])
        murmur_labels.append(murmurs_to_int[label_dict[file_id]])

    np.save(feature_dir + "train_test_split.npy", audio_splits)
    np.save(feature_dir + "outcomes.npy", outcome_labels)
    np.save(feature_dir + "murmurs.npy", murmur_labels)


def extract_and_save_embeddings_baselines(
    feature="audiomae", fine_tuned=None, ckpt_path=None, seed=None
):
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    suffix = "" if not fine_tuned else f"_finetuned_{fine_tuned}_{seed}"

    if feature == "vggish": # no fine-tuning
        vgg_features = extract_vgg_feature(sound_dir_loc)
        np.save(feature_dir + feature + "_feature.npy", np.array(vgg_features))
    elif feature == "clap":
        clap_features = extract_clap_feature(sound_dir_loc, ckpt_path=ckpt_path)
        np.save(feature_dir + feature + suffix + "_feature.npy", np.array(clap_features))
    elif feature == "clap2023":
        clap2023_features = extract_clap_feature(sound_dir_loc, version="2023")
        np.save(feature_dir + feature + "_feature.npy", np.array(clap2023_features))
    elif feature == "audiomae":
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc, ckpt_path=ckpt_path)
        np.save(feature_dir + feature + suffix + "_feature.npy", np.array(audiomae_feature))
    elif feature == "hear": # no fine-tuning possible, not open-source
        hear_feature = extract_HeAR_feature(sound_dir_loc)
        np.save(feature_dir + "hear_feature.npy", np.array(hear_feature))
    elif "audiomae" in feature:
        ckpt_path = get_audiomae_encoder_path(feature)
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc, ckpt_path=ckpt_path)
        np.save(feature_dir + feature + "_feature.npy", np.array(audiomae_feature))

def extract_and_save_embeddings(
    feature="operaCE", input_sec=8, dim=1280, fine_tuned=None, ckpt_path=None, seed=None
):
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    pad0 = True if feature in ["operaCT", "operaCE"] else False
    if feature == "operaCT-heart":
        ckpt_path = OPERACT_HEART_CKPT_PATH 
        pretrain = "operaCT" # necessary as input to extract_opera_feature
    elif feature == "operaCT-heart-all":
        ckpt_path = OPERACT_HEART_ALL_CKPT_PATH
        pretrain = "operaCT"
    elif feature == "operaCT-heart-all-scratch":
        ckpt_path = ENCODER_PATH_OPERA_CT_HEART_ALL_SCRATCH
        pretrain = "operaCT"
    else:
        pretrain = feature
    opera_features = extract_opera_feature(
        sound_dir_loc,
        pretrain=pretrain,
        input_sec=input_sec,
        dim=dim,
        pad0=pad0,
        ckpt_path=ckpt_path,
    )
    feature += str(dim)
    suffix = "" if not fine_tuned else f"_finetuned_{fine_tuned}_{seed}"
    np.save(feature_dir + feature + suffix + "_feature.npy", np.array(opera_features))


@hydra.main(config_path="../configs", config_name="zchsound_config", version_base=None)
def main(cfg: DictConfig):
    global feature_dir, audio_dir
    print(OmegaConf.to_yaml(cfg))
    if cfg.fine_tuned == "None":
        cfg.fine_tuned = None
    if cfg.ckpt_path == "None":
        cfg.ckpt_path = None
    if cfg.seed == "None":
        cfg.seed = None 

    if cfg.data == "clean":
        audio_dir = data_dir + "clean Heartsound Data"
        feature_dir = "feature/zchsound_clean_eval/"
        csv_filename = "Clean Heartsound Data Details.csv"
    elif cfg.data == "noisy":
        audio_dir = data_dir + "Noise Heartsound Data Details"
        feature_dir = "feature/zchsound_noisy_eval/"
        csv_filename = "Noise Heartsound Data Details.csv"
    else:
        raise ValueError("Please select a valid dataset: clean or noisy")

    # Check if audio directory exists
    if not os.path.exists(audio_dir):
        print(os.getcwd())
        raise FileNotFoundError(
            f"Folder not found: {audio_dir}, please ensure the dataset is downloaded."
        )

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        preprocess_split(csv_filename)
    
    if cfg.ckpt_path:
        seed = cfg.ckpt_path.split('/')[-1].split('_')[7][0] # get seed from filename
    else:
        seed = None

    if cfg.pretrain in ["vggish", "clap", "audiomae", "hear", "clap2023"] or "audiomae" in cfg.pretrain:
        extract_and_save_embeddings_baselines(
            cfg.pretrain, cfg.fine_tuned, cfg.ckpt_path, seed
        )
    else:
        if (
            cfg.pretrain == "operaCT"
            or cfg.pretrain == "operaCT-heart"
            or cfg.pretrain == "operaCT-heart-all"
            or cfg.pretrain == "operaCT-heart-all-scratch"
        ):
            input_sec = cfg.min_len_htsat
        elif cfg.pretrain == "operaCE":
            input_sec = cfg.min_len_cnn
        elif cfg.pretrain == "operaGT":
            input_sec = 8.18
        extract_and_save_embeddings(
            cfg.pretrain,
            input_sec=input_sec,
            dim=cfg.dim,
            fine_tuned=cfg.fine_tuned,
            ckpt_path=cfg.ckpt_path,
            seed=seed,
        )

if __name__ == '__main__':
    main()
