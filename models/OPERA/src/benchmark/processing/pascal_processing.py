import argparse
import collections
import glob as gb
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from src.benchmark.model_util import extract_opera_feature
from src.benchmark.baseline.extract_feature import (
    extract_audioMAE_feature,
    extract_clap_feature,
    extract_vgg_feature,
    extract_HeAR_feature,
)

# Directories
data_dir = "datasets/PASCAL/"
OPERACT_HEART_CKPT_PATH = "cks/model/combined/circor_physionet16_zchsound_clean_zchsound_noisy/encoder-operaCT-nopascal-epoch=99--valid_acc=0.93-valid_loss=0.3276.ckpt"
OPERACT_HEART_NONOISY_CKPT_PATH = "cks/model/combined/circor_physionet16_zchsound_clean/encoder-operaCT-nopascal-nonoisy-epoch=159--valid_acc=0.94-valid_loss=0.3256.ckpt"
OPERACT_HEART_ALL_CKPT_PATH = "cks/model/combined/circor_pascal_A_pascal_B_physionet16_zchsound_clean_zchsound_noisy/encoder-operaCT-heart-all-epoch=159--valid_acc=0.94-valid_loss=0.3790.ckpt"

# Check if audio directory exists
if not os.path.exists(data_dir):
    print(os.getcwd())
    raise FileNotFoundError(
        f"Folder not found: {data_dir}, please ensure the dataset is downloaded."
    )


def read_data(dataset):
    """Read data from data_dir and create file -> label mappings."""
    dirs_A = [
        "Atraining_artifact",
        "Atraining_extrahls",
        "Atraining_murmur",
        "Atraining_normal",
    ]
    dirs_B = ["Btraining_extrastole", "Btraining_murmur", "BTraining_normal"]

    if dataset == "A":
        label_to_int = {"normal": 0, "murmur": 1, "extrahls": 2, "artifact": 3}
        int_to_label = {0: "normal", 1: "murmur", 2: "extrahls", 3: "artifact"}
        dirs = dirs_A
    elif dataset == "B":
        label_to_int = {"normal": 0, "murmur": 1, "extrastole": 2}
        int_to_label = {0: "normal", 1: "murmur", 2: "extrastole"}
        dirs = dirs_B
    else:
        raise ValueError("Please input a valid value for dataset: A or B.")

    # Save mappings
    with open(feature_dir + "label_to_int.json", "w") as f:
        json.dump(label_to_int, f)
    with open(feature_dir + "int_to_label.json", "w") as f:
        json.dump(int_to_label, f)

    # Collect sound files and labels
    sound_files = []
    labels = []
    for dir in dirs:
        audio_dir = os.path.join(data_dir, dir)
        label = label_to_int[dir.split("_")[1]]
        files = gb.glob(os.path.join(audio_dir, "*.wav"))
        print(f"{dir}: {len(files)} files")

        sound_files.extend(files)
        labels.extend([label] * len(files))

    # Convert to arrays for safety
    sound_files = np.array(sound_files)
    labels = np.array(labels)

    return sound_files, labels, label_to_int


def preprocess_split(dataset):
    """Split dataset into train, val, and test sets, and save splits."""

    sound_files, labels, label_to_int = read_data(dataset)

    # Verify initial distribution
    print("Initial Class Distribution:", dict(collections.Counter(labels)))

    # Perform stratified splits on the sound files
    _x_train, x_test, _y_train, y_test = train_test_split(
        sound_files, labels, test_size=0.2, random_state=1337, stratify=labels
    )

    x_train, x_val, y_train, y_val = train_test_split(
        _x_train, _y_train, test_size=0.2, random_state=1337, stratify=_y_train
    )

    print("Class distribution:")
    print(f"Train: {collections.Counter(y_train)}")
    print(f"Val: {collections.Counter(y_val)}")
    print(f"Test: {collections.Counter(y_test)}")

    # Save .wav file locations
    np.save(feature_dir + "sound_dir_loc.npy", sound_files)

    # Create train/val/test splits for audio files
    audio_splits = []
    for i, file in enumerate(sound_files):
        if file in x_train:
            print("train")
            audio_splits.append("train")
        elif file in x_val:
            audio_splits.append("val")
        else:
            audio_splits.append("test")

    np.save(feature_dir + "train_test_split.npy", audio_splits)
    np.save(feature_dir + "labels.npy", labels)


def extract_and_save_embeddings_baselines(
    feature="audiomae", fine_tuned=None, ckpt_path=None, seed=None
):
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    suffix = "" if not fine_tuned else f"_finetuned_{fine_tuned}_{seed}"

    if feature == "vggish":  # no fine-tuning
        vgg_features = extract_vgg_feature(sound_dir_loc)
        np.save(feature_dir + feature + "_feature.npy", np.array(vgg_features))
    elif feature == "clap":
        clap_features = extract_clap_feature(sound_dir_loc, ckpt_path=ckpt_path)
        np.save(
            feature_dir + feature + suffix + "_feature.npy", np.array(clap_features)
        )
    elif feature == "clap2023":
        clap2023_features = extract_clap_feature(sound_dir_loc, version="2023")
        np.save(
            feature_dir + feature + suffix + "_feature.npy", np.array(clap2023_features)
        )
    elif feature == "audiomae":
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc, ckpt_path=ckpt_path)
        np.save(
            feature_dir + feature + suffix + "_feature.npy", np.array(audiomae_feature)
        )
    elif feature == "hear":  # no fine-tuning possible, not open-source
        hear_feature = extract_HeAR_feature(sound_dir_loc)
        np.save(feature_dir + "hear_feature.npy", np.array(hear_feature))


def extract_and_save_embeddings(
    feature="operaCE", input_sec=8, dim=1280, fine_tuned=None, ckpt_path=None, seed=None
):
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    if feature == "operaCT-heart":
        ckpt_path = OPERACT_HEART_CKPT_PATH
        pretrain = "operaCT"  # necessary as input to extract_opera_feature
    elif feature == "operaCT-heart-nonoisy":
        ckpt_path = OPERACT_HEART_NONOISY_CKPT_PATH
        pretrain = "operaCT"
    elif feature == "operaCT-heart-all":
        ckpt_path = OPERACT_HEART_ALL_CKPT_PATH
        pretrain = "operaCT"
    else:
        pretrain = feature
    opera_features = extract_opera_feature(
        sound_dir_loc,
        pretrain=pretrain,
        input_sec=input_sec,
        dim=dim,
        ckpt_path=ckpt_path,
    )
    feature += str(dim)
    suffix = "" if not fine_tuned else f"_finetuned_{fine_tuned}_{seed}"
    np.save(feature_dir + feature + suffix + "_feature.npy", np.array(opera_features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--min_len_cnn", type=int, default=2)
    parser.add_argument("--min_len_htsat", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="A")
    parser.add_argument("--fine_tuned", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)

    args = parser.parse_args()
    if args.dataset == "A":
        feature_dir = "feature/pascal_A_eval/"
    elif args.dataset == "B":
        feature_dir = "feature/pascal_B_eval/"
    else:
        raise ValueError("Please input a valid value for dataset: A or B.")

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        preprocess_split(args.dataset)

    if args.ckpt_path:
        seed = args.ckpt_path.split("/")[-1].split("_")[7][0]  # get seed from filename
    else:
        seed = None

    if args.pretrain in ["vggish", "clap", "audiomae", "hear", "clap2023"]:
        extract_and_save_embeddings_baselines(
            args.pretrain, args.fine_tuned, args.ckpt_path, seed
        )
    else:
        if (
            args.pretrain == "operaCT"
            or args.pretrain == "operaCT-heart"
            or args.pretrain == "operaCT-heart-nonoisy"
            or args.pretrain == "operaCT-heart-all"
        ):
            input_sec = args.min_len_htsat
        elif args.pretrain == "operaCE":
            input_sec = args.min_len_cnn
        elif args.pretrain == "operaGT":
            input_sec = 8.18
        extract_and_save_embeddings(
            args.pretrain,
            input_sec=input_sec,
            dim=args.dim,
            fine_tuned=args.fine_tuned,
            ckpt_path=args.ckpt_path,
            seed=seed,
        )
