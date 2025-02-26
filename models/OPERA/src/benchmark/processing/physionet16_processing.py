import argparse
import collections
import glob as gb
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

# Directories
data_dir = "datasets/physionet.org/files/challenge-2016/1.0.0/"
feature_dir = "feature/physionet16_eval/"

# Check if audio directory exists
if not os.path.exists(data_dir):
    print(os.getcwd())
    raise FileNotFoundError(
        f"Folder not found: {data_dir}, please ensure the dataset is downloaded."
    )


def get_files_and_labels(dir):
    files = gb.glob(os.path.join(dir, "*.wav"))
    print(f"{dir}: {len(files)} files")

    labels = []
    # Read label from .hea file
    for file in files:
        hea_file = file.replace("wav", "hea")
        with open(hea_file, "r") as f:
            lines = f.readlines()

        label = lines[-1].strip().lstrip("#").strip().lower()
        labels.append(label)
    return files, labels


def read_data():
    """Read data from data_dir and create file -> label mappings."""
    training_dirs = [
        "training-a",
        "training-b",
        "training-c",
        "training-d",
        "training-e",
        "training-f",
    ]
    # val_dir = "validation"

    label_to_int = {"normal": 0, "abnormal": 1}
    int_to_label = {0: "normal", 1: "abnormal"}

    # Save mappings
    with open(feature_dir + "label_to_int.json", "w") as f:
        json.dump(label_to_int, f)
    with open(feature_dir + "int_to_label.json", "w") as f:
        json.dump(int_to_label, f)

    # Collect sound files and labels
    sound_files = []
    labels = []
    # audio_splits = []

    for dir in training_dirs:
        audio_dir = os.path.join(data_dir, dir)
        files, y = get_files_and_labels(audio_dir)

        sound_files.extend(files)
        labels.extend(y)
        # audio_splits.extend(["train"] * len(files))

    # Validation dataset
    # audio_dir = os.path.join(data_dir, val_dir)
    # files, labels = get_files_and_labels(audio_dir)

    # sound_files.extend(files)
    # labels.extend(labels)
    # datasets.extend(["val"] * len(files))
    # audio_splits.extend(["val"] * len(files)

    # Convert to arrays for safety
    sound_files = np.array(sound_files)
    labels = np.array(labels)
    # audio_splits = np.array(audio_splits)

    return sound_files, labels


def preprocess_split():
    """Split dataset into train, val, and test sets, and save splits."""

    sound_files, labels = read_data()

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
        file_id = os.path.basename(file)
        if file_id in x_train:
            audio_splits.append("train")
        elif file_id in x_val:
            audio_splits.append("val")
        else:
            audio_splits.append("test")

    np.save(feature_dir + "train_test_split.npy", audio_splits)
    np.save(feature_dir + "labels.npy", labels)


def extract_and_save_embeddings(feature="operaCE", input_sec=8, dim=1280):
    from src.benchmark.model_util import extract_opera_feature

    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    opera_features = extract_opera_feature(
        sound_dir_loc,
        pretrain=feature,
        input_sec=input_sec,
        dim=dim,
        sr=1000,  # following pre-processing from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7868819
        lowcut=25,
        highcut=400,
    )
    feature += str(dim)
    np.save(feature_dir + feature + "_feature.npy", np.array(opera_features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--min_len_cnn", type=int, default=8)
    parser.add_argument("--min_len_htsat", type=int, default=8)

    args = parser.parse_args()

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        preprocess_split()

    if args.pretrain == "operaCT":
        input_sec = args.min_len_htsat
    elif args.pretrain == "operaCE":
        input_sec = args.min_len_cnn
    elif args.pretrain == "operaGT":
        input_sec = 8.18
    extract_and_save_embeddings(args.pretrain, input_sec=input_sec, dim=args.dim)
