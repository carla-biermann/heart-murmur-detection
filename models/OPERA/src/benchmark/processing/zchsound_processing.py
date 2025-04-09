import argparse
import collections
import csv
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
data_dir = "datasets/ZCHSound/"
int_to_label = {"0": "ASD", "1": "NORMAL", "2": "PDA", "3": "PFO", "4": "VSD"}


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

    # Create label mappings
    label_to_int = {label: idx for idx, label in int_to_label.items()}

    # Save mappings
    with open(feature_dir + "label_to_int.json", "w") as f:
        json.dump(label_to_int, f)
    with open(feature_dir + "int_to_label.json", "w") as f:
        json.dump(int_to_label, f)

    print(f"Label Mappings: {label_to_int}")
    return label_dict, label_to_int


def preprocess_split(csv_filename="Clean Heartsound Data Details.csv"):
    """Split dataset into train, val, and test sets, and save splits."""
    label_dict, label_to_int = get_labels_from_csv(data_dir + csv_filename)

    # Get patient IDs and labels (convert labels to integers)
    patient_ids = list(label_dict.keys())
    labels = [label_to_int[label_dict[u]] for u in patient_ids]

    # Split: Train (64%), Val (16%), Test (20%)
    _x_train, x_test, _y_train, y_test = train_test_split(
        patient_ids, labels, test_size=0.2, random_state=42, stratify=labels
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
    audio_labels = []
    for file in sound_files:
        file_id = os.path.basename(file)
        if file_id in x_train:
            audio_splits.append("train")
        elif file_id in x_val:
            audio_splits.append("val")
        else:
            audio_splits.append("test")
        audio_labels.append(label_to_int[label_dict[file_id]])

    np.save(feature_dir + "train_test_split.npy", audio_splits)
    np.save(feature_dir + "labels.npy", audio_labels)


def check_demographic(trait="label"):
    """Check the class distribution for train/val/test sets."""
    print(f"Checking class distribution by {trait}")

    sound_files = np.load(feature_dir + "sound_dir_loc.npy")
    labels = np.load(feature_dir + "labels.npy")
    splits = np.load(feature_dir + "train_test_split.npy")

    # Load label mappings
    with open(feature_dir + "int_to_label.json", "r") as f:
        int_to_label = json.load(f)

    for split_name in ["train", "val", "test"]:
        subset = sound_files[splits == split_name]
        counts = collections.defaultdict(int)
        for i, file in enumerate(subset):
            label = labels[i]
            counts[int_to_label[str(label)]] += 1
        print(f"{split_name.capitalize()} Distribution: {dict(counts)}")

def extract_and_save_embeddings_baselines(feature="audiomae"):
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")

    if feature == "vggish":
        vgg_features = extract_vgg_feature(sound_dir_loc)
        np.save(feature_dir + "vggish_feature.npy", np.array(vgg_features))
    elif feature == "clap":
        clap_features = extract_clap_feature(sound_dir_loc)
        np.save(feature_dir + "clap_feature.npy", np.array(clap_features))
    elif feature == "audiomae":
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc)
        np.save(feature_dir + "audiomae_feature.npy", np.array(audiomae_feature))
    elif feature == "hear":
        hear_feature = extract_HeAR_feature(sound_dir_loc)
        np.save(feature_dir + "hear_feature.npy", np.array(hear_feature))

def extract_and_save_embeddings(feature="operaCE", input_sec=8, dim=1280):
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    pad0 = True if feature in ["operaCT", "operaCE"] else False
    opera_features = extract_opera_feature(
        sound_dir_loc,
        pretrain=feature,
        input_sec=input_sec,
        dim=dim,
        pad0=pad0,
        sr=2000,
        butterworth_filter=3,
        lowcut=20,
        highcut=650,
    )
    feature += str(dim)
    np.save(feature_dir + feature + "_feature.npy", np.array(opera_features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--min_len_cnn", type=int, default=8)
    parser.add_argument("--min_len_htsat", type=int, default=8)
    parser.add_argument("--data", type=str, default="clean")

    args = parser.parse_args()

    if args.data == "clean":
        audio_dir = data_dir + "clean Heartsound Data"
        feature_dir = "feature/zchsound_clean_eval/"
        csv_filename = "Clean Heartsound Data Details.csv"
    elif args.data == "noisy":
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
        check_demographic()

    if args.pretrain in ["vggish", "clap", "audiomae", "hear"]:
        extract_and_save_embeddings_baselines(args.pretrain)
    else:
        if args.pretrain == "operaCT":
            input_sec = args.min_len_htsat
        elif args.pretrain == "operaCE":
            input_sec = args.min_len_cnn
        elif args.pretrain == "operaGT":
            input_sec = 8.18
        extract_and_save_embeddings(args.pretrain, input_sec=input_sec, dim=args.dim)
