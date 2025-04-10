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
int_to_murmurs = {"0": "Absent", "1": "Present"}
int_to_outcomes = {"0": "ASD", "1": "NORMAL", "2": "PDA", "3": "PFO", "4": "VSD"}
murmurs_to_int = {"NORMAL": 0, "ASD": 1, "PDA": 1, "PFO": 1, "VSD": 1}
outcomes_to_int = {"ASD": 0, "NORMAL": 1, "PDA": 2, "PFO": 3, "VSD": 4}


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
