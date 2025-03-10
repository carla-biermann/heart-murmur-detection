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
)

# Directories
data_dir = "datasets/circor/"
audio_dir = "datasets/circor/training_data"
feature_dir = "feature/circor_eval/"

# Check if audio directory exists
if not os.path.exists(data_dir):
    print(os.getcwd())
    raise FileNotFoundError(
        f"Folder not found: {data_dir}, please ensure the dataset is downloaded."
    )


def get_labels_from_csv():
    """Read labels from Circor CSV file and create mappings."""
    file_ids, murmurs, outcomes = [], [], []

    with open(data_dir + "training_data.csv", "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        header = next(csvreader)  # Skip header
        for row in csvreader:
            pat_id, recording_loc, murmur, outcome = row[0], row[1], row[7], row[20]
            for loc in recording_loc.split("+"):
                file_ids.append(f"{pat_id}_{loc}")
                murmurs.append(murmur)
                outcomes.append(outcome)

    # Create label mappings
    murmurs_to_int = {murmur: idx for idx, murmur in enumerate(sorted(set(murmurs)))}
    outcome_to_int = {outcome: idx for idx, outcome in enumerate(sorted(set(outcomes)))}
    int_to_murmurs = {idx: murmur for murmur, idx in murmurs_to_int.items()}
    int_to_outcome = {idx: outcome for outcome, idx in outcome_to_int.items()}

    # Save mappings
    with open(feature_dir + "int_to_murmurs.json", "w") as f:
        json.dump(int_to_murmurs, f)
    with open(feature_dir + "int_to_outcome.json", "w") as f:
        json.dump(int_to_outcome, f)

    print(f"Murmur Mappings: {murmurs_to_int}")
    print(f"Outcome Mappings: {outcome_to_int}")

    murmur_ints = [murmurs_to_int[m] for m in murmurs]
    outcome_ints = [outcome_to_int[m] for m in outcomes]

    return np.array(file_ids), np.array(murmur_ints), np.array(outcome_ints)


def preprocess_split():
    """Split dataset into train, val, and test sets, and save splits."""
    file_ids, murmurs, outcomes = get_labels_from_csv()

    # Split: Train (64%), Val (16%), Test (20%)
    _x_train, x_test, _y_train, y_test = train_test_split(
        file_ids, murmurs, test_size=0.2, random_state=42, stratify=murmurs
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
    outcome_labels = []
    for i, file in enumerate(sound_files):
        file_id = os.path.basename(file).split(".")[0] # Remove ".wav" from filename
        if file_id in x_train:
            audio_splits.append("train")
        elif file_id in x_val:
            audio_splits.append("val")
        else:
            audio_splits.append("test")
        audio_labels.append(murmurs[i])
        outcome_labels.append(outcomes[i])

    np.save(feature_dir + "train_test_split.npy", audio_splits)
    np.save(feature_dir + "murmurs.npy", audio_labels)
    np.save(feature_dir + "outcomes.npy", outcome_labels)


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


def extract_and_save_embeddings(feature="operaCE", input_sec=15, dim=1280):
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    opera_features = extract_opera_feature(
        sound_dir_loc,
        pretrain=feature,
        input_sec=input_sec,
        dim=dim,
        pad0=True,
        sr=2000,
        butterworth_filter=3,
        lowcut=20,
        highcut=800,
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

    # Check if audio directory exists
    if not os.path.exists(audio_dir):
        print(os.getcwd())
        raise FileNotFoundError(
            f"Folder not found: {audio_dir}, please ensure the dataset is downloaded."
        )

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        preprocess_split()

    if args.pretrain in ["vggish", "clap", "audiomae"]:
        extract_and_save_embeddings_baselines(args.pretrain)
    else:
        if args.pretrain == "operaCT":
            input_sec = args.min_len_htsat
        elif args.pretrain == "operaCE":
            input_sec = args.min_len_cnn
        elif args.pretrain == "operaGT":
            input_sec = 8.18

        # input_sec = 15 from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10081773

        extract_and_save_embeddings(args.pretrain, input_sec=input_sec, dim=args.dim)
