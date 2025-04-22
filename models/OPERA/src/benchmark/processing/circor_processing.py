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
data_dir = "datasets/circor/"
training_dir = "datasets/circor/training_data"
int_to_murmurs = {"0": "Absent", "1": "Present", "2": "Unknown"}
int_to_outcomes = {"0": "Abnormal", "1": "Normal"}
murmurs_to_int = {"Absent": "0", "Present": "1", "Unknown": "2"}
outcome_to_int = {"Abnormal": "0", "Normal": "1"}
SYSTOLIC_MURMUR_TIMING = "Systolic murmur timing"
SYSTOLIC_MURMUR_SHAPE = "Systolic murmur shape"
SYSTOLIC_MURMUR_GRADING = "Systolic murmur grading"
SYSTOLIC_MURMUR_PITCH = "Systolic murmur pitch"
SYSTOLIC_MURMUR_QUALITY = "Systolic murmur quality"
chars_to_int = {
    SYSTOLIC_MURMUR_TIMING: {
        "nan": np.nan,
        "Early-systolic": "0", 
        "Holosystolic": "1", 
        "Mid-systolic": "2",
        "Late-systolic": "3",
    },
    SYSTOLIC_MURMUR_SHAPE: {
        "nan": np.nan,
        "Decrescendo": "0", 
        "Plateau": "1", 
        "Diamond": "2", 
        "Crescendo": "3"
    },
    SYSTOLIC_MURMUR_GRADING: {"nan": np.nan, "II/VI": "0", "I/VI": "1", "III/VI": "2"},
    SYSTOLIC_MURMUR_PITCH: {"nan": np.nan, "Medium": "0", "Low": "1", "High": "2"},
    SYSTOLIC_MURMUR_QUALITY: {"nan": np.nan, "Harsh": "0", "Blowing": "1", "Musical": "2"},
}

# Check if audio directory exists
if not os.path.exists(data_dir):
    print(os.getcwd())
    raise FileNotFoundError(
        f"Folder not found: {data_dir}, please ensure the dataset is downloaded."
    )

def save_mappings_json():
    with open(feature_dir + "int_to_murmurs.json", "w") as f:
        json.dump(int_to_murmurs, f)
    with open(feature_dir + "int_to_outcomes.json", "w") as f:
        json.dump(int_to_outcomes, f)
    for c, to_int_dict in chars_to_int.items():
        int_to_dict = {v: k for k, v in to_int_dict.items()}
        with open(feature_dir + f"int_to_{'-'.join(c.lower().split(' '))}.json", "w") as f:
            json.dump(int_to_dict, f)

    print(f"Murmur Mappings: {murmurs_to_int}")
    print(f"Outcome Mappings: {outcome_to_int}")

def read_data():
    save_mappings_json()
    dirs = ["test_data", "training_data", "validation_data"]

    # Collect sound files, train_test_split and labels
    sound_files = []
    murmurs = []
    outcomes = []
    murmur_chars = {
        SYSTOLIC_MURMUR_TIMING: [],
        SYSTOLIC_MURMUR_SHAPE: [],
        SYSTOLIC_MURMUR_GRADING: [],
        SYSTOLIC_MURMUR_PITCH: [],
        SYSTOLIC_MURMUR_QUALITY: [],
    }
    audio_splits = []
    for dir in dirs:
        audio_dir = os.path.join(data_dir, dir)
        files = gb.glob(os.path.join(audio_dir, "*.wav"))
        print(f"{dir}: {len(files)} files")

        for file in files:
            pat_id = os.path.basename(file).split("_")[0]
            with open(audio_dir + f"/{pat_id}.txt", "r") as f:
                for line in f:
                    if line.startswith("#Murmur:"):
                        murmurs.append(murmurs_to_int[line.split(":")[1].strip()])
                    elif line.startswith("#Outcome:"):
                        outcomes.append(outcome_to_int[line.split(":")[1].strip()])
                    else:
                        for c in murmur_chars.keys():
                            if line.startswith(f"#{c}"):
                                murmur_chars[c].append(chars_to_int[c][line.split(":")[1].strip()])


        sound_files.extend(files)

        split = dir.split("_")[0]
        split = "val" if split == "validation" else split
        split = "train" if split == "training" else split
        audio_splits.extend([split] * len(files))

    murmurs = np.array(murmurs, dtype=np.int32)
    outcomes = np.array(outcomes, dtype=np.int32)
    for c, val in murmur_chars.items():
        np.save(feature_dir + f"{'-'.join(c.lower().split(' '))}.npy", np.array(val, dtype=np.float32))

    np.save(feature_dir + "sound_dir_loc.npy", np.array(sound_files))
    np.save(feature_dir + "train_test_split.npy", audio_splits)
    np.save(feature_dir + "murmurs.npy", murmurs)
    np.save(feature_dir + "outcomes.npy", outcomes)


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

    # Save mappings
    save_mappings_json()

    murmur_ints = [murmurs_to_int[m] for m in murmurs]
    outcome_ints = [outcome_to_int[m] for m in outcomes]

    return np.array(file_ids), np.array(murmur_ints), np.array(outcome_ints)


def preprocess_split(train_only: bool):
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
    sound_files = np.array(gb.glob(training_dir + "/*.wav"))
    np.save(feature_dir + "sound_dir_loc.npy", sound_files)

    # Create train/val/test splits for audio files
    audio_splits = []
    audio_labels = []
    outcome_labels = []
    for i, file in enumerate(sound_files):
        file_id = os.path.basename(file).split(".")[0]  # Remove ".wav" from filename
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
        np.save(feature_dir + feature + "_feature.npy", np.array(vgg_features))
    elif feature == "clap":
        clap_features = extract_clap_feature(sound_dir_loc)
        np.save(feature_dir + feature + "_feature.npy", np.array(clap_features))
    elif feature == "clap2023":
        clap2023_features = extract_clap_feature(sound_dir_loc, version="2023")
        np.save(feature_dir + feature + "_feature.npy", np.array(clap2023_features))
    elif feature == "audiomae":
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc)
        np.save(feature_dir + feature + "_feature.npy", np.array(audiomae_feature))
    elif feature == "hear":
        hear_feature = extract_HeAR_feature(sound_dir_loc)
        np.save(feature_dir + feature + "_feature.npy", np.array(hear_feature))


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
    parser.add_argument("--train_only", type=bool, default=False)

    args = parser.parse_args()

    # Check if audio directory exists
    if not os.path.exists(training_dir):
        print(os.getcwd())
        raise FileNotFoundError(
            f"Folder not found: {training_dir}, please ensure the dataset is downloaded."
        )

    feature_dir = (
        "feature/circor_eval_train_only/" if args.train_only else "feature/circor_eval/"
    )

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        preprocess_split() if args.train_only else read_data()

    if args.pretrain in ["vggish", "clap", "audiomae", "hear", "clap2023"]:
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
