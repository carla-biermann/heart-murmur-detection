import argparse
import collections
import csv
import glob as gb
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from src.benchmark.baseline.extract_feature import (
    extract_audioMAE_feature,
    extract_clap_feature,
    extract_vgg_feature,
    extract_HeAR_feature,
)

# Directories
data_dir = "datasets/physionet.org/files/challenge-2016/1.0.0/"
feature_dir = "feature/physionet16_eval/"
OPERACT_HEART_CKPT_PATH = "cks/model/combined/circor_pascal_A_pascal_B_zchsound_clean_zchsound_noisy/encoder-operaCT-nophysionet-epoch=159--valid_acc=0.95-valid_loss=0.2932.ckpt"
OPERACT_HEART_NONOISY_CKPT_PATH = "cks/model/combined/circor_pascal_A_pascal_B_zchsound_clean/encoder-operaCT-nophysionet-nonoisy-epoch=249--valid_acc=0.95-valid_loss=0.2898.ckpt"
OPERACT_HEART_INDOMAIN_CKPT_PATH = "cks/model/combined/physionet16/encoder-operaCT-physionet16-indomain-epoch=239--valid_acc=0.98-valid_loss=0.0524.ckpt"
OPERACT_HEART_INDOMAIN_PRETRAINED_CKPT_PATH = "cks/model/combined/physionet16/encoder-operaCT-physionet16-indomain-pretrained-epoch=169--valid_acc=0.99-valid_loss=0.0300.ckpt"
OPERACT_HEART_ALL_CKPT_PATH = "cks/model/combined/circor_pascal_A_pascal_B_physionet16_zchsound_clean_zchsound_noisy/encoder-operaCT-heart-all-epoch=159--valid_acc=0.94-valid_loss=0.3790.ckpt"

# Check if audio directory exists
if not os.path.exists(data_dir):
    print(os.getcwd())
    raise FileNotFoundError(
        f"Folder not found: {data_dir}, please ensure the dataset is downloaded."
    )


def get_files_and_labels(audio_dir, annotations_dir):
    files = gb.glob(os.path.join(audio_dir, "*.wav"))
    print(f"{audio_dir}: {len(files)} files")

    ann_file = os.path.join(annotations_dir, "REFERENCE_withSQI.csv")

    label_to_int = {"normal": 0, "abnormal": 1}

    quality_dict = {}

    # Load the annotations file into a dict
    with open(ann_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # 1st col: file name, 2nd col: label (ignore), 3rd col: 
            if len(row) >= 3:
                basename = row[0].strip()
                quality = row[2].strip() # 0 for unsure, 1 for good quality
                quality_dict[basename] = quality

    labels = []
    annotations = []
    # Read label from .hea file
    for file in files:
        hea_file = file.replace(".wav", ".hea")
        with open(hea_file, "r") as f:
            lines = f.readlines()

        label = lines[-1].strip().lstrip("#").strip().lower()
        labels.append(label_to_int[label])

        basename = os.path.basename(file).split(".")[0]
        if basename in quality_dict:
            annotations.append(quality_dict[basename])
        else:
            print(f"[Warning] No annotation found for {basename}, defaulting to 0 (unsure)")
            annotations.append(0)
    
    return files, labels, annotations


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
    annotations = []

    for dir in training_dirs:
        audio_dir = os.path.join(data_dir, dir)
        annotations_dir = os.path.join(os.path.join(data_dir, "annotations/updated"), dir)
        files, y, ann = get_files_and_labels(audio_dir, annotations_dir)

        sound_files.extend(files)
        labels.extend(y)
        annotations.extend(ann)

    # Convert to arrays for safety
    sound_files = np.array(sound_files)
    labels = np.array(labels)
    annotations = np.array(annotations)

    return sound_files, labels, annotations


def preprocess_split_independent():
    sound_files, labels, annotations = read_data()

    # Save .wav file locations
    np.save(feature_dir + "sound_dir_loc.npy", sound_files)

    # Verify initial distribution
    print("Initial Class Distribution:", dict(collections.Counter(labels)))

    a_files = []
    e_files = []
    train_only_files = []
    test_only_files = []
    a_labels = []
    e_labels = []
    train_only_labels = []
    test_only_labels = []
    for i, file in enumerate(sound_files):
        if "training-a" in file:
            a_files.append(file)
            a_labels.append(labels[i])
        elif "training-e" in file:
            e_files.append(file)
            e_labels.append(labels[i])
        elif "training-b" in file or "training-c" in file:
            train_only_files.append(file)
            train_only_labels.append(labels[i])
        else:
            test_only_files.append(file)
            test_only_labels.append(labels[i])

    # Split training-a and training-e into 80% train/val and 20% test
    a_train_val, a_test, a_train_val_labels, a_test_labels = train_test_split(
        a_files, a_labels, test_size=0.2, random_state=1337, stratify=a_labels
    )
    e_train_val, e_test, e_train_val_labels, e_test_labels = train_test_split(
        e_files, e_labels, test_size=0.2, random_state=1337, stratify=e_labels
    )

    # Combine train/val datasets
    train_val_files = a_train_val + e_train_val + train_only_files
    train_val_labels = a_train_val_labels + e_train_val_labels + train_only_labels

    # Split combined train/val into 80% train and 20% validation
    x_train, x_val, y_train, y_val = train_test_split(
        train_val_files, train_val_labels, test_size=0.2, random_state=42, stratify=train_val_labels
    )

    # Split train set for in-domain pretraining
    x_train_pretrain, x_train_head = train_test_split(
        x_train, test_size=0.5, random_state=42
    )

    # Test files are from training-a, training-e (20%) and other directories
    x_test = a_test + e_test + test_only_files
    y_test = a_test_labels + e_test_labels + test_only_labels

    print("Class distribution:")
    print(f"Train: {collections.Counter(y_train)}")
    print(f"Val: {collections.Counter(y_val)}")
    print(f"Test: {collections.Counter(y_test)}")

    # Create train/val/test splits for audio files
    audio_splits = []
    audio_splits_pretrain = []
    for i, file in enumerate(sound_files):
        if file in x_train:
            audio_splits.append("train")
            if file in x_train_pretrain:
                audio_splits_pretrain.append("train_pretrain")
            if file in x_train_head:
                audio_splits_pretrain.append("train")
        elif file in x_val:
            audio_splits.append("val")
            audio_splits_pretrain.append("val")
        else:
            audio_splits.append("test")
            audio_splits_pretrain.append("test")

    # Save splits and labels
    np.save(feature_dir + "train_test_split.npy", audio_splits)
    np.save(feature_dir + "labels.npy", labels)
    np.save(feature_dir + "train_test_pretrain_split.npy", audio_splits_pretrain)
    np.save(feature_dir + "annotations.npy", annotations)


def preprocess_split():
    """Split dataset into train, val, and test sets, and save splits."""

    sound_files, labels, annotations = read_data()

    # Verify initial distribution
    print("Initial Class Distribution:", dict(collections.Counter(labels)))

    # Perform stratified splits on the sound files
    _x_train, x_test, _y_train, y_test = train_test_split(
        sound_files, labels, test_size=0.2, random_state=1337, stratify=labels
    )

    x_train, x_val, y_train, y_val = train_test_split(
        _x_train, _y_train, test_size=0.2, random_state=1337, stratify=_y_train
    )

    # Split train set for in-domain pretraining
    x_train_pretrain, x_train_head = train_test_split(
        x_train, test_size=0.5, random_state=42
    )

    print("Class distribution:")
    print(f"Train: {collections.Counter(y_train)}")
    print(f"Val: {collections.Counter(y_val)}")
    print(f"Test: {collections.Counter(y_test)}")

    # Save .wav file locations
    np.save(feature_dir + "sound_dir_loc.npy", sound_files)

    # Create train/val/test splits for audio files
    audio_splits = []
    audio_splits_pretrain = []
    for i, file in enumerate(sound_files):
        if file in x_train:
            audio_splits.append("train")
            if file in x_train_pretrain:
                audio_splits_pretrain.append("train_pretrain")
            if file in x_train_head:
                audio_splits_pretrain.append("train")
        elif file in x_val:
            audio_splits.append("val")
            audio_splits_pretrain.append("val")
        else:
            audio_splits.append("test")
            audio_splits_pretrain.append("test")

    np.save(feature_dir + "train_test_split.npy", audio_splits)
    np.save(feature_dir + "labels.npy", labels)
    np.save(feature_dir + "train_test_pretrain_split.npy", audio_splits_pretrain)
    np.save(feature_dir + "annotations.npy", annotations)


def extract_and_save_embeddings_baselines(
    feature="audiomae", fine_tuned=None, ckpt_path=None, seed=None
):
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    suffix = "" if not fine_tuned else f"_finetuned_{fine_tuned}_{seed}"

    if feature == "vggish": # no fine-tuning
        vgg_features = extract_vgg_feature(sound_dir_loc)
        np.save(feature_dir + feature + "_feature.npy", np.array(vgg_features))
    elif feature == "clap":
        clap_features = extract_clap_feature(sound_dir_loc, ckpt_path=ckpt_path,)
        np.save(feature_dir + feature + suffix + "_feature.npy", np.array(clap_features))
    elif feature == "clap2023":
        clap2023_features = extract_clap_feature(sound_dir_loc, version="2023")
        np.save(feature_dir + feature + suffix + "_feature.npy", np.array(clap2023_features))
    elif feature == "audiomae":
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc, ckpt_path=ckpt_path,)
        np.save(feature_dir + feature + suffix + "_feature.npy", np.array(audiomae_feature))
    elif feature == "hear": # no fine-tuning possible, not open-source
        hear_feature = extract_HeAR_feature(sound_dir_loc)
        np.save(feature_dir + "hear_feature.npy", np.array(hear_feature))


def extract_and_save_embeddings(
    feature="operaCE", input_sec=8, dim=1280, fine_tuned=None, ckpt_path=None, seed=None
):
    from src.benchmark.model_util import extract_opera_feature

    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    if feature == "operaCT-heart":
        ckpt_path = OPERACT_HEART_CKPT_PATH
        pretrain = "operaCT"  # necessary as input to extract_opera_feature
    elif feature == "operaCT-heart-nonoisy":
        ckpt_path = OPERACT_HEART_NONOISY_CKPT_PATH
        pretrain = "operaCT"
    elif feature == "operaCT-heart-indomain":
        ckpt_path = OPERACT_HEART_INDOMAIN_CKPT_PATH
        pretrain = "operaCT"
    elif feature == "operaCT-heart-indomain-pretrained":
        ckpt_path = OPERACT_HEART_INDOMAIN_PRETRAINED_CKPT_PATH
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
        pad0=True,
        ckpt_path=ckpt_path,
    )
    feature += str(dim)
    suffix = "" if not fine_tuned else f"_finetuned_{fine_tuned}_{seed}"
    np.save(feature_dir + feature + suffix + "_feature.npy", np.array(opera_features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--min_len_cnn", type=int, default=8)
    parser.add_argument("--min_len_htsat", type=int, default=8)
    parser.add_argument("--fine_tuned", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        preprocess_split_independent()

    if args.pretrain in ["vggish", "clap", "audiomae", "hear", "clap2023"]:
        extract_and_save_embeddings_baselines(
            args.pretrain, args.fine_tuned, args.ckpt_path, args.seed
        )
    else:
        if (
            args.pretrain == "operaCT"
            or args.pretrain == "operaCT-heart"
            or args.pretrain == "operaCT-heart-nonoisy"
            or args.pretrain == "operaCT-heart-indomain"
            or args.pretrain == "operaCT-heart-indomain-pretrained"
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
            seed=args.seed,
        )
