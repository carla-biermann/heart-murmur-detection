import glob as gb
import argparse
import collections
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import json
import os


# Directories
data_dir = "datasets/PASCAL/"

# Check if audio directory exists
if not os.path.exists(data_dir):
    print(os.getcwd())
    raise FileNotFoundError(f"Folder not found: {data_dir}, please ensure the dataset is downloaded.")

def get_labels(dataset):
    """Read labels from ZCHSound CSV file and create mappings."""
    if dataset == 'A':
        label_to_int = {'normal': 0, 'murmur': 1, 'extrahls': 2, 'artifact': 3}
        int_to_label = {0: 'normal', 1: 'murmur', 2: 'extrahls', 3: 'artifact'}  
    elif dataset == 'B':
        label_to_int = {'normal': 0, 'murmur': 1, 'extrastole': 2}
        int_to_label = {0: 'normal', 1: 'murmur', 2: 'extrastole'}  
    else:
        raise ValueError(f"Please input a valid value for dataset: A or B.")

    # Save mappings
    with open(feature_dir + "label_to_int.json", "w") as f:
        json.dump(label_to_int, f)
    with open(feature_dir + "int_to_label.json", "w") as f:
        json.dump(int_to_label, f)

    return label_to_int


def preprocess_split(dataset):
    """Split dataset into train, val, and test sets, and save splits."""
    
    label_to_int = get_labels(dataset)
    int_to_label = {v: k for k, v in label_to_int.items()}

    dirs_A = ['Atraining_artifact', 'Atraining_extrahls', 'Atraining_murmur', 'Atraining_normal']
    dirs_B = ['Btraining_extrastole', 'Btraining_murmur', 'BTraining_normal']
    
    if dataset == 'A':
        dirs = dirs_A
    elif dataset == 'B':
        dirs = dirs_B
    else:
        raise ValueError(f"Please input a valid value for dataset: A or B.")
    
    # Collect sound files and labels
    sound_files = []
    labels = []
    for dir in dirs:
        audio_dir = os.path.join(data_dir, dir)
        label = label_to_int[dir.split('_')[1]]
        files = gb.glob(os.path.join(audio_dir, '*.wav'))
        print(f"{dir}: {len(files)} files")
        
        sound_files.extend(files)
        labels.extend([label] * len(files))

    # Convert to arrays for safety
    sound_files = np.array(sound_files)
    labels = np.array(labels)

    # Verify initial distribution
    print("Initial Class Distribution:", dict(collections.Counter(labels)))

    # Perform stratified splits on the actual sound files
    _x_train, x_test, _y_train, y_test = train_test_split(
        sound_files, labels, 
        test_size=0.2, 
        random_state=1337, 
        stratify=labels
    )

    x_train, x_val, y_train, y_val = train_test_split(
        _x_train, _y_train, 
        test_size=0.2,  # 0.25 * 0.8 = 0.2 of total dataset
        random_state=1337, 
        stratify=_y_train
    )

    # Print distributions
    def print_distribution(name, y):
        dist = collections.Counter(y)
        readable = {int_to_label[k]: v for k, v in dist.items()}
        print(f"{name} Distribution:", readable)

    print_distribution("Train", y_train)
    print_distribution("Val", y_val)
    print_distribution("Test", y_test)

    # Save splits and labels
    np.save(feature_dir + "x_train.npy", x_train)
    np.save(feature_dir + "y_train.npy", y_train)
    np.save(feature_dir + "x_val.npy", x_val)
    np.save(feature_dir + "y_val.npy", y_val)
    np.save(feature_dir + "x_test.npy", x_test)
    np.save(feature_dir + "y_test.npy", y_test)

def extract_and_save_embeddings(feature="operaCE", input_sec=8, dim=1280):
    from src.benchmark.model_util import extract_opera_feature
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    opera_features = extract_opera_feature(
        sound_dir_loc,  pretrain=feature, input_sec=input_sec, dim=dim)
    feature += str(dim)
    np.save(feature_dir + feature + "_feature.npy", np.array(opera_features))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--min_len_cnn", type=int, default=8)
    parser.add_argument("--min_len_htsat", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="A")

    args = parser.parse_args()
    if args.dataset == 'A':
        feature_dir = "feature/pascal_eval_A/"
    elif args.dataset == 'B':
        feature_dir = "feature/pascal_eval_B/"
    else:
        raise ValueError(f"Please input a valid value for dataset: A or B.")

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        preprocess_split(args.dataset)

    if args.pretrain == "operaCT":
        input_sec = args.min_len_htsat
    elif args.pretrain == "operaCE":
        input_sec = args.min_len_cnn
    elif args.pretrain == "operaGT":
        input_sec = 8.18
    #extract_and_save_embeddings(
    #    args.pretrain, input_sec=input_sec, dim=args.dim)
