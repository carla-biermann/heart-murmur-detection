import argparse
import os

import numpy as np
from tqdm import tqdm

from src.util import get_entire_signal_librosa

# for pretraining


def preprocess_spectrogram_SSL(feature_dir: str, input_sec=8):
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    y_set = np.load(feature_dir + "train_test_split.npy")

    sound_dir_loc_train_val = sound_dir_loc[(y_set == "train") | (y_set == "val")]

    invalid_data = 0

    filename_list = []

    # Only use train and val data
    for audio_file in tqdm(sound_dir_loc_train_val):
        file_id = audio_file.split("/")[-1][:-4]

        data = get_entire_signal_librosa(
            "", audio_file[:-4], spectrogram=True, input_sec=input_sec
        )

        if data is None:
            invalid_data += 1
            continue

        os.makedirs(feature_dir + "entire_spec_npy", exist_ok=True)

        # saving to individual npy files
        np.save(feature_dir + "entire_spec_npy/" + file_id + ".npy", data)
        filename_list.append(feature_dir + "entire_spec_npy/" + file_id)

    np.save(feature_dir + "entire_spec_filenames.npy", filename_list)
    print(
        f"finished preprocessing {feature_dir.split('/')[1].removesuffix('_eval')}: valid data",
        len(filename_list),
        "; invalid data",
        invalid_data,
    )


# finished preprocessing circor: valid data 3614 ; invalid data 35
# finished preprocessing physionet16: valid data 2580 ; invalid data 12
# finished preprocessing zchsound_clean: valid data 752 ; invalid data 0
# finished preprocessing zchsound_noisy: valid data 253 ; invalid data 1
# finished preprocessing pascal_A: valid data 96 ; invalid data 3 -> input_sec=2
# finished preprocessing pascal_B: valid data 217 ; invalid data 32 -> input_sec=2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="feature/circor_eval/")
    parser.add_argument("--input_sec", type=int, default=8)
    args = parser.parse_args()

    preprocess_spectrogram_SSL(feature_dir=args.feature_dir, input_sec=args.input_sec)
