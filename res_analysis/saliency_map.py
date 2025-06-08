import os
import pandas as pd
from tqdm import tqdm


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter

from src.model.models_cola import Cola
from src.model.models_mae import mae_vit_small, vit_base_patch16
from src.benchmark.model_util import get_encoder_path
from src.util import get_weights_tensor, get_split_signal_librosa


class Model(nn.Module):
    def __init__(self, pretrain, input_dim, output_dim):
        super(Model, self).__init__()
        self.pretrain = pretrain
        if pretrain == "operaCT":
            encoder = Cola(encoder="htsat")
            ckpt = torch.load("cks/model/encoder-operaCT.ckpt")
            encoder.load_state_dict(ckpt["state_dict"], strict=False)
        elif pretrain in [
            "operaCT-heart-all",
            "operaCT-heart-indomain-pretrained-physionet16",
            "operaCT-heart-indomain-pretrained-circor",
            "operaCT-heart-nonoisy-physionet16",
            "operaCT-heart-nonoisy-circor",
            "operaCT-heart-nonoisy-pascal",
            "operaCT-heart-nonoisy-zchsound",
            "operaCT-heart-nonoisy-zchsound_clean",
            "operaCT-heart-nonoisy-zchsound_noisy",
            "operaCT-heart-cross-physionet16",
            "operaCT-heart-cross-circor",
            "operaCT-heart-cross-pascal",
            "operaCT-heart-cross-zchsound",
            "operaCT-heart-cross-zchsound_clean",
            "operaCT-heart-cross-zchsound_noisy",
            "operaCT-ft-circor-murmurs",
            "operaCT-ft-circor-outcomes",
            "operaCT-ft-zchsound-clean-outcomes",
            "operaCT-ft-zchsound-clean-murmurs",
            "operaCT-ft-zchsound-noisy-outcomes",
            "operaCT-ft-zchsound-noisy-murmurs",
        ]:
            encoder = Cola(encoder="htsat")
            ckpt = torch.load(get_encoder_path(pretrain))
            encoder.load_state_dict(ckpt["state_dict"], strict=False)
            self.pretrain = "operaCT"
        elif pretrain == "operaCE":
            encoder = Cola(encoder="efficientnet")
            ckpt = torch.load("cks/model/encoder-operaCE.ckpt")
            encoder.load_state_dict(ckpt["state_dict"], strict=False)
        elif pretrain == "audiomae":
            encoder = vit_base_patch16(
                in_chans=1,
                img_size=(1024, 128),
                drop_path_rate=0.1,
                global_pool=True,
                mask_2d=False,
                use_custom_patch=False,
            ).float()
            ckpt = torch.load(get_encoder_path(pretrain), map_location=torch.device('cpu'))
            encoder.load_state_dict(ckpt["model"], strict=False)
            self.pretrain = "audiomae"
        elif "audiomae" in pretrain:
            encoder = vit_base_patch16(
                in_chans=1,
                img_size=(1024, 128),
                drop_path_rate=0.1,
                global_pool=True,
                mask_2d=False,
                use_custom_patch=False,
            ).float()
            ckpt = torch.load(get_encoder_path(pretrain), map_location=torch.device('cpu'))
            encoder.load_state_dict(ckpt["state_dict"], strict=False)
            self.pretrain = "audiomae"
        elif pretrain == "operaGT":
            encoder = mae_vit_small(
                norm_pix_loss=False,
                in_chans=1,
                audio_exp=True,
                img_size=(256, 64),
                alpha=0.0,
                mode=0,
                use_custom_patch=False,
                split_pos=False,
                pos_trainable=False,
                use_nce=False,
                decoder_mode=1,  # decoder mode 0: global attn 1: swined local attn
                mask_2d=False,
                mask_t_prob=0.7,
                mask_f_prob=0.3,
                no_shift=False,
            ).float()
            ckpt = torch.load("cks/model/encoder-operaGT.ckpt")
            encoder.load_state_dict(ckpt["state_dict"], strict=False)

        self.encoder = encoder
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Pass the input through the encoder
        if self.pretrain == "operaCT":
            x = self.encoder.extract_feature(x, dim=768)

        elif self.pretrain == "operaGT":
            x = self.encoder.forward_feature(x)

        elif self.pretrain == "audiomae":
            x = self.encoder.forward_feature(x)

        # If the encoder's output is not flattened, flatten it before passing to the linear layer
        # Assuming the encoder output is of shape (batch_size, num_features, ...)
        x = torch.flatten(x, start_dim=1)
        x = self.bn(x)
        # Pass the flattened output through the linear layer
        pre = self.fc(x)

        return pre


def compute_saliency_map(model, input_tensor, output_tensor=None):
    model.eval()

    # Ensure the input tensor requires gradients
    input_tensor.requires_grad = True

    # Forward pass
    pre = model(input_tensor)

    if output_tensor is not None:
        # regression
        Loss = nn.MSELoss()
        loss = Loss(pre, output_tensor)
        model.zero_grad()
        loss.backward()

    else:
        # classification
        target_classes = pre.argmax(dim=1)

        for i, target_class in enumerate(target_classes):
            model.zero_grad()
            pre[i, target_class].backward(retain_graph=True)

    # Compute saliency map
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()

    return saliency


def linear_evaluation_nosemic(
    use_feature="operaGT",
    method="LOOCV",
    l2_strength=1e-5,
    epochs=64,
    lr=1e-5,
    batch_size=40,
    modality="breath",
    head="linear",
):

    print("*" * 48)
    print(
        "training dataset MMLung using feature extracted by " + use_feature,
        "By sklearn",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    data_dir = "datasets/nosemic/audio/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_feature == "operaCT":
        model = Model(pretrain=use_feature, input_dim=768, output_dim=1)

    elif use_feature == "operaGT":
        model = Model(pretrain=use_feature, input_dim=384, output_dim=1)

    model.to(device)
    mode = model.float()

    labels = []
    data = []

    num = 0
    for filename in sorted(os.listdir(data_dir)):
        user, _, _, label = filename[:-4].split("_")
        labels.append(float(label))

        x = get_split_signal_librosa(
            "", data_dir + filename[:-4], spectrogram=True, input_sec=6
        )[0]  # only keep the first

        data.append(x)
        num += 1
        print(num)
        if num >= 100:
            break

    data = np.array(data)
    data = torch.tensor(data, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    print("data shape:", data.shape, "y shape:", labels.shape)
    Loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i in range(2):
        x = data[i]
        y = labels[i]
        plt.figure(figsize=(12, 6))  # Set the figure size
        X = x.cpu().detach().numpy()
        sns.heatmap(np.flipud(X.T), annot=False, cmap="magma", cbar=False)
        plt.xlabel("Time frame", fontsize=15)
        plt.ylabel("Frequency bin", fontsize=15)
        plt.savefig("fig/saliency/Nose_orignal_Spec_" + str(i) + ".png")

    labels_train = labels[50:,]
    data_train = data[50:, :, :]
    print("data shape:", data_train.shape, "y shape:", labels_train.shape)

    for epoch in range(30):
        pre = model(data_train)
        loss = Loss(pre, labels_train)
        print(epoch, loss)
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Update model parameters using optimizer
        optimizer.step()

        if loss.item() < 5:
            break

    print("test:")
    pre = model(data[:50, :, :])
    loss = Loss(pre, labels[:50,])
    print(pre, labels[:50,])
    print(epoch, loss)

    ## Test
    saliency_maps = compute_saliency_map(model, data[:2], labels[:2])
    print("saliency_map:", saliency_maps.shape)

    saliency_map = saliency_maps[0]
    saliency_map = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min()
    )

    sigma = 2  # Standard deviation for Gaussian kernel
    saliency_map = gaussian_filter(saliency_map, sigma=sigma)
    plt.figure(figsize=(12, 6))  # Set the figure size
    # plt.imshow(np.flipud(saliency_map.T), cmap='hot', interpolation='bilinear')
    sns.heatmap(np.flipud(saliency_map.T), annot=False, cmap="viridis", cbar=False)
    plt.axis("off")
    plt.savefig("fig/saliency/Nose_saliency_map_0.png")

    saliency_map = saliency_maps[1]
    saliency_map = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min()
    )

    sigma = 2  # Standard deviation for Gaussian kernel
    saliency_map = gaussian_filter(saliency_map, sigma=sigma)
    plt.figure(figsize=(12, 6))  # Set the figure size
    # plt.imshow(np.flipud(saliency_map.T), cmap='hot', interpolation='bilinear')
    sns.heatmap(np.flipud(saliency_map.T), annot=False, cmap="viridis", cbar=False)
    plt.axis("off")
    plt.savefig("fig/saliency/Nose_saliency_map_1.png")

    return model


def linear_evaluation_mmlung(
    use_feature="operaGT",
    method="LOOCV",
    l2_strength=1e-5,
    epochs=64,
    lr=1e-5,
    batch_size=40,
    modality="breath",
    label="FVC",
    head="linear",
):

    print("*" * 48)
    print(
        "training dataset MMLung using feature extracted by " + use_feature,
        "By sklearn",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    data_dir = "datasets/mmlung/Trimmed_Data_from_phone/"
    meta_dir = "datasets/mmlung/"
    feature_dir = "feature/mmlung_eval/"

    Used_modality = ["Deep_Breath_file"]  # , 'O_Single_file']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_feature == "operaCT":
        model = Model(pretrain=use_feature, input_dim=768, output_dim=1)

    elif use_feature == "operaGT":
        model = Model(pretrain=use_feature, input_dim=384, output_dim=1)

    model.to(device)
    mode = model.float()

    df = pd.read_excel(meta_dir + "All_path.xlsx")
    labels = df[label].tolist()
    labels = np.array(labels).reshape(40, 1)
    for modality in Used_modality:
        sound_dir_loc = df[modality].str.replace(" ", "_")
        loc = sound_dir_loc.tolist()
        sound_dir_loc = ["datasets/" + path for path in sound_dir_loc]
        print(sound_dir_loc)

        data = []
        for audio_file in tqdm(sound_dir_loc):
            x = get_split_signal_librosa(
                "", audio_file[:-4], spectrogram=True, input_sec=4
            )[0]  # only keep the first
            data.append(x)

        data = np.array(data)
        data = torch.tensor(data, dtype=torch.float32).to(device)

    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    print("data shape:", data.shape, "y shape:", labels.shape)
    Loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i in range(2):
        x = data[i]
        y = labels[i]
        plt.figure(figsize=(12, 6))  # Set the figure size
        X = x.cpu().detach().numpy()
        sns.heatmap(np.flipud(X.T), annot=False, cmap="magma", cbar=False)
        plt.xlabel("Time frame", fontsize=15)
        plt.ylabel("Frequency bin", fontsize=15)
        plt.savefig("fig/saliency/orignal_Spec_" + str(i) + "_FVC" + str(y) + ".png")

    labels_train = labels[2:,]
    data_train = data[2:, :, :]
    print("data shape:", data_train.shape, "y shape:", labels_train.shape)

    for epoch in range(30):
        pre = model(data_train)
        loss = Loss(pre, labels_train)
        print(epoch, loss)
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Update model parameters using optimizer
        optimizer.step()

        if loss.item() < 0.5:
            break

    print("test:")
    pre = model(data[:2, :, :])
    loss = Loss(pre, labels[:2,])
    print(pre, labels[:2,])
    print(epoch, loss)

    ##

    sigma = 2  # Standard deviation for Gaussian kernel

    saliency_maps = compute_saliency_map(model, data[:3], labels[:3])
    print("saliency_map:", saliency_maps.shape)

    saliency_map = saliency_maps[0]
    saliency_map = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min()
    )
    saliency_map = gaussian_filter(saliency_map, sigma=sigma)
    plt.figure(figsize=(12, 6))  # Set the figure size
    # plt.imshow(np.flipud(saliency_map.T), cmap='hot', interpolation='bilinear')
    sns.heatmap(np.flipud(saliency_map.T), annot=False, cmap="viridis", cbar=False)
    plt.axis("off")
    plt.savefig("fig/saliency/mmlung_saliency_map_0.png")

    saliency_map = saliency_maps[1]
    saliency_map = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min()
    )
    saliency_map = gaussian_filter(saliency_map, sigma=sigma)
    plt.figure(figsize=(12, 6))  # Set the figure size
    # plt.imshow(np.flipud(saliency_map.T), cmap='hot', interpolation='bilinear')
    sns.heatmap(np.flipud(saliency_map.T), annot=False, cmap="viridis", cbar=False)
    plt.axis("off")
    plt.savefig("fig/saliency/mmlung_saliency_map_1.png")

    saliency_map = saliency_maps[2]
    saliency_map = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min()
    )


    sigma = 2  # Standard deviation for Gaussian kernel
    saliency_map = gaussian_filter(saliency_map, sigma=sigma)
    plt.figure(figsize=(12, 6))  # Set the figure size
    # plt.imshow(np.flipud(saliency_map.T), cmap='hot', interpolation='bilinear')
    sns.heatmap(np.flipud(saliency_map.T), annot=False, cmap="viridis", cbar=False)
    plt.axis("off")
    plt.savefig("fig/saliency/mmlung_saliency_map_2.png")

    return model


def linear_evaluation_icbhidisease(
    use_feature="operaCT",
    feature_dim=768,
    l2_strength=1e-4,
    epochs=64,
    batch_size=64,
    lr=1e-4,
    head="linear",
):
    print("*" * 48)
    print(
        "training dataset icbhi disease using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    feature_dir = "feature/icbhidisease_eval/"
    y_set = np.load(feature_dir + "split.npy")
    y_label = np.load(feature_dir + "labels.npy")

    # x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()
    x_data = np.load(feature_dir + "spectrogram_pad8.npy")

    mask = (y_label == "Healthy") | (y_label == "COPD")
    y_label = y_label[mask]
    y_set = y_set[mask]
    x_data = x_data[mask]

    num_test = 2

    label_dict = {"Healthy": 0, "COPD": 1}
    y_label = np.array([label_dict[y] for y in y_label])
    print(y_label[:num_test])

    zero_indices = np.where(y_label == 0)[0]
    print(zero_indices)

    # X_train, X_test, y_train, y_test = train_test_split_from_list(x_data, y_label, y_set)

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train, y_train, test_size=0.2, random_state=1337, stratify=y_train
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, labels = x_data, y_label
    data = torch.tensor(data).to(device)

    labels = torch.tensor(labels).to(device)

    model = Model(pretrain=use_feature, input_dim=feature_dim, output_dim=2)
    # model = Model(pretrain=use_feature, input_dim=512, output_dim=1)
    model.to(device)
    # model = model.float()

    labels_train = labels[num_test:,]
    data_train = data[num_test:, :, :]
    print("data shape:", data_train.shape, "y shape:", labels_train.shape)
    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(30):
        pre = model(data_train)
        loss = Loss(pre, labels_train)
        print(epoch, loss)
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Update model parameters using optimizer
        optimizer.step()

    print("test:")
    pre = model(data[:num_test, :, :])
    loss = Loss(pre, labels[:num_test,])
    print(pre, labels[:num_test,])
    print(epoch, loss)

    ##

    saliency_maps = compute_saliency_map(
        model, data[: num_test + 5], labels[: num_test + 5]
    )
    print("saliency_map:", saliency_maps.shape)

    # saliency_map = saliency_maps[0]
    # saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # from scipy.ndimage import gaussian_filter
    # sigma = 2 # Standard deviation for Gaussian kernel
    # saliency_map = gaussian_filter(saliency_map, sigma=sigma)
    # plt.figure(figsize=(12, 6))  # Set the figure size
    # # plt.imshow(np.flipud(saliency_map.T), cmap='hot', interpolation='bilinear')
    # sns.heatmap(np.flipud(saliency_map.T), annot=False, cmap='viridis', cbar=False)
    # plt.axis('off')
    # plt.savefig('fig/saliency/icbhi_saliency_map_0.png')

    # ###
    # x = data[1]
    # y = labels[1]
    # plt.figure(figsize=(12, 6))  # Set the figure size
    # X = x.cpu().detach().numpy()
    # sns.heatmap(np.flipud(X.T), annot=False, cmap='magma', cbar=False)
    # plt.xlabel('Time frame', fontsize=15)
    # plt.ylabel('Frequency bin', fontsize=15)
    # plt.savefig('fig/saliency/icbhi_orignal_Spec_1.png')
    # saliency_map = saliency_maps[1]
    # saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # from scipy.ndimage import gaussian_filter
    # sigma = 2 # Standard deviation for Gaussian kernel
    # saliency_map = gaussian_filter(saliency_map, sigma=sigma)
    # plt.figure(figsize=(12, 6))  # Set the figure size
    # # plt.imshow(np.flipud(saliency_map.T), cmap='hot', interpolation='bilinear')
    # sns.heatmap(np.flipud(saliency_map.T), annot=False, cmap='viridis', cbar=False)
    # plt.axis('off')
    # plt.savefig('fig/saliency/icbhi_saliency_map_1.png')

    for i in range(num_test + 5):
        x = data[i]
        y = labels[i]
        plt.figure(figsize=(12, 6))  # Set the figure size
        X = x.cpu().detach().numpy()
        sns.heatmap(np.flipud(X.T), annot=False, cmap="magma", cbar=False)
        plt.xlabel("Time frame", fontsize=15)
        plt.ylabel("Frequency bin", fontsize=15)
        plt.savefig(f"fig/saliency/icbhi_{i}_orignal_Spec.png")

        saliency_map = saliency_maps[i]
        saliency_map = (saliency_map - saliency_map.min()) / (
            saliency_map.max() - saliency_map.min()
        )

        sigma = 2  # Standard deviation for Gaussian kernel
        saliency_map = gaussian_filter(saliency_map, sigma=sigma)
        plt.figure(figsize=(12, 6))  # Set the figure size
        # plt.imshow(np.flipud(saliency_map.T), cmap='hot', interpolation='bilinear')
        sns.heatmap(np.flipud(saliency_map.T), annot=False, cmap="viridis", cbar=False)
        plt.axis("off")
        plt.savefig(f"fig/saliency/icbhi_{i}_{use_feature}_saliency_map.png")


def linear_evaluation_coviduk(
    use_feature="operaCT",
    feature_dim=768,
    l2_strength=1e-6,
    epochs=64,
    lr=1e-5,
    batch_size=64,
    modality="cough",
    head="linear",
):
    print("*" * 48)
    print(
        f"training dataset covidUK of {modality} and using feature extracted by {use_feature} with l2_strength {l2_strength} lr {lr}  head"
    )

    feature_dir = "feature/coviduk_eval/"

    y_label = np.load(feature_dir + f"label_{modality}.npy")

    x_data = np.load(feature_dir + f"spectrogram_pad8_{modality}.npy")

    y_label = y_label[:128]
    x_data = x_data[:128]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, labels = x_data, y_label
    data = torch.tensor(data).to(device)

    labels = torch.tensor(labels).to(device)

    model = Model(pretrain=use_feature, input_dim=feature_dim, output_dim=2)
    model.to(device)

    num_test = 2

    labels_train = labels[num_test:,]
    data_train = data[num_test:, :, :]
    print("data shape:", data_train.shape, "y shape:", labels_train.shape)
    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(30):
        pre = model(data_train)
        loss = Loss(pre, labels_train)
        print(epoch, loss)
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Update model parameters using optimizer
        optimizer.step()

    print("test:")
    pre = model(data[:num_test, :, :])
    loss = Loss(pre, labels[:num_test,])
    print(pre, labels[:num_test,])
    print(epoch, loss)

    saliency_maps = compute_saliency_map(model, data[:num_test])
    print("saliency_map:", saliency_maps.shape)

    for i in range(num_test):
        x = data[i]
        y = labels[i]
        plt.figure(figsize=(12, 6))  # Set the figure size
        X = x.cpu().detach().numpy()
        sns.heatmap(np.flipud(X.T), annot=False, cmap="magma", cbar=False)
        plt.xlabel("Time frame", fontsize=15)
        plt.ylabel("Frequency bin", fontsize=15)
        plt.savefig(f"fig/saliency/coviduk_{modality}_{i}_orignal_Spec.png")

        saliency_map = saliency_maps[i]
        saliency_map = (saliency_map - saliency_map.min()) / (
            saliency_map.max() - saliency_map.min()
        )

        sigma = 2  # Standard deviation for Gaussian kernel
        saliency_map = gaussian_filter(saliency_map, sigma=sigma)
        plt.figure(figsize=(12, 6))  # Set the figure size
        # plt.imshow(np.flipud(saliency_map.T), cmap='hot', interpolation='bilinear')
        sns.heatmap(np.flipud(saliency_map.T), annot=False, cmap="viridis", cbar=False)
        plt.axis("off")
        plt.savefig(
            f"fig/saliency/coviduk_{modality}_{i}_{use_feature}_saliency_map.png"
        )

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def linear_evaluation_heart(
    use_feature="operaCT",
    feature_dim=768,
    l2_strength=1e-4,
    epochs=64,
    lr=1e-4,
    batch_size=32,
    head="linear",
    loss="weighted",
    dataset_name="circor",
    task="murmurs",
    feature_dir="feature/circor_eval/",
    labels_filename="murmurs.npy",
    dataset_title="CirCor",
    task_title="Murmur",
    model_title="OPERA-CT",
    method_title="Fine-tuned"
):
    print("*" * 48)
    print(
        f"training dataset {dataset_name} {task} and using feature extracted by {use_feature} with l2_strength {l2_strength} lr {lr}  head"
    )

    if "operaCT" in use_feature:
        spec_file = "spectrogram_pad8.npy"
        base_model = "operaCT"
    elif "audiomae" in use_feature:
        spec_file = "fbank_audiomae.npy"
        base_model = "audiomae"

    y_label = np.load(feature_dir + labels_filename)
    x_data = np.load(feature_dir + spec_file)
    y_set = np.load(feature_dir + "train_test_split.npy")

    data_train = x_data[y_set == "train"]
    labels_train = y_label[y_set == "train"]
    data_test = x_data[y_set == "test"]
    labels_test = y_label[y_set == "test"]

    # y_label = y_label[:128]
    # x_data = x_data[:128]
    n_cls = len(set(y_label))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data, labels = x_data, y_label
    # data = torch.tensor(data).to(device)
    # labels = torch.tensor(labels).long().to(device)
    data_train = torch.tensor(data_train)
    labels_train = torch.tensor(labels_train).long()
    data_test = torch.tensor(data_test).to(device)
    labels_test = torch.tensor(labels_test).long().to(device)

    train_dataset = FeatureDataset(data_train, labels_train)
    # test_dataset = FeatureDataset(data_test, labels_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    # )

    model = Model(pretrain=use_feature, input_dim=feature_dim, output_dim=n_cls)
    model.to(device)

    num_test = 2

    #labels_train = labels[num_test:,]
    #data_train = data[num_test:, :, :]
    print("data shape:", data_train.shape, "y shape:", labels_train.shape)
    if loss == "weighted":
        weights_tensor = get_weights_tensor(y_label, n_cls).to(device)
        print("Weights:", weights_tensor)
        Loss = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) #0.001)

    for epoch in range(30):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = Loss(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}")

    print("test:")
    #pre = model(data[:num_test, :, :])
    pre = model(data_test[:num_test, :, :])
    #loss = Loss(pre, labels[:num_test,])
    loss = Loss(pre, labels_test[:num_test,])
    print(pre, labels_test[:num_test,])
    print(epoch, loss)

    saliency_maps = compute_saliency_map(model, data_test[:num_test])
    print("saliency_map:", saliency_maps.shape)

    for i in range(num_test):
        x = data_test[i]
        y = labels_test[i]
        plt.figure(figsize=(12, 6))  # Set the figure size
        X = x.cpu().detach().numpy()
        sns.heatmap(X.T, annot=False, cmap="magma", cbar=False)
        plt.xlabel("Time frame", fontsize=15)
        plt.ylabel("Frequency bin", fontsize=15)
        # n_freq_bins = X.shape[1]
        # plt.yticks(
        #     ticks=np.linspace(0, n_freq_bins, n_freq_bins/2),
        #     labels=np.linspace(n_freq_bins - 1, 0, n_freq_bins/2, dtype=int)
        # )
        plt.savefig(f"fig/saliency/{dataset_name}_{i}_orignal_Spec_{base_model}.png")

        saliency_map = saliency_maps[i]
        saliency_map = (saliency_map - saliency_map.min()) / (
            saliency_map.max() - saliency_map.min()
        )

        sigma = 2  # Standard deviation for Gaussian kernel
        saliency_map = gaussian_filter(saliency_map, sigma=sigma)
        plt.figure(figsize=(12, 6))  # Set the figure size
        # plt.imshow(np.flipud(saliency_map.T), cmap='hot', interpolation='bilinear')
        sns.heatmap(saliency_map.T, annot=False, cmap="viridis", cbar=True)
        plt.title(f"Saliency Map Generated by {model_title} {method_title} on {dataset_title} {task_title}", fontsize=15)
        plt.xlabel("Time frame", fontsize=15)
        plt.ylabel("Frequency bin", fontsize=15)

        plt.savefig(
            f"fig/saliency/{dataset_name}_{task}_{i}_{use_feature}_saliency_map.pdf", bbox_inches='tight'
        )


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    fig_dir = "fig/saliency/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # linear_evaluation_coviduk(use_feature="operaGT", feature_dim=384)
    # linear_evaluation_coviduk(use_feature="operaCT", feature_dim=768)
    # linear_evaluation_mmlung(use_feature="operaGT", modality="breath", label="FVC")
    # linear_evaluation_nosemic(use_feature="operaCT", modality="breath")

    # H2.1 analysis:
    linear_evaluation_heart(use_feature="operaCT-ft-circor-murmurs", feature_dim=768, dataset_name='circor', task='murmurs', feature_dir="feature/circor_eval/")
    linear_evaluation_heart(use_feature="operaCT-ft-circor-outcomes", feature_dim=768, dataset_name='circor', task='outcomes', feature_dir="feature/circor_eval/", labels_filename="outcomes.npy", task_title="Outcome")
    linear_evaluation_heart(use_feature="operaCT-ft-zchsound-clean-outcomes", feature_dim=768, dataset_name='zchsound', task='clean', feature_dir="feature/zchsound_clean_eval/", labels_filename="labels.npy", dataset_title="ZCHSound clean", task_title="Diagnosis")
    linear_evaluation_heart(use_feature="operaCT-ft-zchsound-clean-murmurs", feature_dim=768, dataset_name='zchsound_clean', task='murmurs', feature_dir="feature/zchsound_clean_eval/", labels_filename="murmurs.npy", dataset_title="ZCHSound clean", task_title="Outcome")
    # linear_evaluation_heart(use_feature="operaCT-ft-zchsound-noisy-outcomes", feature_dim=768, dataset_name='zchsound', task='noisy', feature_dir="feature/zchsound_noisy_eval/", labels_filename="labels.npy", dataset_title="ZCHSound noisy", task_title="Diagnosis")
    # linear_evaluation_heart(use_feature="operaCT-ft-zchsound-noisy-murmurs", feature_dim=768, dataset_name='zchsound_noisy', task='murmurs', feature_dir="feature/zchsound_noisy_eval/", labels_filename="murmurs.npy", dataset_title="ZCHSound noisy", task_title="Outcome")