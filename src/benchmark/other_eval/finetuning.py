import collections
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.model_selection import train_test_split
from torchlibrosa.augmentation import SpecAugmentation
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import ViTConfig, ViTModel
import wandb

from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model, get_audiomae_encoder_path
from src.model.models_eval import (
    AudioClassifier,
    AudioClassifierAudioMAE,
    AudioClassifierCLAP,
    AudioClassifierHeAR,
)
from src.util import (
    crop_first,
    random_crop,
    random_mask,
    random_multiply,
    train_test_split_from_list,
    get_weights_tensor,
)

torch.backends.cudnn.deterministic = True


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        max_len=256,
        augment=True,
        from_npy=False,
        crop_mode="first",
        from_audio=False,
        spec_augment=False,
        time_drop_width=100,
        time_stripes_num=2,
        freq_drop_width=20,
        freq_stripes_num=2,
    ):
        self.data = data[0]
        self.label = data[1]
        self.annotations = data[2] if len(data) > 2 else None
        self.max_len = max_len
        self.augment = augment
        self.from_npy = from_npy
        self.crop_mode = crop_mode
        self.from_audio = from_audio
        self.spec_augment = spec_augment
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=time_drop_width,
            time_stripes_num=time_stripes_num,
            freq_drop_width=freq_drop_width,
            freq_stripes_num=freq_stripes_num,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.from_npy:
            npy_path = self.data[idx]
            x = np.load(npy_path + ".npy")
        else:
            x = self.data[idx]

        label = self.label[idx]

        if self.from_audio:
            if self.annotations is not None:
                annotation = self.annotations[idx]
                annotation = torch.tensor(annotation, dtype=torch.long)
                return x, label, annotation
            else:
                return x, label

        if self.max_len:
            if self.crop_mode == "random":
                x = random_crop(x, crop_size=self.max_len)
            else:
                x = crop_first(x, crop_size=self.max_len)

        if self.augment:
            x = random_mask(x)
            x = random_multiply(x)

        x = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        if self.spec_augment:
            original_dim = x.ndim
            if original_dim == 2:
                x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
            elif original_dim == 3:
                x = x.unsqueeze(0)               # (1, C, F, T)
            
            x = self.spec_augmenter(x)           # requires 4 dim

            if original_dim == 2:
                x = x.squeeze(0).squeeze(0)     # (F, T)
            elif original_dim == 3:
                x = x.squeeze(0)                # (1, C, F, T)

        if self.annotations is not None:
            annotation = self.annotations[idx]
            annotation = torch.tensor(annotation, dtype=torch.long)
            return x, label, annotation
        else:
            return x, label


class DecayLearningRate(pl.Callback):
    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = []
            for param_group in optimizer.param_groups:
                group.append(param_group["lr"])
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.99
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group


def finetune_covid19sounds(
    task=1,
    pretrain="operaCE",
    modality="cough",
    epochs=64,
    batch_size=64,
    l2_strength=1e-4,
    lr=1e-4,
    head="linear",
    feat_dim=1280,
):
    print(
        "fine-tuning covid19 task",
        task,
        modality,
        "from model pretrained on",
        pretrain,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "*" * 28,
    )
    folders = {
        1: "feature/covid19sounds_eval/",
        2: "feature/task2_eval/",
        "asthma": "feature/asthma_eval/",
        "1downsample": "feature/covid19sounds_eval/downsampled/",
    }
    feature_dir = folders[task]

    from_audio = False

    if pretrain == "null":
        # from scratch
        lr = 1e-4

    if pretrain == "audiomae":
        from src.benchmark.baseline.audioMAE.models_mae import (
            vit_base_patch16,
        )

        if not os.path.exists(feature_dir + "fbank_audiomae.npy"):
            from src.util import get_split_signal_fbank_pad

            sound_dir_loc = np.load(feature_dir + f"sound_dir_loc_{modality}.npy")
            x_data = []
            for audio_file in sound_dir_loc:
                data = get_split_signal_fbank_pad(
                    "", audio_file[:-4], spectrogram=True, input_sec=10, trim_tail=False
                )[0]
                # print(data.shape)
                x_data.append(data)
            x_data = np.array(x_data)
            print(x_data.shape)
            np.save(feature_dir + "fbank_audiomae.npy", x_data)

        x_data = np.load(feature_dir + "fbank_audiomae.npy")

        encoder_path = "src/benchmark/baseline/audioMAE/ViTB_pretrained.pth"
        ckpt = torch.load(encoder_path)
        net = vit_base_patch16(
            in_chans=1,
            img_size=(1024, 128),
            drop_path_rate=0.1,
            global_pool=True,
            mask_2d=False,
            use_custom_patch=False,
        )

        net.load_state_dict(ckpt["model"], strict=False)

        model = AudioClassifierAudioMAE(
            net=net,
            head=head,
            classes=2,
            lr=lr,
            l2_strength=l2_strength,
            feat_dim=feat_dim,
        )
    elif pretrain == "clap":
        from src.benchmark.baseline.msclap import CLAP

        audio_files = np.load(feature_dir + f"sound_dir_loc_{modality}.npy")
        x_data = np.array(audio_files)
        clap_model = CLAP(version="2022", use_cuda=True)
        net = clap_model.clap.audio_encoder
        model = AudioClassifierCLAP(
            net=net,
            head=head,
            classes=2,
            lr=lr,
            l2_strength=l2_strength,
            feat_dim=feat_dim,
        )
        from_audio = True
    else:
        if not os.path.exists(feature_dir + f"spectrogram_pad8_{modality}.npy"):
            from src.util import get_split_signal_librosa

            sound_dir_loc = np.load(feature_dir + f"sound_dir_loc_{modality}.npy")
            x_data = []
            for audio_file in sound_dir_loc:
                data = get_split_signal_librosa(
                    "",
                    audio_file[:-4],
                    spectrogram=True,
                    input_sec=8.18,
                    trim_tail=False,
                )[0]
                # print(data.shape)
                x_data.append(data)
            x_data = np.array(x_data)
            np.save(feature_dir + f"spectrogram_pad8_{modality}.npy", x_data)

        x_data = np.load(feature_dir + f"spectrogram_pad8_{modality}.npy")
        pretrained_model = initialize_pretrained_model(pretrain)
        if pretrain == "null":
            lr = 1e-4
            epochs = 64
            print("-" * 20 + "training from scratch")
        else:
            encoder_path = get_encoder_path(pretrain)
            print("loading weights from", encoder_path)
            ckpt = torch.load(encoder_path)
            pretrained_model.load_state_dict(ckpt["state_dict"], strict=False)

        if "mae" in pretrain or "GT" in pretrain:
            model = AudioClassifierAudioMAE(
                net=pretrained_model,
                classes=2,
                lr=lr,
                l2_strength=l2_strength,
                feat_dim=feat_dim,
            )
        else:
            freeze_encoder = "early" if pretrain == "operaCE" else "none"
            net = pretrained_model.encoder
            model = AudioClassifier(
                net=net,
                head=head,
                classes=2,
                lr=lr,
                l2_strength=l2_strength,
                feat_dim=feat_dim,
                freeze_encoder=freeze_encoder,
            )

    print(x_data.shape)
    y_label = np.load(feature_dir + "labels.npy")
    y_set = np.load(feature_dir + "data_split.npy")

    if task == 1 or task == "1downsample":
        x_data_train = x_data[y_set == 0]
        y_label_train = y_label[y_set == 0]
        x_data_vad = x_data[y_set == 1]
        y_label_vad = y_label[y_set == 1]
        x_data_test = x_data[y_set == 2]
        y_label_test = y_label[y_set == 2]
    else:
        raise NotImplementedError(
            f"Task not implemented: Covid-19 sounds task {task}, please check the cfg."
        )

    print(collections.Counter(y_label_train))
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))

    train_data = AudioDataset(
        (x_data_train, y_label_train),
        augment=False,
        max_len=False,
        from_audio=from_audio,
    )
    test_data = AudioDataset(
        (x_data_test, y_label_test), augment=False, max_len=False, from_audio=from_audio
    )
    val_data = AudioDataset(
        (x_data_vad, y_label_vad), augment=False, max_len=False, from_audio=from_audio
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=8, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=8, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=8
    )

    logger = CSVLogger(
        save_dir="cks/logs/finetune",
        name="covid-task" + str(task) + modality,
        version="_".join(
            [head, pretrain, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/finetune/task" + str(task) + "/" + modality + "/",
        filename="_".join(
            [
                "finetuning",
                head,
                pretrain,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="valid_auc", min_delta=0.001, patience=10, verbose=True, mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=21,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=train_loader)
    trainer.test(dataloaders=val_loader)
    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    return auc


def finetune_ssbpr(
    n_cls=5,
    pretrain="operaCE",
    l2_strength=1e-4,
    epochs=64,
    batch_size=64,
    lr=1e-4,
    head="linear",
    feat_dim=1280,
):
    print("*" * 48)
    print(
        "training dataset SSBPR from model pretrained on",
        pretrain,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    feature_dir = "feature/snoring_eval/"

    from_audio = False
    if pretrain == "audiomae":
        from src.benchmark.baseline.audioMAE.models_mae import vit_base_patch16

        if not os.path.exists(feature_dir + "fbank_audiomae.npy"):
            from src.util import get_split_signal_fbank_pad

            sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
            x_data = []
            for audio_file in sound_dir_loc:
                data = get_split_signal_fbank_pad(
                    "", audio_file[:-4], spectrogram=True, input_sec=10, trim_tail=False
                )[0]
                # print(data.shape)
                x_data.append(data)
            x_data = np.array(x_data)
            print(x_data.shape)
            np.save(feature_dir + "fbank_audiomae.npy", x_data)

        x_data = np.load(feature_dir + "fbank_audiomae.npy")

        encoder_path = "src/benchmark/baseline/audioMAE/ViTB_pretrained.pth"
        ckpt = torch.load(encoder_path)
        net = vit_base_patch16(
            in_chans=1,
            img_size=(1024, 128),
            drop_path_rate=0.1,
            global_pool=True,
            mask_2d=False,
            use_custom_patch=False,
        )

        net.load_state_dict(ckpt["model"], strict=False)

        model = AudioClassifierAudioMAE(
            net=net,
            head=head,
            classes=n_cls,
            lr=lr,
            l2_strength=l2_strength,
            feat_dim=feat_dim,
        )
    elif pretrain == "clap":
        from src.benchmark.baseline.msclap import CLAP

        audio_files = np.load(feature_dir + "sound_dir_loc.npy")
        x_data = np.array(audio_files)
        clap_model = CLAP(version="2022", use_cuda=True)
        net = clap_model.clap.audio_encoder
        model = AudioClassifierCLAP(
            net=net,
            head=head,
            classes=n_cls,
            lr=lr,
            l2_strength=l2_strength,
            feat_dim=feat_dim,
        )
        from_audio = True
    else:
        if not os.path.exists(feature_dir + "spectrogram_pad4.npy"):
            from src.util import get_split_signal_librosa

            sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
            x_data = []
            for audio_file in sound_dir_loc:
                data = get_split_signal_librosa(
                    "",
                    audio_file[:-4],
                    spectrogram=True,
                    input_sec=4.09,
                    trim_tail=False,
                )[0]
                # print(data.shape)
                x_data.append(data)
            x_data = np.array(x_data)
            np.save(feature_dir + "spectrogram_pad4.npy", x_data)

        x_data = np.load(feature_dir + "spectrogram_pad4.npy")
        pretrained_model = initialize_pretrained_model(pretrain)
        if pretrain == "null":
            lr = 1e-4
            epochs = 64
            print("-" * 20 + "training from scratch")
        else:
            encoder_path = get_encoder_path(pretrain)
            print("loading weights from", encoder_path)
            ckpt = torch.load(encoder_path)
            pretrained_model.load_state_dict(ckpt["state_dict"], strict=False)

        if "mae" in pretrain or "GT" in pretrain:
            model = AudioClassifierAudioMAE(
                net=pretrained_model,
                classes=n_cls,
                lr=lr,
                l2_strength=l2_strength,
                feat_dim=feat_dim,
            )
        else:
            freeze_encoder = "early" if pretrain == "operaCE" else "none"
            net = pretrained_model.encoder
            model = AudioClassifier(
                net=net,
                head=head,
                classes=n_cls,
                lr=lr,
                l2_strength=l2_strength,
                feat_dim=feat_dim,
                freeze_encoder=freeze_encoder,
            )

    y_label = np.load(feature_dir + "labels.npy")
    print(collections.Counter(y_label))

    train_ratio = 0.6
    validation_ratio = 0.2
    test_ratio = 0.2

    seed = 42
    _x_train, x_data_test, _y_train, y_label_test = train_test_split(
        x_data, y_label, test_size=test_ratio, random_state=seed, stratify=y_label
    )

    x_data_train, x_data_vad, y_label_train, y_label_vad = train_test_split(
        _x_train,
        _y_train,
        test_size=validation_ratio / (validation_ratio + train_ratio),
        random_state=seed,
        stratify=_y_train,
    )

    print(collections.Counter(y_label_train))
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))

    train_data = AudioDataset(
        (x_data_train, y_label_train),
        augment=False,
        max_len=False,
        from_audio=from_audio,
    )
    test_data = AudioDataset(
        (x_data_test, y_label_test), augment=False, max_len=False, from_audio=from_audio
    )
    val_data = AudioDataset(
        (x_data_vad, y_label_vad), augment=False, max_len=False, from_audio=from_audio
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    logger = CSVLogger(
        save_dir="cks/logs/finetune",
        name="ssbpr",
        version="_".join(
            [head, pretrain, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/finetune/ssbpr/",
        filename="_".join(
            [
                "finetuning",
                head,
                pretrain,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
        every_n_epochs=3,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=train_loader)
    trainer.test(dataloaders=val_loader)
    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset SSBPR from model pretrained on",
        pretrain,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )
    return auc


def finetune_icbhidisease(
    n_cls=2,
    pretrain="operaCE",
    l2_strength=1e-4,
    epochs=64,
    batch_size=64,
    lr=1e-4,
    head="linear",
    feat_dim=1280,
):
    print("*" * 48)
    print(
        "training dataset ICBHI disease from model pretrained on",
        pretrain,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    feature_dir = "feature/icbhidisease_eval/"

    from_audio = False
    if pretrain == "audiomae":
        from src.benchmark.baseline.audioMAE.models_mae import (
            vit_base_patch16,
        )

        if not os.path.exists(feature_dir + "fbank_audiomae.npy"):
            from src.util import get_split_signal_fbank_pad

            sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
            x_data = []
            for audio_file in sound_dir_loc:
                data = get_split_signal_fbank_pad(
                    "", audio_file[:-4], spectrogram=True, input_sec=10, trim_tail=False
                )[0]
                # print(data.shape)
                x_data.append(data)
            x_data = np.array(x_data)
            print(x_data.shape)
            np.save(feature_dir + "fbank_audiomae.npy", x_data)

        x_data = np.load(feature_dir + "fbank_audiomae.npy")

        encoder_path = "src/benchmark/baseline/audioMAE/ViTB_pretrained.pth"
        ckpt = torch.load(encoder_path)
        net = vit_base_patch16(
            in_chans=1,
            img_size=(1024, 128),
            drop_path_rate=0.1,
            global_pool=True,
            mask_2d=False,
            use_custom_patch=False,
        )

        net.load_state_dict(ckpt["model"], strict=False)

        model = AudioClassifierAudioMAE(
            net=net,
            head=head,
            classes=n_cls,
            lr=lr,
            l2_strength=l2_strength,
            feat_dim=feat_dim,
        )

    elif pretrain == "clap":
        from src.benchmark.baseline.msclap import CLAP

        audio_files = np.load(feature_dir + "sound_dir_loc.npy")
        x_data = np.array(audio_files)
        clap_model = CLAP(version="2022", use_cuda=True)
        net = clap_model.clap.audio_encoder
        model = AudioClassifierCLAP(
            net=net,
            head=head,
            classes=n_cls,
            lr=lr,
            l2_strength=l2_strength,
            feat_dim=feat_dim,
        )
        from_audio = True

    else:
        if not os.path.exists(feature_dir + "spectrogram_pad8.npy"):
            from src.util import get_split_signal_librosa

            sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
            x_data = []
            for audio_file in sound_dir_loc:
                data = get_split_signal_librosa(
                    "",
                    audio_file[:-4],
                    spectrogram=True,
                    input_sec=8.18,
                    trim_tail=False,
                )[0]
                # print(data.shape)
                x_data.append(data)
            x_data = np.array(x_data)
            np.save(feature_dir + "spectrogram_pad8.npy", x_data)

        x_data = np.load(feature_dir + "spectrogram_pad8.npy")
        pretrained_model = initialize_pretrained_model(pretrain)
        if pretrain == "null":
            lr = 1e-4
            epochs = 64
            print("-" * 20 + "training from scratch")
        else:
            encoder_path = get_encoder_path(pretrain)
            print("loading weights from", encoder_path)
            ckpt = torch.load(encoder_path)
            pretrained_model.load_state_dict(ckpt["state_dict"], strict=False)

        if "mae" in pretrain or "GT" in pretrain:
            model = AudioClassifierAudioMAE(
                net=pretrained_model,
                classes=n_cls,
                lr=lr,
                l2_strength=l2_strength,
                feat_dim=feat_dim,
            )
        else:
            freeze_encoder = "early" if pretrain == "operaCE" else "none"
            net = pretrained_model.encoder
            model = AudioClassifier(
                net=net,
                head=head,
                classes=n_cls,
                lr=lr,
                l2_strength=l2_strength,
                feat_dim=feat_dim,
                freeze_encoder=freeze_encoder,
            )

    y_set = np.load(feature_dir + "split.npy")
    y_label = np.load(feature_dir + "labels.npy")
    print(collections.Counter(y_label))

    mask = (y_label == "Healthy") | (y_label == "COPD")
    y_label = y_label[mask]
    y_set = y_set[mask]
    x_data = x_data[mask]

    label_dict = {"Healthy": 0, "COPD": 1}
    y_label = np.array([label_dict[y] for y in y_label])

    x_data_train, x_data_test, y_label_train, y_label_test = train_test_split_from_list(
        x_data, y_label, y_set
    )

    x_data_train, x_data_vad, y_label_train, y_label_vad = train_test_split(
        x_data_train,
        y_label_train,
        test_size=0.2,
        random_state=1337,
        stratify=y_label_train,
    )

    print(collections.Counter(y_label_train))
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))

    train_data = AudioDataset(
        (x_data_train, y_label_train),
        augment=False,
        max_len=False,
        from_audio=from_audio,
    )
    test_data = AudioDataset(
        (x_data_test, y_label_test), augment=False, max_len=False, from_audio=from_audio
    )
    val_data = AudioDataset(
        (x_data_vad, y_label_vad), augment=False, max_len=False, from_audio=from_audio
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    logger = CSVLogger(
        save_dir="cks/logs/finetune",
        name="icbhi",
        version="_".join(
            [head, pretrain, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/finetune/icbhi/",
        filename="_".join(
            [
                "finetuning",
                head,
                pretrain,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
        every_n_epochs=3,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=train_loader)
    trainer.test(dataloaders=val_loader)
    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset icbhi disease from model pretrained on",
        pretrain,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )
    return auc


def get_wandb_name(use_feature, data, head):
    s = time.gmtime(time.time())
    return f"{time.strftime('%Y-%m-%d %H:%M:%S', s)}-{use_feature}-{data}-{head}"


def finetune_heart(
    seed,
    pretrain="operaCE",
    l2_strength=1e-4,
    epochs=64,
    batch_size=64,
    lr=1e-4,
    head="linear",
    loss="unweighted",
    feat_dim=1280,
    dataset_name="circor",
    task="murmurs",
    feature_dir="feature/circor_eval/",
    labels_filename="murmurs.npy",
    freeze_encoder="none",  # Control freezing
    spec_augment=False,
):
    run_name = get_wandb_name(pretrain, f"{dataset_name}-{task}", head)
    wandb_logger = WandbLogger(
        project="Heart-Sound-Analysis-FT",
        name=run_name,
        log_model=True,
    )

    metrics = [
        "weighted_accuracy",
        "weighted_auroc",
        "weighted_specificity",
        "weighted_recall",
        "weighted_F1",
        "unweighted_recall",
        "avg_unweighted_recall",
        "unweighted_precision",
        "avg_unweighted_precision",
        "unweighted_specificity",
        "avg_unweighted_specificity",
        "circor_weighted_murmur_acc",
        "circor_weighted_outcome_acc",
        "unweighted_accuracy",
        "circor_outcome_cost",
        "macro_F1",
        "macro_auroc",
        "physionet16_score",
    ]

    print("*" * 48)
    print(
        f"training dataset {dataset_name} {task} from model pretrained on",
        pretrain,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    y_set = np.load(feature_dir + "train_test_split.npy")
    y_label = np.load(feature_dir + labels_filename)

    # Filter out NaN values (Circor murmur characteristics)
    valid_indices = ~np.isnan(y_label)
    y_label = y_label[valid_indices].astype(np.int32)
    y_set = y_set[valid_indices]

    n_cls = len(set(y_label))

    print(f"Label distribution: {collections.Counter(y_label)}")
    print(f"Unique labels: {collections.Counter(y_set)}")
    y_label_train = y_label[y_set == "train"]

    if loss == "weighted":
        weights_tensor = get_weights_tensor(y_label_train, n_cls)
        print("Weights:", weights_tensor)
        loss_func = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        loss_func = None

    from_audio = False
    if "audiomae" in pretrain:
        from src.benchmark.baseline.audioMAE.models_mae import (
            vit_base_patch16,
        )

        time_drop_width = 100
        freq_drop_width = 20

        if not os.path.exists(feature_dir + "fbank_audiomae.npy"):
            from src.util import get_split_signal_fbank_pad

            sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
            x_data = []
            for audio_file in sound_dir_loc:
                data = get_split_signal_fbank_pad(
                    "", audio_file[:-4], spectrogram=True, input_sec=10, trim_tail=False
                )[0]
                # print(data.shape)
                x_data.append(data)
            x_data = np.array(x_data)
            print(x_data.shape)
            np.save(feature_dir + "fbank_audiomae.npy", x_data)

        x_data = np.load(feature_dir + "fbank_audiomae.npy")
        batch_size = 32

        encoder_path = get_audiomae_encoder_path(pretrain)
        ckpt = torch.load(encoder_path)
        net = vit_base_patch16(
            in_chans=1,
            img_size=(1024, 128),
            drop_path_rate=0.1,
            global_pool=True,
            mask_2d=False,
            use_custom_patch=False,
        )

        try:
            net.load_state_dict(ckpt["model"], strict=False)
        except KeyError:
            net.load_state_dict(ckpt["state_dict"], strict=False)

        model = AudioClassifierAudioMAE(
            net=net,
            head=head,
            classes=n_cls,
            lr=lr,
            l2_strength=l2_strength,
            feat_dim=feat_dim,
            loss_func=loss_func,
            freeze_encoder=freeze_encoder,
            metrics=metrics,
            dataset=dataset_name,
            task=task
        )

    elif pretrain == "clap":
        from src.benchmark.baseline.msclap import CLAP

        time_drop_width = 64
        freq_drop_width = 8

        audio_files = np.load(feature_dir + "sound_dir_loc.npy")
        x_data = np.array(audio_files)
        clap_model = CLAP(version="2022", use_cuda=True)
        net = clap_model.clap.audio_encoder
        model = AudioClassifierCLAP(
            net=net,
            head=head,
            classes=n_cls,
            lr=lr,
            l2_strength=l2_strength,
            feat_dim=feat_dim,
            metrics=metrics,
            dataset=dataset_name,
            task=task,
            loss_func=loss_func,
        )
        from_audio = True
    elif pretrain == "clap2023":
        from src.benchmark.baseline.msclap import CLAP

        time_drop_width = 64
        freq_drop_width = 8

        audio_files = np.load(feature_dir + "sound_dir_loc.npy")
        x_data = np.array(audio_files)
        clap_model = CLAP(version="2023", use_cuda=True)
        net = clap_model.clap.audio_encoder
        model = AudioClassifierCLAP(
            net=net,
            head=head,
            classes=n_cls,
            lr=lr,
            l2_strength=l2_strength,
            feat_dim=feat_dim,
            metrics=metrics,
            dataset=dataset_name,
            task=task,
            loss_func=loss_func,
        )
        from_audio = True
    elif pretrain == "hear":
        time_drop_width = 0
        freq_drop_width = 0
        if not os.path.exists(feature_dir + "fbank_hear.npy"):
            from src.util import get_split_signal_fbank_pad

            sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
            x_data = []
            for audio_file in sound_dir_loc:
                data = get_split_signal_fbank_pad(
                    "", audio_file[:-4], spectrogram=False, sample_rate=16000, input_sec=2, trim_tail=False
                )[0]
                x_data.append(data)
            x_data = np.array(x_data)
            print(x_data.shape)
            np.save(feature_dir + "fbank_hear.npy", x_data)

        x_data = np.load(feature_dir + "fbank_hear.npy")
        feat_dim = 1024
        batch_size = 16
        configuration = ViTConfig(
            image_size=(192, 128),
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=1024 * 4,
            hidden_act="gelu_fast",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-6,
            pooled_dim=512,
            patch_size=16,
            num_channels=1,
            qkv_bias=True,
            encoder_stride=16,
            pooler_act='linear',
            pooler_output_size=512,
        )
        pretrained_model = ViTModel.from_pretrained(
            "google/hear-pytorch",
            config=configuration,
            ignore_mismatched_sizes=True # doesn't work without, 
        )
        model = AudioClassifierHeAR(
            net=pretrained_model,
            head=head,
            classes=n_cls,
            lr=lr,
            l2_strength=l2_strength,
            feat_dim=feat_dim,
            loss_func=loss_func,
            metrics=metrics,
            dataset=dataset_name,
            task=task
        )
    else:
        time_drop_width = 40
        freq_drop_width = 8
        if not os.path.exists(feature_dir + "spectrogram_pad8.npy"):
            from src.util import get_split_signal_librosa

            sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
            x_data = []
            for audio_file in sound_dir_loc:
                data = get_split_signal_librosa(
                    "",
                    audio_file[:-4],
                    spectrogram=True,
                    input_sec=8.18,
                    trim_tail=False,
                )[0]
                # print(data.shape)
                x_data.append(data)
            x_data = np.array(x_data)
            np.save(feature_dir + "spectrogram_pad8.npy", x_data)

        x_data = np.load(feature_dir + "spectrogram_pad8.npy")
        if (
            pretrain == "operaCT-heart-indomain" or 
            pretrain == "operaCT-heart-indomain-pretrained" or
            pretrain == "operaCT-heart-nonoisy"
        ):
            pretrained_model = initialize_pretrained_model("operaCT")
            encoder_path = get_encoder_path(f"{pretrain}-{dataset_name}")
            print("loading weights from", encoder_path)
            ckpt = torch.load(encoder_path)
            pretrained_model.load_state_dict(ckpt["state_dict"], strict=False)
        elif pretrain == "operaCT-heart-all" or "operaCT-heart-all-scratch":
            pretrained_model = initialize_pretrained_model("operaCT")
            encoder_path = get_encoder_path(pretrain)
            print("loading weights from", encoder_path)
            ckpt = torch.load(encoder_path)
            pretrained_model.load_state_dict(ckpt["state_dict"], strict=False)
        elif pretrain == "operaCT-heart":
            pretrained_model = initialize_pretrained_model("operaCT")
            encoder_path = get_encoder_path(f"{pretrain}-cross-{dataset_name}")
            print("loading weights from", encoder_path)
            ckpt = torch.load(encoder_path)
            pretrained_model.load_state_dict(ckpt["state_dict"], strict=False)
        elif pretrain == "null":
            pretrained_model = initialize_pretrained_model(pretrain)
            lr = 1e-4
            epochs = 64
            print("-" * 20 + "training from scratch")
        else:
            pretrained_model = initialize_pretrained_model(pretrain)
            encoder_path = get_encoder_path(pretrain)
            print("loading weights from", encoder_path)
            ckpt = torch.load(encoder_path)
            pretrained_model.load_state_dict(ckpt["state_dict"], strict=False)

        if "mae" in pretrain or "GT" in pretrain:
            model = AudioClassifierAudioMAE(
                net=pretrained_model,
                classes=n_cls,
                lr=lr,
                l2_strength=l2_strength,
                feat_dim=feat_dim,
                loss_func=loss_func,
                metrics=metrics,
                dataset=dataset_name,
                task=task
            )
        else:
            #freeze_encoder = "early" if pretrain == "operaCE" else "none"
            net = pretrained_model.encoder
            model = AudioClassifier(
                net=net,
                head=head,
                classes=n_cls,
                lr=lr,
                l2_strength=l2_strength,
                feat_dim=feat_dim,
                freeze_encoder=freeze_encoder,
                loss_func=loss_func,
                metrics=metrics,
                dataset=dataset_name,
                task=task
            )

    wandb_logger.experiment.config.update(
        {
            "n_cls": n_cls,
            "pretrain": pretrain,
            "l2_strength": l2_strength,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "head": head,
            "seed": seed,
            "dataset": dataset_name,
            "task": task,
            "freeze_encoder": freeze_encoder,
            "loss": loss,
            "spec_augment": spec_augment,
            "time_drop_width": time_drop_width,
            "freq_drop_width": freq_drop_width
        }
    )

    # Filter out NaN values
    x_data = x_data[valid_indices]

    x_data_train = x_data[y_set == "train"]
    y_label_train = y_label[y_set == "train"]
    x_data_vad = x_data[y_set == "val"]
    y_label_vad = y_label[y_set == "val"]
    x_data_test = x_data[y_set == "test"]
    y_label_test = y_label[y_set == "test"]

    print(f"Train set label distributions {collections.Counter(y_label_train)}")
    print(f"Val set label distributions {collections.Counter(y_label_vad)}")
    print(f"Test set label distributions {collections.Counter(y_label_test)}")

    train_data = AudioDataset(
        (x_data_train, y_label_train),
        augment=False,
        max_len=False,
        from_audio=from_audio,
        spec_augment=spec_augment,
        time_drop_width=time_drop_width,
        freq_drop_width=freq_drop_width
    )
    test_data = AudioDataset(
        (x_data_test, y_label_test), augment=False, max_len=False, from_audio=from_audio
    )
    val_data = AudioDataset(
        (x_data_vad, y_label_vad), augment=False, max_len=False, from_audio=from_audio
    )

    if dataset_name == "physionet16":
        annotations = np.load(feature_dir + "annotations.npy").astype(np.int32)
        annotations = annotations[valid_indices]
        annotations_train = annotations[y_set == "train"]
        annotations_vad = annotations[y_set == "val"]
        annotations_test = annotations[y_set == "test"]

        train_data = AudioDataset(
            (x_data_train, y_label_train, annotations_train),
            augment=False,
            max_len=False,
            from_audio=from_audio,
            spec_augment=spec_augment,
            time_drop_width=time_drop_width,
            freq_drop_width=freq_drop_width
        )
        test_data = AudioDataset(
            (x_data_test, y_label_test, annotations_test), augment=False, max_len=False, from_audio=from_audio
        )
        val_data = AudioDataset(
            (x_data_vad, y_label_vad, annotations_vad), augment=False, max_len=False, from_audio=from_audio
        )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=False
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=2
    )

    ck_path = (
        f"cks/finetune/{dataset_name}_{task}/"
        if task
        else f"cks/finetune/{dataset_name}"
    )

    ck_filename ="_".join(
        [
            "finetuning",
            head,
            pretrain,
            str(batch_size),
            str(lr),
            str(epochs),
            str(l2_strength),
            str(seed)
        ]
    )

    ck_filename = ck_filename + "_early" if freeze_encoder == "early" else ck_filename
    ck_filename = ck_filename + "_weighted" if loss == "weighted" else ck_filename

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath=ck_path,
        filename=ck_filename
        + "-{epoch:02d}-{valid_auc:.2f}",
        every_n_epochs=3,
    )

    early_stop_callback = EarlyStopping(
        monitor="valid_auc", min_delta=0.001, patience=10, verbose=True, mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[DecayLearningRate(), checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    if trainer.should_stop:
        print("Early stopping triggered. Training stopped.")

    #trainer.test(dataloaders=train_loader) # logging overwritten by test
    #trainer.test(dataloaders=val_loader) # logging overwritten by test
    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        f"finished training dataset {dataset_name} {task} from model pretrained on",
        pretrain,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    # Save weights to wandb
    # weights_path = os.path.join(ck_path, ck_filename + "-weights_only.pt")
    # torch.save(model.state_dict(), weights_path)

    # wandb_run = wandb_logger.experiment
    # artifact_name = f"{run_name}-weights"
    # artifact = wandb.Artifact(name=artifact_name, type="model")
    # artifact.add_file(weights_path, artifact_name + "_weights.pt")
    # wandb_run.log_artifact(artifact)
    wandb.finish()
    return auc


@hydra.main(config_path="../configs", config_name="finetune_config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if not cfg.LOOCV:
        auc_scores = []
        for seed in range(cfg.n_run):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            seed_everything(seed, workers=True)
            if cfg.task == "covid19sounds":
                auc = finetune_covid19sounds(
                    task=1,
                    pretrain=cfg.pretrain,
                    modality=cfg.modality,
                    epochs=64,
                    l2_strength=1e-4,
                    feat_dim=cfg.dim,
                )
            elif cfg.task == "covid19soundsdownsample":
                auc = finetune_covid19sounds(
                    task="1downsample",
                    pretrain=cfg.pretrain,
                    modality=cfg.modality,
                    epochs=64,
                    l2_strength=1e-4,
                    feat_dim=cfg.dim,
                )
            elif cfg.task == "snoring":
                auc = finetune_ssbpr(
                    pretrain=cfg.pretrain, epochs=64, feat_dim=cfg.dim
                )
            elif cfg.task == "icbhidisease":
                auc = finetune_icbhidisease(
                    pretrain=cfg.pretrain,
                    epochs=64,
                    l2_strength=1e-4,
                    feat_dim=cfg.dim,
                )
            elif (
                cfg.task == "circor_murmurs"
                or cfg.task == "circor_outcomes"
                or cfg.task == "circor_systolic-murmur-grading"
                or cfg.task == "circor_systolic-murmur-grading-w-absent"
                or cfg.task == "circor_systolic-murmur-pitch"
                or cfg.task == "circor_systolic-murmur-quality"
                or cfg.task == "circor_systolic-murmur-shape"
                or cfg.task == "circor_systolic-murmur-timing"
            ):
                task = cfg.task.split("_")[1]
                auc = finetune_heart(
                    pretrain=cfg.pretrain,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    feat_dim=cfg.dim,
                    dataset_name="circor",
                    task=task,
                    feature_dir="feature/circor_eval/",
                    labels_filename=f"{task}.npy",
                    seed=seed,
                    freeze_encoder=cfg.freeze_encoder,
                    loss=cfg.loss,
                    spec_augment=cfg.spec_augment
                )
            elif cfg.task == "zchsound_clean" or cfg.task == "zchsound_noisy": # ZCHSound outcomes
                task = cfg.task.split("_")[1]
                auc = finetune_heart(
                    pretrain=cfg.pretrain,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    feat_dim=cfg.dim,
                    dataset_name="zchsound",
                    task=task,
                    feature_dir=f"feature/{cfg.task}_eval/",
                    labels_filename="outcomes.npy",
                    seed=seed,
                    freeze_encoder=cfg.freeze_encoder,
                    loss=cfg.loss,
                    spec_augment=cfg.spec_augment
                )
            elif cfg.task == "zchsound_clean_murmurs" or cfg.task == "zchsound_noisy_murmurs": # ZCHSound murmurs
                data_task_list = cfg.task.split("_")
                dataset_name = f"{data_task_list[0]}_{data_task_list[1]}"
                task = data_task_list[2]
                auc = finetune_heart(
                    pretrain=cfg.pretrain,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    feat_dim=cfg.dim,
                    dataset_name=dataset_name,
                    task=task,
                    feature_dir=f"feature/{dataset_name}_eval/",
                    labels_filename=f"{task}.npy",
                    seed=seed,
                    freeze_encoder=cfg.freeze_encoder,
                    loss=cfg.loss,
                    spec_augment=cfg.spec_augment
                )
            elif cfg.task == "pascal_A" or cfg.task == "pascal_B":
                task = cfg.task.split("_")[1]
                auc = finetune_heart(
                    pretrain=cfg.pretrain,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    feat_dim=cfg.dim,
                    dataset_name="pascal",
                    task=task,
                    feature_dir=f"feature/{cfg.task}_eval/",
                    labels_filename="labels.npy",
                    seed=seed,
                    freeze_encoder=cfg.freeze_encoder,
                    loss=cfg.loss,
                    spec_augment=cfg.spec_augment
                )
            elif cfg.task == "physionet16":
                auc = finetune_heart(
                    pretrain=cfg.pretrain,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    feat_dim=cfg.dim,
                    dataset_name="physionet16",
                    task="",
                    feature_dir=f"feature/{cfg.task}_eval/",
                    labels_filename="labels.npy",
                    seed=seed,
                    freeze_encoder=cfg.freeze_encoder,
                    loss=cfg.loss,
                    spec_augment=cfg.spec_augment
                )
            auc_scores.append(auc)
        print("=" * 48)
        print(auc_scores)
        print(
            f"Five times mean task {cfg.task} finetuning from {cfg.pretrain} results: auc mean {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}"
        )
        print("=" * 48)

if __name__ == '__main__':
    main()