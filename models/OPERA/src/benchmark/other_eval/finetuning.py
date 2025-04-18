import collections
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
from lightning.pytorch import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import ViTConfig, ViTModel
import wandb

from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model
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
    ):
        self.data = data[0]
        self.label = data[1]
        self.max_len = max_len
        self.augment = augment
        self.from_npy = from_npy
        self.crop_mode = crop_mode
        self.from_audio = from_audio

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
            f"Task not implemented: Covid-19 sounds task {task}, please check the args."
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
    feat_dim=1280,
    dataset_name="circor",
    task="murmurs",
    feature_dir="feature/circor_eval/",
    labels_filename="murmurs.npy",
    freeze_encoder="none",  # Control freezing
):
    n_cls = len(set(np.load(feature_dir + labels_filename)))

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
        batch_size = 32

        encoder_path = "src/benchmark/baseline/audioMAE/pretrained.pth"
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
            metrics=metrics,
            dataset=dataset_name,
            task=task
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
            metrics=metrics,
            dataset=dataset_name,
            task=task
        )
        from_audio = True
    elif pretrain == "clap2023":
        from src.benchmark.baseline.msclap import CLAP

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
            task=task
        )
        from_audio = True
    elif pretrain == "hear":
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
        feat_dim=1024
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
            metrics=metrics,
            dataset=dataset_name,
            task=task
        )
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
                metrics=metrics,
                dataset=dataset_name,
                task=task
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
            "freeze_encoder": freeze_encoder
        }
    )

    y_set = np.load(feature_dir + "train_test_split.npy")
    y_label = np.load(feature_dir + labels_filename)
    print(collections.Counter(y_label))
    print(collections.Counter(y_set))

    x_data_train = x_data[y_set == "train"]
    y_label_train = y_label[y_set == "train"]
    x_data_vad = x_data[y_set == "val"]
    y_label_vad = y_label[y_set == "val"]
    x_data_test = x_data[y_set == "test"]
    y_label_test = y_label[y_set == "test"]

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="icbhidisease")
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--gridsearch", type=bool, default=False)
    parser.add_argument(
        "--lr", type=float, default=1e-4
    )  # not used if gridsearch = True
    parser.add_argument(
        "--l2_strength", type=float, default=1e-5
    )  # not used if gridsearch = True
    parser.add_argument("--head", type=str, default="linear")
    parser.add_argument("--modality", type=str, default="cough")
    parser.add_argument("--mapgoogle", type=bool, default=False)  # align test set
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--n_run", type=int, default=5)
    parser.add_argument("--label", type=str, default="smoker")  # align test set
    parser.add_argument("--LOOCV", type=bool, default=False)
    parser.add_argument("--avgprob", type=bool, default=False)
    args = parser.parse_args()

    if not args.LOOCV:
        auc_scores = []
        for seed in range(args.n_run):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            seed_everything(seed, workers=True)
            if args.task == "covid19sounds":
                auc = finetune_covid19sounds(
                    task=1,
                    pretrain=args.pretrain,
                    modality=args.modality,
                    epochs=64,
                    l2_strength=1e-4,
                    feat_dim=args.dim,
                )
            elif args.task == "covid19soundsdownsample":
                auc = finetune_covid19sounds(
                    task="1downsample",
                    pretrain=args.pretrain,
                    modality=args.modality,
                    epochs=64,
                    l2_strength=1e-4,
                    feat_dim=args.dim,
                )
            elif args.task == "snoring":
                auc = finetune_ssbpr(
                    pretrain=args.pretrain, epochs=64, feat_dim=args.dim
                )
            elif args.task == "icbhidisease":
                auc = finetune_icbhidisease(
                    pretrain=args.pretrain,
                    epochs=64,
                    l2_strength=1e-4,
                    feat_dim=args.dim,
                )
            elif args.task == "circor_murmurs" or args.task == "circor_outcomes":
                task = args.task.split("_")[1]
                auc = finetune_heart(
                    pretrain=args.pretrain,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    feat_dim=args.dim,
                    dataset_name="circor",
                    task=task,
                    feature_dir="feature/circor_eval/",
                    labels_filename=f"{task}.npy",
                    seed=seed,
                )
            elif args.task == "zchsound_clean" or args.task == "zchsound_noisy":
                task = args.task.split("_")[1]
                auc = finetune_heart(
                    pretrain=args.pretrain,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    feat_dim=args.dim,
                    dataset_name="zchsound",
                    task=task,
                    feature_dir=f"feature/{args.task}_eval/",
                    labels_filename="labels.npy",
                    seed=seed,
                )
            elif args.task == "pascal_A" or args.task == "pascal_B":
                task = args.task.split("_")[1]
                auc = finetune_heart(
                    pretrain=args.pretrain,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    feat_dim=args.dim,
                    dataset_name="pascal",
                    task=task,
                    feature_dir=f"feature/{args.task}_eval/",
                    labels_filename="labels.npy",
                    seed=seed,
                )
            elif args.task == "physionet16":
                auc = finetune_heart(
                    pretrain=args.pretrain,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    feat_dim=args.dim,
                    dataset_name="physionet16",
                    task="",
                    feature_dir=f"feature/{args.task}_eval/",
                    labels_filename="labels.npy",
                    seed=seed,
                )
            auc_scores.append(auc)
        print("=" * 48)
        print(auc_scores)
        print(
            f"Five times mean task {args.task} finetuning from {args.pretrain} results: auc mean {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}"
        )
        print("=" * 48)
