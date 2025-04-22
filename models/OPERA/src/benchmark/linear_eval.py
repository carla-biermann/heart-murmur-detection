import collections
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.models_eval import LinearHead, LinearHeadR
from src.util import downsample_balanced_dataset, train_test_split_from_list, get_weights_tensor
import hydra
from omegaconf import DictConfig, OmegaConf

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data[0]
        self.label = data[1]
        self.annotations = data[2] if len(data) > 2 else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        label = self.label[idx]
        x = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        if self.annotations is not None:
            annotation = self.annotations[idx]
            annotation = torch.tensor(annotation, dtype=torch.long)
            return x, label, annotation
        else:
            return x, label


class FeatureDatasetR(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data[0]
        self.label = data[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        label = self.label[idx]
        x = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        return x, label


class DecayLearningRate(pl.Callback):
    def __init__(self, weight=0.97):
        self.old_lrs = []
        self.weight = weight

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
                new_lr = old_lr * self.weight
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group


def get_weights_tensor(labels, n_cls):
    label_counts = collections.Counter(labels)
    total_count = len(labels)
    class_freqs = np.array([label_counts[i] / total_count for i in range(n_cls)])

    # Inverse frequency and normalize
    class_weights = 1.0 / class_freqs
    class_weights = class_weights / class_weights.sum()

    return torch.tensor(class_weights, dtype=torch.float)


def linear_evaluation_covid19sounds(
    task=1,
    use_feature="opensmile",
    modality="cough",
    l2_strength=1e-4,
    lr=1e-5,
    head="linear",
    batch_size=64,
    epochs=64,
):
    print(
        f"linear evaluation ({head}) of task",
        task,
        "with feature set",
        use_feature,
        "modality",
        modality,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "*" * 28,
    )
    folders = {1: "feature/covid19sounds_eval/downsampled/"}
    folder = folders[task]

    x_data = np.load(folder + use_feature + f"_feature_{modality}.npy").squeeze()
    y_label = np.load(folder + "labels.npy")
    y_set = np.load(folder + "data_split.npy")

    feat_dim = x_data.shape[1]

    if task == 1:
        x_data_train = x_data[y_set == 0]
        y_label_train = y_label[y_set == 0]
        x_data_vad = x_data[y_set == 1]
        y_label_vad = y_label[y_set == 1]
        x_data_test = x_data[y_set == 2]
        y_label_test = y_label[y_set == 2]
    else:
        raise NotImplementedError(
            f"Task not implemented: Covid-19 sounds task {task}, please check the config."
        )

    print(collections.Counter(y_label_train))
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))

    train_data = FeatureDataset((x_data_train, y_label_train))
    test_data = FeatureDataset((x_data_test, y_label_test))
    val_data = FeatureDataset((x_data_vad, y_label_vad))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(feat_dim=feat_dim, classes=2, l2_strength=l2_strength, head=head)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/covidtask" + str(task) + "/" + modality + "/",
        filename="_".join(
            [
                "linear",
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    logger = CSVLogger(
        save_dir="cks/logs",
        name="covid-task" + str(task) + modality,
        version="_".join(
            [
                "linear",
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        ),
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

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        f"finished training linear evaluation ({head}) of task",
        task,
        "with feature set",
        use_feature,
        "modality",
        modality,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "*" * 28,
    )
    return auc


def linear_evaluation_icbhidisease(
    use_feature="opensmile",
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

    print(collections.Counter(y_label))
    x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()

    mask = (y_label == "Healthy") | (y_label == "COPD")
    y_label = y_label[mask]
    y_set = y_set[mask]
    x_data = x_data[mask]

    label_dict = {"Healthy": 0, "COPD": 1}
    y_label = np.array([label_dict[y] for y in y_label])

    if use_feature == "vggish":
        x_data = np.nan_to_num(x_data)
    feat_dim = x_data.shape[1]

    X_train, X_test, y_train, y_test = train_test_split_from_list(
        x_data, y_label, y_set
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1337, stratify=y_train
    )
    print(collections.Counter(y_train))
    print(collections.Counter(y_val))
    print(collections.Counter(y_test))

    train_data = FeatureDataset((X_train, y_train))
    test_data = FeatureDataset((X_test, y_test))
    val_data = FeatureDataset((X_val, y_val))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )
    loss_func = None

    model = LinearHead(
        feat_dim=feat_dim,
        classes=2,
        l2_strength=l2_strength,
        loss_func=loss_func,
        head=head,
    )

    logger = CSVLogger(
        save_dir="cks/logs",
        name="icbhidisease",
        version="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/icbhidisease/",
        filename="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
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
        "finished training dataset icbhi disease using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
    )
    return auc


def linear_evaluation_kauh(
    use_feature="opensmile",
    l2_strength=1e-6,
    epochs=64,
    lr=1e-5,
    batch_size=64,
    head="linear",
):
    print("*" * 48)
    print(
        "training dataset kauh using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )
    folder = "feature/kauh_eval/"

    labels = np.load(folder + "labels_both.npy")
    y_set = np.load(folder + "train_test_split.npy")

    x_data = np.load(folder + use_feature + "_feature_both.npy").squeeze()

    label_dict = {"healthy": 0, "asthma": 1, "COPD": 1, "obstructive": 1}
    y_label = np.array([label_dict[y] for y in labels])

    feat_dim = x_data.shape[1]

    X_train, X_test, y_train, y_test = train_test_split_from_list(
        x_data, y_label, y_set
    )
    print("training distribution", collections.Counter(y_train))
    print("testing distribution", collections.Counter(y_test))

    # generate a validation set (seed fixed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=1337, stratify=y_train
    )

    train_data = FeatureDataset((X_train, y_train))
    test_data = FeatureDataset((X_test, y_test))
    val_data = FeatureDataset((X_val, y_val))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=1, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=1, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=1
    )

    model = LinearHead(feat_dim=feat_dim, classes=2, l2_strength=l2_strength, head=head)

    logger = CSVLogger(
        save_dir="cks/logs",
        name="kauh",
        version="_".join(
            [
                "linear",
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/kauh/",
        filename="_".join(
            [
                "linear",
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
        every_n_epochs=5,
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

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset kauh classes using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
    )
    return auc


def linear_evaluation_coswara(
    use_feature="operaCE1280",
    l2_strength=1e-6,
    epochs=256,
    lr=1e-5,
    batch_size=32,
    modality="breathing-deep",
    label="smoker",
    head="linear",
    map_google=False,
):
    print("*" * 48)
    print(
        f"training dataset coswara of task {label} and modality {modality} using feature extracted by {use_feature} with l2_strength {l2_strength} lr {lr} head {head}"
    )

    feature_dir = "feature/coswara_eval/"

    broad_modality = modality.split("-")[0]
    labels = np.load(
        feature_dir + f"{broad_modality}_aligned_{label}_label_{modality}.npy"
    )
    features = np.load(
        feature_dir + use_feature + f"_feature_{modality}_{label}.npy"
    ).squeeze()
    print(collections.Counter(labels))

    feat_dim = features.shape[1]

    if map_google:
        if "cough" not in modality:
            raise NotImplementedError("not supported")
        split = np.load(feature_dir + f"google_{label}_{modality}_split.npy")
        X_train, X_test, y_train, y_test = train_test_split_from_list(
            features, labels, split
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=1337, stratify=labels
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1337, stratify=y_train
    )

    if use_feature == "vggish":
        # for vggish, this produces a better and more reasonable result, otherwise an extremely low AUROC
        X_train, y_train = downsample_balanced_dataset(X_train, y_train)

    print(collections.Counter(y_train))
    print(collections.Counter(y_val))
    print(collections.Counter(y_test))

    train_data = FeatureDataset((X_train, y_train))
    test_data = FeatureDataset((X_test, y_test))
    val_data = FeatureDataset((X_val, y_val))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(feat_dim=feat_dim, classes=2, l2_strength=l2_strength, head=head)

    logger = CSVLogger(
        save_dir="cks/logs",
        name="coswara",
        version="_".join(
            [
                head,
                label,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/coswara/",
        filename="_".join(
            [
                head,
                label,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
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

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset coswara using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "with an auc of",
        auc,
    )
    return auc


def linear_evaluation_copd(
    n_cls=5,
    use_feature="opensmile",
    l2_strength=1e-5,
    epochs=64,
    batch_size=32,
    lr=1e-4,
    head="linear",
):
    print("*" * 48)
    print(
        "training dataset RespiratoryDatabase@TR using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    feature_dir = "feature/copd_eval/"

    y_set = np.load(feature_dir + "train_test_split.npy")
    y_label = np.load(feature_dir + "labels.npy")
    print(collections.Counter(y_label))
    x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()

    feat_dim = x_data.shape[1]
    print(feat_dim)

    x_data_train = x_data[y_set == "train"]
    y_label_train = y_label[y_set == "train"]
    x_data_vad = x_data[y_set == "val"]
    y_label_vad = y_label[y_set == "val"]
    x_data_test = x_data[y_set == "test"]
    y_label_test = y_label[y_set == "test"]

    print(collections.Counter(y_label_train))
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))

    train_data = FeatureDataset((x_data_train, y_label_train))
    test_data = FeatureDataset((x_data_test, y_label_test))
    val_data = FeatureDataset((x_data_vad, y_label_vad))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=1, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=1, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=1
    )

    model = LinearHead(
        feat_dim=feat_dim, classes=n_cls, l2_strength=l2_strength, head=head
    )

    logger = CSVLogger(
        save_dir="cks/logs",
        name="copd",
        version="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/copd/",
        filename="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
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

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset RespiratoryDatabase@TR using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )
    return auc


def linear_evaluation_coughvid(
    use_feature="operaCE1280",
    l2_strength=1e-6,
    epochs=64,
    lr=1e-5,
    batch_size=64,
    label="sex",
    head="linear",
):
    print("*" * 48)
    print(
        f"training dataset coughvid of task {label} and using feature extracted by {use_feature} with l2_strength {l2_strength} lr {lr}  head"
    )

    feature_dir = "feature/coughvid_eval/"

    y_set = np.load(feature_dir + f"split_{label}.npy")
    y_label = np.load(feature_dir + f"label_{label}.npy")
    print(collections.Counter(y_label))
    x_data = np.load(feature_dir + use_feature + f"_feature_{label}.npy").squeeze()

    if use_feature == "vggish":
        x_data = np.nan_to_num(x_data)

    feat_dim = x_data.shape[1]

    x_data_train = x_data[y_set == "train"]
    y_label_train = y_label[y_set == "train"]
    x_data_vad = x_data[y_set == "val"]
    y_label_vad = y_label[y_set == "val"]
    x_data_test = x_data[y_set == "test"]
    y_label_test = y_label[y_set == "test"]

    train_data = FeatureDataset((x_data_train, y_label_train))
    test_data = FeatureDataset((x_data_test, y_label_test))
    val_data = FeatureDataset((x_data_vad, y_label_vad))

    print(collections.Counter(y_label_train))
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(feat_dim=feat_dim, classes=2, l2_strength=l2_strength)

    logger = CSVLogger(
        save_dir="cks/logs",
        name="coughvid",
        version="_".join(
            [
                head,
                label,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/coughvid/",
        filename="_".join(
            [
                head,
                label,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
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

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        f"finished training dataset coughvid of task {label} and using feature extracted by {use_feature} with l2_strength {l2_strength} lr {lr} "
    )
    return auc


def linear_evaluation_coviduk(
    use_feature="operaCE1280",
    l2_strength=1e-6,
    epochs=64,
    lr=1e-5,
    batch_size=64,
    modality="exhalation",
    head="linear",
):
    print("*" * 48)
    print(
        f"training dataset covidUK of {modality} and using feature extracted by {use_feature} with l2_strength {l2_strength} lr {lr}  head"
    )

    feature_dir = "feature/coviduk_eval/"

    y_set = np.load(feature_dir + f"split_{modality}.npy")
    y_label = np.load(feature_dir + f"label_{modality}.npy")
    print(collections.Counter(y_label))
    x_data = np.load(feature_dir + use_feature + f"_feature_{modality}.npy").squeeze()

    if use_feature == "vggish":
        x_data = np.nan_to_num(x_data)

    feat_dim = x_data.shape[1]

    x_data_train = x_data[y_set == "train"]
    y_label_train = y_label[y_set == "train"]
    x_data_vad = x_data[y_set == "val"]
    y_label_vad = y_label[y_set == "val"]
    x_data_test = x_data[y_set == "test"]
    y_label_test = y_label[y_set == "test"]

    train_data = FeatureDataset((x_data_train, y_label_train))
    test_data = FeatureDataset((x_data_test, y_label_test))
    val_data = FeatureDataset((x_data_vad, y_label_vad))

    print(collections.Counter(y_label_train))
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(feat_dim=feat_dim, classes=2, l2_strength=l2_strength)

    logger = CSVLogger(
        save_dir="cks/logs",
        name="coviduk",
        version="_".join(
            [
                head,
                modality,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/coviduk/",
        filename="_".join(
            [
                head,
                modality,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        f"finished training dataset covidUK of {modality} and using feature extracted by {use_feature} with l2_strength {l2_strength} lr {lr}  head"
    )
    return auc


def linear_evaluation_ssbpr(
    n_cls=5,
    use_feature="opensmile",
    l2_strength=1e-5,
    epochs=64,
    batch_size=64,
    lr=1e-4,
    head="linear",
    seed=None,
    five_fold=False,
    split_fold=False,
):
    print("*" * 48)
    print(
        "training dataset SSBPR using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    feature_dir = "feature/snoring_eval/"

    y_label = np.load(feature_dir + "labels.npy")
    print(collections.Counter(y_label))
    x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()

    if use_feature == "vggish":
        x_data = np.nan_to_num(x_data)

    feat_dim = x_data.shape[1]
    print(feat_dim)

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

    train_data = FeatureDataset((x_data_train, y_label_train))
    test_data = FeatureDataset((x_data_test, y_label_test))
    val_data = FeatureDataset((x_data_vad, y_label_vad))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(
        feat_dim=feat_dim, classes=n_cls, l2_strength=l2_strength, head=head
    )

    logger = CSVLogger(
        save_dir="cks/logs",
        name="ssbpr",
        version="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/ssbpr/",
        filename="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
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
        "finished training dataset SSBPR using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )
    return auc


def linear_evaluation_mmlung(
    use_feature="opensmile",
    method="LOOCV",
    l2_strength=1e-5,
    epochs=64,
    lr=1e-5,
    batch_size=40,
    modality="cough",
    label="FVC",
    head="mlp",
):
    from sklearn.preprocessing import StandardScaler

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

    feature_dir = "feature/mmlung_eval/"

    y_label = np.load(feature_dir + "label.npy")
    if label == "FVC":
        y_label = y_label[:, 0]
    if label == "FEV1":
        y_label = y_label[:, 1]
    if label == "FEV1_FVC":
        y_label = y_label[:, 2]

    if modality == "breath":
        x_data = np.load(
            feature_dir + "Deep_Breath_file_" + use_feature + "_feature.npy"
        ).squeeze()

    if modality == "vowels":
        x_data = np.load(
            feature_dir + "O_Single_file_" + use_feature + "_feature.npy"
        ).squeeze()

    if use_feature == "vggish":
        x_data = np.nan_to_num(x_data)

    print(label, "distribution:", np.mean(y_label), np.std(y_label))
    y_label = y_label.reshape((-1, 1))

    # # leave one out cross validation
    MAEs = []
    MAPEs = []
    for s in tqdm(range(40)):
        x_test, y_test = x_data[s], y_label[s]
        x_test = x_test.reshape(1, -1)
        y_test = y_test.reshape(1, -1)

        X_train = np.delete(x_data, s, axis=0)
        y_train = np.delete(y_label, s, axis=0)

        if "opensmile" in use_feature:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            x_test = scaler.transform(x_test)

        x_train, x_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.11, random_state=42
        )

        train_mean, train_std = np.mean(y_train), np.std(y_train)
        print("mean and std of training data:", train_mean, train_std)

        train_data = FeatureDatasetR((x_train, y_train))
        test_data = FeatureDatasetR((x_test, y_test))
        val_data = FeatureDatasetR((x_val, y_val))

        train_loader = DataLoader(
            train_data, batch_size=batch_size, num_workers=8, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, num_workers=4, shuffle=False
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=4
        )

        feat_dim = x_data.shape[1]
        model = LinearHeadR(
            feat_dim=feat_dim,
            output_dim=1,
            l2_strength=l2_strength,
            head=head,
            mean=train_mean,
            std=train_std,
        )

        logger = CSVLogger(
            save_dir="cks/logs",
            name="mmlung",
            version="_".join(
                [
                    head,
                    use_feature,
                    str(batch_size),
                    str(lr),
                    str(epochs),
                    str(l2_strength),
                ]
            ),
        )

        early_stop_callback = EarlyStopping(
            monitor="valid_MAE", min_delta=0.001, patience=5, verbose=True, mode="min"
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="valid_MAE",
            mode="min",
            dirpath="cks/linear/mmlung/",
            filename="_".join(
                [
                    head,
                    use_feature,
                    str(batch_size),
                    str(lr),
                    str(epochs),
                    str(l2_strength),
                ]
            )
            + "-{epoch:02d}-{valid_MAE:.3f}",
            every_n_epochs=3,
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu",
            devices=1,
            # logger=logger,
            logger=False,
            callbacks=[
                DecayLearningRate(weight=0.97),
                checkpoint_callback,
                early_stop_callback,
            ],
            gradient_clip_val=1.0,
            log_every_n_steps=1,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_loader, val_loader)

        test_res = trainer.test(dataloaders=test_loader)
        mae, mape = test_res[0]["test_MAE"], test_res[0]["test_MAPE"]

        MAEs.append(mae)
        MAPEs.append(mape)

    return MAEs, MAPEs


def linear_evaluation_nosemic(
    use_feature="opensmile",
    method="LOOCV",
    l2_strength=1e-5,
    epochs=64,
    batch_size=32,
    lr=1e-4,
    head="mlp",
):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("*" * 48)
    print(
        "training dataset Nose Breathing audio using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    feature_dir = "feature/nosemic_eval/"

    uids = np.load(feature_dir + "uids.npy")
    y_label = np.load(feature_dir + "labels.npy")
    y_label = np.array([float(y) for y in y_label]).reshape(-1, 1)
    print("labels:", y_label)
    x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()

    feat_dim = x_data.shape[1]
    print(feat_dim)
    print(uids.shape, x_data.shape)

    MAEs = []
    MAPEs = []
    for uid in [
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "17",
        "18",
        "19",
        "20",
        "21",
    ]:
        x_train = x_data[uids != uid, :]
        x_test = x_data[uids == uid, :]
        y_train = y_label[uids != uid]
        y_test = y_label[uids == uid]

        if "opensmile" in use_feature:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        train_mean, train_std = np.mean(y_train), np.std(y_train)
        print("mean and std of training data:", train_mean, train_std)

        train_data = FeatureDatasetR((x_train, y_train))
        test_data = FeatureDatasetR((x_test, y_test))
        val_data = FeatureDatasetR((x_val, y_val))

        train_loader = DataLoader(
            train_data, batch_size=batch_size, num_workers=8, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, num_workers=4, shuffle=False
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=4
        )

        model = LinearHeadR(
            feat_dim=feat_dim,
            output_dim=1,
            l2_strength=l2_strength,
            head=head,
            mean=train_mean,
            std=train_std,
        )

        logger = CSVLogger(
            save_dir="cks/logs",
            name="nosemic",
            version="_".join(
                [
                    head,
                    use_feature,
                    str(batch_size),
                    str(lr),
                    str(epochs),
                    str(l2_strength),
                ]
            ),
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="valid_MAE",
            mode="min",
            dirpath="cks/linear/nosemic/",
            filename="_".join(
                [
                    head,
                    use_feature,
                    str(batch_size),
                    str(lr),
                    str(epochs),
                    str(l2_strength),
                ]
            )
            + "-{epoch:02d}-{valid_MAE:.3f}",
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu",
            devices=1,
            # logger=logger,
            logger=False,
            callbacks=[DecayLearningRate(weight=0.97), checkpoint_callback],
            gradient_clip_val=1.0,
            log_every_n_steps=1,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_loader, val_loader)

        test_res = trainer.test(dataloaders=test_loader)
        mae, mape = test_res[0]["test_MAE"], test_res[0]["test_MAPE"]

        MAEs.append(mae)
        MAPEs.append(mape)

    return MAEs, MAPEs


def get_wandb_name(use_feature, data, head):
    s = time.gmtime(time.time())
    return f"{time.strftime('%Y-%m-%d %H:%M:%S', s)}-{use_feature}-{data}-{head}"


def linear_evaluation_heart(
    seed,
    use_feature="operaCE1280",
    l2_strength=1e-5,
    epochs=64,
    batch_size=32,
    lr=1e-4,
    head="linear",
    loss="unweighted",
    dataset_name="circor",
    task="murmurs",
    feature_dir="feature/circor_eval/",
    labels_filename="murmurs.npy",
):
    print("*" * 48)
    print(
        f"training dataset {dataset_name} {task} using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    y_set = np.load(feature_dir + "train_test_split.npy")
    y_label = np.load(feature_dir + labels_filename)
    print(f"Label distribution: {collections.Counter(y_label)}")
    x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()

    feat_dim = x_data.shape[1]
    print(f"Feat_dim: {feat_dim}")

    n_cls = len(set(y_label))
    print(f"Number of classes: {n_cls}")

    x_data_train = x_data[y_set == "train"]
    y_label_train = y_label[y_set == "train"]
    x_data_vad = x_data[y_set == "val"]
    y_label_vad = y_label[y_set == "val"]
    x_data_test = x_data[y_set == "test"]
    y_label_test = y_label[y_set == "test"]

    print(f"Train set label distributions {collections.Counter(y_label_train)}")
    print(f"Val set label distributions {collections.Counter(y_label_vad)}")
    print(f"Test set label distributions {collections.Counter(y_label_test)}")

    train_data = FeatureDataset((x_data_train, y_label_train))
    test_data = FeatureDataset((x_data_test, y_label_test))
    val_data = FeatureDataset((x_data_vad, y_label_vad))

    if dataset_name == "physionet16":
        annotations = np.load(feature_dir + "annotations.npy").astype(np.int32)
        annotations = annotations[valid_indices]
        annotations_train = annotations[y_set == "train"]
        annotations_vad = annotations[y_set == "val"]
        annotations_test = annotations[y_set == "test"]

        train_data = FeatureDataset((x_data_train, y_label_train, annotations_train))
        test_data = FeatureDataset((x_data_test, y_label_test, annotations_test))
        val_data = FeatureDataset((x_data_vad, y_label_vad, annotations_vad))


    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=1, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=1, shuffle=False
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=1
    )

    args = dict(
        feat_dim=feat_dim,
        classes=n_cls,
        l2_strength=l2_strength,
        head=head,
        metrics=[
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
            "unweighted_accuracy",
            "circor_weighted_outcome_acc",
            "circor_outcome_cost",
            "macro_F1",
            "macro_auroc",
            "physionet16_score",
        ],
        dataset=dataset_name,
        task=task,
    )

    if loss == "weighted":
        weights_tensor = get_weights_tensor(y_label_train, n_cls)
        loss_func = nn.CrossEntropyLoss(weight=weights_tensor)
        args["loss_func"] = loss_func

    model = LinearHead(**args)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath=f"cks/linear/{dataset_name}_{task}/",
        filename="_".join(
            [
                head,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
                str(seed),
            ]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    wandb_logger = WandbLogger(
        project="Heart-Sound-Analysis",
        name=get_wandb_name(use_feature, f"{dataset_name}-{task}", head),
        log_model=True,
    )

    wandb_logger.experiment.config.update(
        {
            "n_cls": n_cls,
            "use_feature": use_feature,
            "l2_strength": l2_strength,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "head": head,
            "dataset": dataset_name,
            "task": task,
            "seed": seed,
            "gradient_clip_val": 1.0,
            "loss": loss,
        }
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    wandb_logger.experiment.log({"test_auc": auc})
    print(
        f"finished training dataset {dataset_name} {task} using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )
    wandb.finish()
    return auc


def linear_evaluation_heart_cv(
    seed,
    use_feature="operaCE1280",
    l2_strength=1e-5,
    epochs=64,
    batch_size=32,
    lr=1e-4,
    head="linear",
    loss="unweighted",
    dataset_name="circor",
    task="murmurs",
    feature_dir="feature/circor_eval/",
    labels_filename="murmurs.npy",
    n_splits=5,
):
    print("*" * 48)
    print(
        f"Cross-validation on dataset {dataset_name} {task} using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    # Load data
    y_set = np.load(feature_dir + "train_test_split.npy")
    y_label = np.load(feature_dir + labels_filename)
    x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()

    x_data_train = x_data[y_set == "train"]
    y_label_train = y_label[y_set == "train"]

    print(f"Train set label distributions {collections.Counter(y_label_train)}")

    feat_dim = x_data_train.shape[1]
    n_cls = len(set(y_label_train))

    all_scores = []

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_data_train, y_label_train)):
        print(f"\nFold {fold + 1}/{n_splits}")
        x_fold_train, y_fold_train = x_data_train[train_idx], y_label_train[train_idx]
        x_fold_val, y_fold_val = x_data_train[val_idx], y_label_train[val_idx]

        train_data = FeatureDataset((x_fold_train, y_fold_train))
        val_data = FeatureDataset((x_fold_val, y_fold_val))

        train_loader = DataLoader(
            train_data, batch_size=batch_size, num_workers=1, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, num_workers=1, shuffle=False
        )

        args = dict(
            feat_dim=feat_dim,
            classes=n_cls,
            l2_strength=l2_strength,
            head=head,
            metrics=[
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
                "unweighted_accuracy",
                "circor_weighted_outcome_acc",
                "circor_outcome_cost",
            ],
            dataset=dataset_name,
            task=task,
        )

        if loss == "weighted":
            weights_tensor = get_weights_tensor(y_fold_train, n_cls)
            loss_func = nn.CrossEntropyLoss(weight=weights_tensor)
            args["loss_func"] = loss_func

        model = LinearHead(**args)

        checkpoint_callback = ModelCheckpoint(
            monitor="valid_auc",
            mode="max",
            dirpath=f"cks/linear_cv/{dataset_name}_{task}/fold_{fold}/",
            filename="_".join(
                [
                    head,
                    use_feature,
                    str(batch_size),
                    str(lr),
                    str(epochs),
                    str(l2_strength),
                    str(seed),
                ]
            )
            + "-{epoch:02d}-{valid_auc:.2f}",
        )

        wandb_logger = WandbLogger(
            project="Heart-Sound-Analysis-CV",
            name=f"{use_feature}-{dataset_name}-{task}-fold{fold}",
            log_model=False,
        )

        wandb_logger.experiment.config.update(
            {
                "n_cls": n_cls,
                "use_feature": use_feature,
                "l2_strength": l2_strength,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "head": head,
                "dataset": dataset_name,
                "task": task,
                "seed": seed,
                "gradient_clip_val": 1.0,
                "loss": loss,
                "fold": fold,
            }
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu",
            devices=1,
            logger=wandb_logger,
            callbacks=[DecayLearningRate(), checkpoint_callback],
            gradient_clip_val=1.0,
            log_every_n_steps=1,
            enable_progress_bar=False,
        )

        trainer.fit(model, train_loader, val_loader)
        val_res = trainer.test(dataloaders=val_loader)
        fold_auc = val_res[0]["test_auc"]
        if fold_auc is not None:
            all_scores.append(fold_auc)
            wandb_logger.experiment.log({"valid_auc": fold_auc})
        wandb.finish()

    print(f"\nCross-validation AUC scores: {all_scores}")
    print(f"Mean AUC: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
    return all_scores


@hydra.main(config_path="configs", config_name="linear_eval_config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    feature = cfg.pretrain
    if (
        feature not in ["vggish", "opensmile", "clap", "audiomae", "hear", "clap2023"]
        and "finetuned" not in feature
    ):  # baselines
        feature += str(cfg.dim)

    if cfg.grid_search:
        # Perform grid search over specified hyperparameters
        best_auc = -1
        best_params = None
        for l2_strength in cfg.l2_strength_grid:
            for lr in cfg.lr_grid:
                print(f"Testing with l2_strength={l2_strength}, lr={lr}")
                auc_scores = []
                for seed in range(cfg.n_run):
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    if (
                        cfg.task == "zchsound_clean" or cfg.task == "zchsound_noisy"
                    ):  # ZCHSound outcomes
                        data_task_list = cfg.task.split("_")
                        dataset_name = data_task_list[0]
                        task = data_task_list[1]
                        feature_dir = f"feature/{cfg.task}_eval/"
                        labels_filename = "outcomes.npy"
                    elif (
                        cfg.task == "zchsound_clean_murmurs"
                        or cfg.task == "zchsound_noisy_murmurs"
                    ):  # ZCHSound murmurs
                        data_task_list = cfg.task.split("_")
                        dataset_name = f"{data_task_list[0]}_{data_task_list[1]}"
                        task = data_task_list[2]
                        feature_dir = f"feature/{dataset_name}_eval/"
                        labels_filename = f"{task}.npy"
                    elif cfg.task == "pascal_A" or cfg.task == "pascal_B":
                        data_task_list = cfg.task.split("_")
                        dataset_name = data_task_list[0]
                        task = data_task_list[1]
                        feature_dir = f"feature/{cfg.task}_eval/"
                        labels_filename = "labels.npy"
                    elif cfg.task == "circor_murmurs" or cfg.task == "circor_outcomes":
                        data_task_list = cfg.task.split("_")
                        dataset_name = data_task_list[0]
                        task = data_task_list[1]
                        feature_dir = "feature/circor_eval/"
                        labels_filename = f"{data_task_list[1]}.npy"
                    elif cfg.task == "physionet16":
                        dataset_name = cfg.task
                        task = ""
                        feature_dir = f"feature/{cfg.task}_eval/"
                        labels_filename = "labels.npy"
                    auc = linear_evaluation_heart_cv(
                        seed=seed,
                        use_feature=feature,
                        l2_strength=l2_strength,
                        lr=lr,
                        loss=cfg.loss,
                        head=cfg.head,
                        epochs=64,
                        dataset_name=dataset_name,
                        task=task,
                        feature_dir=feature_dir,
                        labels_filename=labels_filename,
                        n_splits=5,
                    )

                auc_scores.append(auc)
                print("=" * 48)
                print(auc_scores)
                mean_auc = np.mean(auc_scores)
                print(
                    f"Mean AUC for l2_strength={l2_strength}, lr={lr}: {mean_auc:.3f} ± {np.std(auc_scores):.3f}"
                )
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = {"l2_strength": l2_strength, "lr": lr}

        print("=" * 48)
        print(f"Best AUC: {best_auc:.3f} with params: {best_params}")
        print("=" * 48)
    elif not cfg.LOOCV:
        # report mean and std for 5 runs with random seeds
        auc_scores = []
        for seed in range(cfg.n_run):
            # fix seeds for reproducibility
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            if cfg.task == "covid19sounds":
                auc = linear_evaluation_covid19sounds(
                    1,
                    feature,
                    modality=cfg.modality,
                    l2_strength=cfg.l2_strength,
                    lr=cfg.lr,
                    head=cfg.head,
                )
            elif cfg.task == "icbhidisease":
                auc = linear_evaluation_icbhidisease(
                    use_feature=feature,
                    epochs=64,
                    batch_size=32,
                    l2_strength=cfg.l2_strength,
                    lr=cfg.lr,
                    head=cfg.head,
                )
            elif cfg.task == "kauh":
                auc = linear_evaluation_kauh(
                    use_feature=feature,
                    epochs=50,
                    batch_size=32,
                    l2_strength=cfg.l2_strength,
                    lr=cfg.lr,
                    head=cfg.head,
                )
            elif cfg.task == "coswarasmoker":
                auc = linear_evaluation_coswara(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    batch_size=32,
                    lr=cfg.lr,
                    modality=cfg.modality,
                    label="smoker",
                    head=cfg.head,
                )
            elif cfg.task == "coswarasex":
                auc = linear_evaluation_coswara(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    batch_size=32,
                    lr=cfg.lr,
                    modality=cfg.modality,
                    label="sex",
                    head=cfg.head,
                )
            elif cfg.task == "copd":
                auc = linear_evaluation_copd(
                    use_feature=feature,
                    l2_strength=cfg.l2_strength,
                    lr=cfg.lr,
                    head=cfg.head,
                    epochs=64,
                )
            elif cfg.task == "coughvidcovid":
                auc = linear_evaluation_coughvid(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    lr=cfg.lr,
                    batch_size=64,
                    label="covid",
                    head=cfg.head,
                )
            elif cfg.task == "coughvidsex":
                auc = linear_evaluation_coughvid(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    lr=cfg.lr,
                    batch_size=64,
                    label="gender",
                    head=cfg.head,
                )
            elif cfg.task == "coviduk":
                auc = linear_evaluation_coviduk(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    lr=cfg.lr,
                    batch_size=64,
                    modality=cfg.modality,
                    head=cfg.head,
                )
            elif cfg.task == "snoring":
                auc = linear_evaluation_ssbpr(
                    use_feature=feature,
                    l2_strength=cfg.l2_strength,
                    lr=cfg.lr,
                    head=cfg.head,
                    epochs=32,
                    seed=seed,
                )
            else:
                if (
                    cfg.task == "zchsound_clean" or cfg.task == "zchsound_noisy"
                ):  # ZCHSound outcomes
                    data_task_list = cfg.task.split("_")
                    dataset_name = data_task_list[0]
                    task = data_task_list[1]
                    feature_dir = f"feature/{cfg.task}_eval/"
                    labels_filename = "outcomes.npy"
                elif (
                    cfg.task == "zchsound_clean_murmurs"
                    or cfg.task == "zchsound_noisy_murmurs"
                ):  # ZCHSound murmurs
                    data_task_list = cfg.task.split("_")
                    dataset_name = f"{data_task_list[0]}_{data_task_list[1]}"
                    task = data_task_list[2]
                    feature_dir = f"feature/{dataset_name}_eval/"
                    labels_filename = f"{task}.npy"
                elif cfg.task == "pascal_A" or cfg.task == "pascal_B":
                    data_task_list = cfg.task.split("_")
                    dataset_name = data_task_list[0]
                    task = data_task_list[1]
                    feature_dir = f"feature/{cfg.task}_eval/"
                    labels_filename = "labels.npy"
                elif cfg.task == "circor_murmurs" or cfg.task == "circor_outcomes":
                    data_task_list = cfg.task.split("_")
                    dataset_name = data_task_list[0]
                    task = data_task_list[1]
                    feature_dir = "feature/circor_eval/"
                    labels_filename = f"{data_task_list[1]}.npy"
                elif cfg.task == "physionet16":
                    dataset_name = cfg.task
                    task = ""
                    feature_dir = f"feature/{cfg.task}_eval/"
                    labels_filename = "labels.npy"
                auc = linear_evaluation_heart(
                    seed=seed,
                    use_feature=feature,
                    l2_strength=cfg.l2_strength,
                    lr=cfg.lr,
                    loss=cfg.loss,
                    head=cfg.head,
                    epochs=64,
                    dataset_name=dataset_name,
                    task=task,
                    feature_dir=feature_dir,
                    labels_filename=labels_filename,
                )
            auc_scores.append(auc)
        print("=" * 48)
        print(auc_scores)
        print(
            f"Five times mean task {cfg.task} feature {feature} results: auc mean {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}"
        )
        print("=" * 48)
    else:
        # Leave one out cross validation

        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        if cfg.task == "spirometry":
            maes, mapes = linear_evaluation_mmlung(
                use_feature=feature,
                method="LOOCV",
                l2_strength=1e-1,
                epochs=64,
                lr=1e-1,
                batch_size=64,
                modality=cfg.modality,
                label=cfg.label,
                head=cfg.head,
            )

        if cfg.task == "rr":
            maes, mapes = linear_evaluation_nosemic(
                use_feature=feature,
                method="LOOCV",
                l2_strength=1e-1,
                epochs=64,
                batch_size=64,
                lr=1e-4,
                head=cfg.head,
            )

        print("=" * 48)
        print(maes)
        print(mapes)
        print(
            f"Five times mean task {cfg.task} feature {feature} results: MAE mean {np.mean(maes):.3f} ± {np.std(maes):.3f}"
        )
        print(
            f"Five times mean task {cfg.task} feature {feature} results: MAPE mean {np.mean(mapes):.3f} ± {np.std(mapes):.3f}"
        )
        print("=" * 48)


if __name__ == "__main__":
    main()
