import collections
import time

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.models_eval import LinearHead, LinearHeadR
from src.util import downsample_balanced_dataset, train_test_split_from_list, get_weights_tensor


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
            f"Task not implemented: Covid-19 sounds task {task}, please check the args."
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
        val_data, batch_size=batch_size, num_workers=1, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=1
    )

    model = LinearHead(
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # these args need to be entered according to tasks
    parser.add_argument("--task", type=str, default="covid19sounds")
    parser.add_argument("--label", type=str, default="smoker")  # prediction target
    parser.add_argument("--modality", type=str, default="cough")
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--LOOCV", type=bool, default=False)

    # these can follow default
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2_strength", type=float, default=1e-5)
    parser.add_argument("--head", type=str, default="linear")
    parser.add_argument(
        "--mapgoogle", type=bool, default=False
    )  # align test set with HeAR
    parser.add_argument("--n_run", type=int, default=5)

    args = parser.parse_args()

    feature = args.pretrain
    if (
        feature not in ["vggish", "opensmile", "clap", "audiomae", "hear",  "clap2023"]
        and "finetuned" not in feature
    ):  # baselines
        feature += str(args.dim)

    if not args.LOOCV:
        # report mean and std for 5 runs with random seeds
        auc_scores = []
        for seed in range(args.n_run):
            # fix seeds for reproducibility
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            if args.task == "covid19sounds":
                auc = linear_evaluation_covid19sounds(
                    1,
                    feature,
                    modality=args.modality,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                )
            elif args.task == "icbhidisease":
                auc = linear_evaluation_icbhidisease(
                    use_feature=feature,
                    epochs=64,
                    batch_size=32,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                )
            elif args.task == "kauh":
                auc = linear_evaluation_kauh(
                    use_feature=feature,
                    epochs=50,
                    batch_size=32,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                )
            elif args.task == "coswarasmoker":
                auc = linear_evaluation_coswara(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    batch_size=32,
                    lr=args.lr,
                    modality=args.modality,
                    label="smoker",
                    head=args.head,
                )
            elif args.task == "coswarasex":
                auc = linear_evaluation_coswara(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    batch_size=32,
                    lr=args.lr,
                    modality=args.modality,
                    label="sex",
                    head=args.head,
                )
            elif args.task == "copd":
                auc = linear_evaluation_copd(
                    use_feature=feature,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                    epochs=64,
                )
            elif args.task == "coughvidcovid":
                auc = linear_evaluation_coughvid(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    batch_size=64,
                    label="covid",
                    head=args.head,
                )
            elif args.task == "coughvidsex":
                auc = linear_evaluation_coughvid(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    batch_size=64,
                    label="gender",
                    head=args.head,
                )
            elif args.task == "coviduk":
                auc = linear_evaluation_coviduk(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    batch_size=64,
                    modality=args.modality,
                    head=args.head,
                )
            elif args.task == "snoring":
                auc = linear_evaluation_ssbpr(
                    use_feature=feature,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                    epochs=32,
                    seed=seed,
                )
            elif args.task == "zchsound_clean" or args.task == "zchsound_noisy":
                data_task_list = args.task.split("_")
                auc = linear_evaluation_heart(
                    seed=seed,
                    use_feature=feature,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                    epochs=64,
                    dataset_name=data_task_list[0],
                    task=data_task_list[1],
                    feature_dir=f"feature/{args.task}_eval/",
                    labels_filename="labels.npy",
                )
            elif args.task == "pascal_A" or args.task == "pascal_B":
                data_task_list = args.task.split("_")
                auc = linear_evaluation_heart(
                    seed=seed,
                    use_feature=feature,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                    epochs=64,
                    dataset_name=data_task_list[0],
                    task=data_task_list[1],
                    feature_dir=f"feature/{args.task}_eval/",
                    labels_filename="labels.npy",
                )
            elif args.task == "circor_murmurs" or args.task == "circor_outcomes":
                data_task_list = args.task.split("_")
                auc = linear_evaluation_heart(
                    seed=seed,
                    use_feature=feature,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                    epochs=64,
                    dataset_name=data_task_list[0],
                    task=data_task_list[1],
                    feature_dir="feature/circor_eval/",
                    labels_filename=f"{data_task_list[1]}.npy",
                )
            elif args.task == "physionet16":
                auc = linear_evaluation_heart(
                    seed=seed,
                    use_feature=feature,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                    epochs=64,
                    dataset_name=args.task,
                    task="",
                    feature_dir=f"feature/{args.task}_eval/",
                    labels_filename="labels.npy",
                )
            auc_scores.append(auc)
        print("=" * 48)
        print(auc_scores)
        print(
            f"Five times mean task {args.task} feature {feature} results: auc mean {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}"
        )
        print("=" * 48)
    else:
        # Leave one out cross validation

        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        if args.task == "spirometry":
            maes, mapes = linear_evaluation_mmlung(
                use_feature=feature,
                method="LOOCV",
                l2_strength=1e-1,
                epochs=64,
                lr=1e-1,
                batch_size=64,
                modality=args.modality,
                label=args.label,
                head=args.head,
            )

        if args.task == "rr":
            maes, mapes = linear_evaluation_nosemic(
                use_feature=feature,
                method="LOOCV",
                l2_strength=1e-1,
                epochs=64,
                batch_size=64,
                lr=1e-4,
                head=args.head,
            )

        print("=" * 48)
        print(maes)
        print(mapes)
        print(
            f"Five times mean task {args.task} feature {feature} results: MAE mean {np.mean(maes):.3f} ± {np.std(maes):.3f}"
        )
        print(
            f"Five times mean task {args.task} feature {feature} results: MAPE mean {np.mean(mapes):.3f} ± {np.std(mapes):.3f}"
        )
        print("=" * 48)
