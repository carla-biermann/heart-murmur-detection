import collections
import random
import json

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
from torchmetrics import AUROC
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassSpecificity,
    MulticlassRecall,
    MulticlassPrecision,
)
import wandb

def circor_weighted_murmur_acc(predicted_tensor, y_tensor):
    """
    Compute the weighted murmur accuracy score.

    Class labels:
    0 = Absent
    1 = Present
    2 = Unknown

    Args:
        predicted_tensor (torch.Tensor): Predicted class indices (N,)
        y_tensor (torch.Tensor): Ground truth class indices (N,)

    Returns:
        float: Weighted murmur accuracy score
    """
    # Initialize 3x3 confusion matrix: [true_class][pred_class]
    cm = torch.zeros((3, 3), dtype=torch.int32)

    for true, pred in zip(y_tensor, predicted_tensor):
        cm[true.item(), pred.item()] += 1

    # Extract needed counts
    mPP = cm[1, 1]
    mPU = cm[1, 2]
    mPA = cm[1, 0]
    mUP = cm[2, 1]
    mUU = cm[2, 2]
    mUA = cm[2, 0]
    mAP = cm[0, 1]
    mAU = cm[0, 2]
    mAA = cm[0, 0]

    # Weighted accuracy formula
    numerator = 5 * mPP + 3 * mUU + mAA
    denominator = 5 * (mPP + mUP + mAP) + 3 * (mPU + mUU + mAU) + (mPA + mUA + mAA)

    if denominator == 0:
        return torch.tensor(0.0)

    return numerator.float() / denominator.float()


def circor_weighted_outcome_acc(predicted_tensor, y_tensor):
    """
    Compute the weighted murmur accuracy score.

    Class labels:
    0 = Abnormal
    1 = Normal

    Args:
        predicted_tensor (torch.Tensor): Predicted class indices (N,)
        y_tensor (torch.Tensor): Ground truth class indices (N,)

    Returns:
        float: Weighted murmur accuracy score
    """
    # Initialize 2x2 confusion matrix: [true_class][pred_class]
    cm = torch.zeros((2, 2), dtype=torch.int32)

    for true, pred in zip(y_tensor, predicted_tensor):
        cm[true.item(), pred.item()] += 1

    # Extract needed counts
    nTP = cm[0, 0]
    nFP = cm[1, 0]
    nFN = cm[0, 1]
    nTN = cm[1, 1]

    # Weighted accuracy formula
    numerator = 5 * nTP + nTN
    denominator = 5 * (nTP + nFN) + (nFP + nTN)

    if denominator == 0:
        return torch.tensor(0.0)

    return numerator.float() / denominator.float()


# Define total cost for algorithmic prescreening of m patients.
def cost_algorithm(m):
    return 10*m

# Define total cost for expert screening of m patients out of a total of n total patients.
def cost_expert(m, n):
    return (25 + 397*(m/n) -1718*(m/n)**2 + 11296*(m/n)**4) * n

# Define total cost for treatment of m patients.
def cost_treatment(m):
    return 10000*m

# Define total cost for missed/late treatement of m patients.
def cost_error(m):
    return 50000*m

# Compute Challenge cost metric.
def compute_cost(task, predicted_tensor, y_tensor):

    y_true = y_tensor.cpu().numpy()
    y_pred = predicted_tensor.cpu().numpy()

    # Define classes for referral.
    if task == "murmurs":
        referral_classes = [1, 2] # ['Present', 'Unknown']
    elif task == "outcomes":
        referral_classes = [0] # ['Abnormal']

    y_true_referral = np.isin(y_true, referral_classes)
    y_pred_referral = np.isin(y_pred, referral_classes)

    # Identify true positives, false positives, false negatives, and true negatives.
    tp = np.sum(y_true_referral & y_pred_referral)
    fp = np.sum(~y_true_referral & y_pred_referral)
    fn = np.sum(y_true_referral & ~y_pred_referral)
    tn = np.sum(~y_true_referral & ~y_pred_referral)
    total_patients = tp + fp + fn + tn

    # Compute total cost for all patients.
    total_cost = cost_algorithm(total_patients) \
        + cost_expert(tp + fp, total_patients) \
        + cost_treatment(tp) \
        + cost_error(fn)

    # Compute mean cost per patient.
    if total_patients > 0:
        mean_cost = total_cost / total_patients
    else:
        mean_cost = float('nan')

    return torch.tensor(mean_cost)

def compute_cost_murmurs(predicted_tensor, y_tensor):
    return compute_cost("murmurs", predicted_tensor, y_tensor)

def compute_cost_outcomes(predicted_tensor, y_tensor):
    return compute_cost("outcomes", predicted_tensor, y_tensor)


def initialize_metrics(classes, device, metrics, dataset, task):
    available_metrics = {
        "weighted_accuracy": MulticlassAccuracy(
            num_classes=classes, average="weighted"
        ).to(device),
        "weighted_auroc": MulticlassAUROC(num_classes=classes, average="weighted").to(
            device
        ),
        "weighted_specificity": MulticlassSpecificity(
            num_classes=classes, average="weighted"
        ).to(device),
        "weighted_recall": MulticlassRecall(num_classes=classes, average="weighted").to(
            device
        ),
        "weighted_precision": MulticlassPrecision(
            num_classes=classes, average="weighted"
        ).to(device),
        "weighted_F1": MulticlassF1Score(num_classes=classes, average="weighted").to(
            device
        ),
        "unweighted_accuracy": MulticlassAccuracy(num_classes=classes).to(device),
        "unweighted_recall": MulticlassRecall(num_classes=classes, average=None).to(
            device
        ),
        "avg_unweighted_recall": MulticlassRecall(
            num_classes=classes, average="macro"
        ).to(device),
        "unweighted_specificity": MulticlassSpecificity(
            num_classes=classes, average=None
        ).to(device),
        "avg_unweighted_specificity": MulticlassSpecificity(
            num_classes=classes, average="macro"
        ).to(device),
        "unweighted_precision": MulticlassPrecision(
            num_classes=classes, average=None
        ).to(device),
        "avg_unweighted_precision": MulticlassPrecision(
            num_classes=classes, average="macro"
        ).to(device),
    }
    if dataset == "circor" and task == "murmurs":
        available_metrics["circor_weighted_murmur_acc"] = circor_weighted_murmur_acc
        #available_metrics["circor_murmur_cost"] = compute_cost_murmurs
    elif dataset == "circor" and task == "outcomes":
        available_metrics["circor_weighted_outcome_acc"] = circor_weighted_outcome_acc
        available_metrics["circor_outcome_cost"] = compute_cost_outcomes
    selected_metrics = {}
    for metric in metrics:
        if metric in available_metrics:
            selected_metrics[metric] = available_metrics[metric]
        else:
            print(f"Unsupported metric: {metric}")
    return selected_metrics


def get_int_to_label_mapping(dataset, task):
    if dataset in "physionet16":
        file_path = f"feature/{dataset}_eval/int_to_label.json"
    # ZCHSound murmurs: dataset = "zchsound_clean" | "zchsound_noisy" and task = "murmurs"
    elif dataset in ["circor", "zchsound_clean", "zchsound_noisy"]:
        file_path = f"feature/{dataset}_eval/int_to_{task}.json"
    # ZCHSound outcomes: dataset = "zchsound" and task = "noisy" | "clean"
    # Goal: migrate this to same naming convention as ZCHSound murmurs
    elif dataset in ["pascal", "zchsound"]: 
        file_path = f"feature/{dataset}_{task}_eval/int_to_label.json"
    else:
        raise ValueError(f"No support for this dataset: {dataset}")

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


class AudioClassifier(pl.LightningModule):
    def __init__(
        self,
        net,
        head="linear",
        feat_dim=1280,
        classes=4,
        lr=1e-4,
        loss_func=None,
        freeze_encoder="none",
        l2_strength=0.0005,
        metrics=["auroc"],
        dataset=None,
        task=None,
    ):
        super().__init__()
        self.net = net
        self.freeze_encoder = freeze_encoder
        # self.l2_strength = l2_strength
        # print(self.net)
        if freeze_encoder == "all":
            for param in self.net.parameters():
                param.requires_grad = False
        elif freeze_encoder == "early":
            # print(self.net)
            # Selective freezing (fine-tuning only the last few layers), name not matching yet
            for name, param in self.net.named_parameters():
                # print(name)
                if (
                    "cnn1" in name
                    or "efficientnet._blocks.0." in name
                    or "efficientnet._blocks.1." in name
                    or "efficientnet._blocks.2." in name
                    or "efficientnet._blocks.3." in name
                    or "efficientnet._blocks.4." in name
                ):
                    # for efficientnet
                    param.requires_grad = True
                    print(name)
                elif (
                    "patch_embed" in name
                    or "layers.0" in name
                    or "layers.1" in name
                    or "layers.2" in name
                    or "htsat.norm" in name
                    or "htsat.head" in name
                    or "htsat.tscam_conv" in name
                ):
                    # for htsat
                    param.requires_grad = True
                    print(name)
                else:
                    param.requires_grad = False
                    # print(name)

        if head == "linear":
            print(feat_dim, classes)
            self.head = nn.Sequential(nn.Linear(feat_dim, classes))
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, classes),
            )
        else:
            raise NotImplementedError(f"head not supported: {head}")

        weights_init(self.head)
        self.lr = lr
        # self.l2_strength = l2_strength
        self.l2_strength_new_layers = l2_strength
        self.l2_strength_encoder = l2_strength * 0.2
        self.loss = loss_func if loss_func else nn.CrossEntropyLoss()
        self.classes = classes
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # self.fc.weight.data.normal_(mean=0.0, std=0.01)
        # self.fc.bias.data.zero_()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = initialize_metrics(classes, device, metrics, dataset, task)
        self.dataset = dataset
        self.task = task

    def forward_feature(self, x):
        return self.net(x)

    def forward(self, x):
        x = self.net(x)
        return self.head(x)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        probabilities = F.softmax(y_hat, dim=1)
        return probabilities

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x) + 1e-10
        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)

        # Apply L2 regularization on head
        l2_regularization = 0
        for param in self.head.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_head", l2_regularization)
        loss += self.l2_strength_new_layers * l2_regularization

        # Apply L2 regularization on encoder
        l2_regularization = 0
        for param in self.net.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_encoder", l2_regularization)
        loss += self.l2_strength_encoder * l2_regularization

        probabilities = F.softmax(y_hat, dim=1)
        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("train_acc", acc)

        # Compute and log selected metrics
        self.log_metrics("train", probabilities, predicted, y)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_hat = self(x)

        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

        self.validation_step_outputs.append(
            (y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy())
        )

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.test_step_outputs.append(
            (y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy())
        )

    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        y_tensor = torch.from_numpy(y).to(device)
        predicted_tensor = torch.from_numpy(predicted).to(device)
        probs_tensor = torch.from_numpy(probs).to(device)

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))

        # print("valid_auc", auc)
        self.log("valid_auc", auc)

        # Compute and log selected metrics
        self.log_metrics("val", probs_tensor, predicted_tensor, y_tensor)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        all_outputs = self.test_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        y_tensor = torch.from_numpy(y).to(device)
        predicted_tensor = torch.from_numpy(predicted).to(device)
        probs_tensor = torch.from_numpy(probs).to(device)

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))

        print("test_auc", auc)
        self.log("test_auc", auc)

        # Compute and log selected metrics
        self.log_metrics("test", probs_tensor, predicted_tensor, y_tensor)

        self.test_step_outputs.clear()
        
        return auc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def log_metrics(self, split, probs_tensor, predicted_tensor, y_tensor):
        for metric_name, metric in self.metrics.items():
            metric_value = metric(
                probs_tensor if "auroc" in metric_name else predicted_tensor, y_tensor
            )
            if metric_value.numel() > 1:
                label_dict = get_int_to_label_mapping(self.dataset, self.task)
                for i, val in enumerate(metric_value):
                    label = label_dict[str(i)]
                    self.log(f"{split}_{metric_name}_{label}", val, prog_bar=True)
                    wandb.log({f"{split}_{metric_name}_{label}": val.item()})
            else:
                self.log(f"{split}_{metric_name}", metric_value, prog_bar=True)
                wandb.log({f"{split}_{metric_name}": metric_value.item()})


class AudioClassifierAudioMAE(pl.LightningModule):
    def __init__(
        self,
        net,
        head="linear",
        feat_dim=1280,
        classes=4,
        lr=1e-4,
        loss_func=None,
        freeze_encoder="none",
        l2_strength=0.0005,
        metrics=["auroc"],
        dataset=None,
        task=None,
    ):
        super().__init__()
        self.net = net
        self.freeze_encoder = freeze_encoder

        # print(self.net)

        if head == "linear":
            print(feat_dim, classes)
            self.head = nn.Sequential(nn.Linear(feat_dim, classes))
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, classes),
            )
        else:
            raise NotImplementedError(f"head not supported: {head}")

        weights_init(self.head)
        self.lr = lr
        # self.l2_strength = l2_strength
        self.l2_strength_new_layers = l2_strength
        self.l2_strength_encoder = l2_strength * 0.2
        self.loss = loss_func if loss_func else nn.CrossEntropyLoss()
        self.classes = classes
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # self.fc.weight.data.normal_(mean=0.0, std=0.01)
        # self.fc.bias.data.zero_()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = initialize_metrics(classes, device, metrics, dataset, task)
        self.dataset = dataset
        self.task = task

    def forward_feature(self, x):
        return self.net.forward_feature(x)

    def forward(self, x):
        x = self.net.forward_feature(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x) + 1e-10
        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)

        # Apply L2 regularization on head
        l2_regularization = 0
        for param in self.head.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_head", l2_regularization)
        loss += self.l2_strength_new_layers * l2_regularization

        # Apply L2 regularization on encoder
        l2_regularization = 0
        for param in self.net.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_encoder", l2_regularization)
        loss += self.l2_strength_encoder * l2_regularization

        probabilities = F.softmax(y_hat, dim=1)
        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("train_acc", acc)

        # Compute and log selected metrics
        self.log_metrics("train", probabilities, predicted, y)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_hat = self(x)

        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

        self.validation_step_outputs.append(
            (y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy())
        )

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.test_step_outputs.append(
            (y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy())
        )

    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        y_tensor = torch.from_numpy(y).to(device)
        predicted_tensor = torch.from_numpy(predicted).to(device)
        probs_tensor = torch.from_numpy(probs).to(device)

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))

        # print("valid_auc", auc)
        self.log("valid_auc", auc)

        # Compute and log selected metrics
        self.log_metrics("val", probs_tensor, predicted_tensor, y_tensor)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        all_outputs = self.test_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        y_tensor = torch.from_numpy(y).to(device)
        predicted_tensor = torch.from_numpy(predicted).to(device)
        probs_tensor = torch.from_numpy(probs).to(device)

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))

        print("test_auc", auc)
        self.log("test_auc", auc)

        # Compute and log selected metrics
        self.log_metrics("test", probs_tensor, predicted_tensor, y_tensor)

        self.test_step_outputs.clear()
        return auc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def log_metrics(self, split, probs_tensor, predicted_tensor, y_tensor):
        for metric_name, metric in self.metrics.items():
            metric_value = metric(
                probs_tensor if "auroc" in metric_name else predicted_tensor, y_tensor
            )
            if metric_value.numel() > 1:
                label_dict = get_int_to_label_mapping(self.dataset, self.task)
                for i, val in enumerate(metric_value):
                    label = label_dict[str(i)]
                    self.log(f"{split}_{metric_name}_{label}", val, prog_bar=True)
                    wandb.log({f"{split}_{metric_name}_{label}": val.item()})
            else:
                self.log(f"{split}_{metric_name}", metric_value, prog_bar=True)
                wandb.log({f"{split}_{metric_name}": metric_value.item()})


class AudioClassifierCLAP(pl.LightningModule):
    def __init__(
        self,
        net,
        head="linear",
        feat_dim=1280,
        classes=4,
        lr=1e-4,
        loss_func=None,
        freeze_encoder="none",
        l2_strength=0.0005,
        metrics=["auroc"],
        dataset=None,
        task=None,
    ):
        super().__init__()
        self.net = net
        self.freeze_encoder = freeze_encoder
        # self.l2_strength = l2_strength
        # print(self.net)
        self.net.train()

        if head == "linear":
            print(feat_dim, classes)
            self.head = nn.Sequential(nn.Linear(feat_dim, classes))
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, classes),
            )
        else:
            raise NotImplementedError(f"head not supported: {head}")

        weights_init(self.head)
        self.lr = lr
        # self.l2_strength = l2_strength
        self.l2_strength_new_layers = l2_strength
        self.l2_strength_encoder = l2_strength * 0.2
        self.loss = loss_func if loss_func else nn.CrossEntropyLoss()
        self.classes = classes
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # self.fc.weight.data.normal_(mean=0.0, std=0.01)
        # self.fc.bias.data.zero_()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = initialize_metrics(classes, device, metrics, dataset, task)
        self.dataset = dataset
        self.task = task

    def default_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
        ):
            if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        self.default_collate_err_msg_format.format(elem.dtype)
                    )

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(
                *(self.default_collate(samples) for samples in zip(*batch))
            )
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    "each element in list of batch should be of equal size"
                )
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))

    def read_audio(self, audio_path, resample=True):
        r"""Loads audio file or array and returns a torch tensor"""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)

        resample_rate = 16000
        # print(sample_rate)
        if resample and resample_rate != sample_rate:
            import torchaudio.transforms as T

            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
        return audio_time_series, resample_rate

    def load_audio_into_tensor(self, audio_path, audio_duration, resample=False):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = self.read_audio(audio_path, resample=resample)
        audio_time_series = audio_time_series.reshape(-1)

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if audio_duration * sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(
                np.ceil((audio_duration * sample_rate) / audio_time_series.shape[0])
            )
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0 : audio_duration * sample_rate]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration * sample_rate
            )
            audio_time_series = audio_time_series[
                start_index : start_index + audio_duration * sample_rate
            ]
        return torch.FloatTensor(audio_time_series)

    def preprocess_audio(self, audio_files, resample):
        r"""Load list of audio files and return raw audio"""
        audio_tensors = []
        for audio_file in audio_files:
            audio_tensor = self.load_audio_into_tensor(audio_file, 5, resample)
            audio_tensor = (
                audio_tensor.reshape(1, -1).cuda()
                if torch.cuda.is_available()
                else audio_tensor.reshape(1, -1)
            )
            audio_tensors.append(audio_tensor)
        return self.default_collate(audio_tensors)

    def forward_feature(self, x, resample=True):
        preprocessed_audio = self.preprocess_audio(x, resample)
        preprocessed_audio = preprocessed_audio.reshape(
            preprocessed_audio.shape[0], preprocessed_audio.shape[2]
        )
        audio_embed, _ = self.net(preprocessed_audio)
        return audio_embed

    def forward(self, x, resample=True):
        preprocessed_audio = self.preprocess_audio(x, resample)
        preprocessed_audio = preprocessed_audio.reshape(
            preprocessed_audio.shape[0], preprocessed_audio.shape[2]
        )
        # print(preprocessed_audio.shape)
        # return self._get_audio_embeddings(preprocessed_audio)
        # x = self.net.get_audio_embeddings(x)
        audio_embed, _ = self.net(preprocessed_audio)
        # print(audio_embed.shape)
        return self.head(audio_embed)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x) + 1e-10
        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y.long())
        self.log("train_loss", loss)

        # Apply L2 regularization on head
        l2_regularization = 0
        for param in self.head.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_head", l2_regularization)
        loss += self.l2_strength_new_layers * l2_regularization

        # Apply L2 regularization on encoder
        l2_regularization = 0
        # for param in self.net.clap.audio_encoder.parameters():
        for param in self.net.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_encoder", l2_regularization)
        loss += self.l2_strength_encoder * l2_regularization

        probabilities = F.softmax(y_hat, dim=1)
        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("train_acc", acc)

        # Compute and log selected metrics
        self.log_metrics("train", probabilities, predicted, y)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_hat = self(x)

        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y.long())

        probabilities = F.softmax(y_hat, dim=1)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

        self.validation_step_outputs.append(
            (y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy())
        )

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y.long())

        probabilities = F.softmax(y_hat, dim=1)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.test_step_outputs.append(
            (y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy())
        )

    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        y_tensor = torch.from_numpy(y).to(device)
        predicted_tensor = torch.from_numpy(predicted).to(device)
        probs_tensor = torch.from_numpy(probs).to(device)

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))

        # print("valid_auc", auc)
        self.log("valid_auc", auc)

        # Compute and log selected metrics
        self.log_metrics("val", probs_tensor, predicted_tensor, y_tensor)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        all_outputs = self.test_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        y_tensor = torch.from_numpy(y).to(device)
        predicted_tensor = torch.from_numpy(predicted).to(device)
        probs_tensor = torch.from_numpy(probs).to(device)

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))

        print("test_auc", auc)
        self.log("test_auc", auc)

        # Compute and log selected metrics
        self.log_metrics("test", probs_tensor, predicted_tensor, y_tensor)

        self.test_step_outputs.clear()
        return auc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def log_metrics(self, split, probs_tensor, predicted_tensor, y_tensor):
        for metric_name, metric in self.metrics.items():
            metric_value = metric(
                probs_tensor if "auroc" in metric_name else predicted_tensor, y_tensor
            )
            if metric_value.numel() > 1:
                label_dict = get_int_to_label_mapping(self.dataset, self.task)
                for i, val in enumerate(metric_value):
                    label = label_dict[str(i)]
                    self.log(f"{split}_{metric_name}_{label}", val, prog_bar=True)
                    wandb.log({f"{split}_{metric_name}_{label}": val.item()})
            else:
                self.log(f"{split}_{metric_name}", metric_value, prog_bar=True)
                wandb.log({f"{split}_{metric_name}": metric_value.item()})


class LinearHead(pl.LightningModule):
    def __init__(
        self,
        net=None,
        head="linear",
        feat_dim=1280,
        classes=4,
        from_feature=True,
        lr=1e-4,
        loss_func=None,
        l2_strength=0.0005,
        metrics=["auroc"],
        dataset=None,
        task=None,
    ):
        super().__init__()
        self.from_feature = from_feature

        if not from_feature:
            if net is None:
                raise ValueError("no encoder given and not from feature input")
            self.net = net
            for param in self.net.parameters():
                param.requires_grad = False

        if head == "linear":
            print(feat_dim, classes)
            self.head = nn.Sequential(nn.Linear(feat_dim, classes))
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, classes),
            )
        else:
            raise NotImplementedError(f"head not supported: {head}")

        weights_init(self.head)
        self.lr = lr
        self.l2_strength = l2_strength
        self.loss = loss_func if loss_func else nn.CrossEntropyLoss()
        self.classes = classes
        self.validation_step_outputs = []
        self.test_step_outputs = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = initialize_metrics(classes, device, metrics, dataset, task)
        self.dataset = dataset
        self.task = task

    def forward(self, x):
        if self.from_feature:
            return self.head(x)

        with torch.no_grad():
            x = self.net(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x) + 1e-10
        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        # Apply L2 regularization
        l2_regularization = 0
        for param in self.head.parameters():
            l2_regularization += param.pow(2).sum()
        self.log("train_l2", l2_regularization)
        loss += self.l2_strength * l2_regularization

        probabilities = F.softmax(y_hat, dim=1)
        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("train_acc", acc)
        # print("train_loss", loss)
        # print("train_acc", acc)f

        # Compute and log selected metrics
        self.log_metrics("train", probabilities, predicted, y)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_hat = self(x)

        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

        self.validation_step_outputs.append(
            (y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy())
        )

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.test_step_outputs.append(
            (y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy())
        )

    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        y_tensor = torch.from_numpy(y).to(device)
        predicted_tensor = torch.from_numpy(predicted).to(device)
        probs_tensor = torch.from_numpy(probs).to(device)

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(probs_tensor, y_tensor)

        # print("valid_auc", auc)
        self.log("valid_auc", auc)

        # Compute and log selected metrics
        self.log_metrics("val", probs_tensor, predicted_tensor, y_tensor)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        all_outputs = self.test_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        y_tensor = torch.from_numpy(y).to(device)
        predicted_tensor = torch.from_numpy(predicted).to(device)
        probs_tensor = torch.from_numpy(probs).to(device)

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(probs_tensor, y_tensor)

        print("test_auc", auc)
        self.log("test_auc", auc)

        # Compute and log selected metrics
        self.log_metrics("test", probs_tensor, predicted_tensor, y_tensor)

        self.test_step_outputs.clear()
        return auc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def log_metrics(self, split, probs_tensor, predicted_tensor, y_tensor):
        for metric_name, metric in self.metrics.items():
            metric_value = metric(
                probs_tensor if "auroc" in metric_name else predicted_tensor, y_tensor
            )
            if metric_value.numel() > 1:
                label_dict = get_int_to_label_mapping(self.dataset, self.task)
                for i, val in enumerate(metric_value):
                    label = label_dict[str(i)]
                    self.log(f"{split}_{metric_name}_{label}", val, prog_bar=True)
                    wandb.log({f"{split}_{metric_name}_{label}": val.item()})
            else:
                self.log(f"{split}_{metric_name}", metric_value, prog_bar=True)
                wandb.log({f"{split}_{metric_name}": metric_value.item()})


class LinearHeadR(pl.LightningModule):
    def __init__(
        self,
        net=None,
        head="linear",
        feat_dim=1280,
        output_dim=1,
        from_feature=True,
        lr=1e-4,
        loss_func=None,
        l2_strength=0.0005,
        random_seed=1,
        mean=0,
        std=0,
    ):
        super().__init__()
        self.from_feature = from_feature

        if not from_feature:
            if net is None:
                raise ValueError("no encoder given and not from feature input")
            self.net = net
            for param in self.net.parameters():
                param.requires_grad = False

        if head == "linear":
            # print(feat_dim, output_dim)
            self.head = nn.Sequential(nn.Linear(feat_dim, output_dim))
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, output_dim),
            )
        else:
            raise NotImplementedError(f"head not supported: {head}")

        # self.head = nn.Linear(dim_in, dim_out)

        weights_init(self.head)
        self.lr = lr
        self.l2_strength = l2_strength
        self.loss = loss_func if loss_func else nn.MSELoss()
        self.classes = output_dim
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.mean = mean
        self.std = std

        # self.fc.weight.data.normal_(mean=0.0, std=0.01)
        # self.fc.bias.data.zero_()

    def forward(self, x):
        if self.from_feature:
            # x = (x-self.mean)/self.std
            y = self.head(x)

            return y * self.std + self.mean

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) + 1e-10
        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)
        # print('training loss:', loss.item())

        # Apply L2 regularization
        l2_regularization = 0
        for param in self.head.parameters():
            l2_regularization += param.pow(2).sum()
        loss += self.l2_strength * l2_regularization

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        mae = torch.mean(torch.abs(y_hat - y))
        mape = torch.mean(torch.abs((y_hat - y) / y)) * 100

        self.log("valid_loss", loss)
        self.log("valid_MAE", mae)
        self.log("valid_MAPE", mape)

        self.validation_step_outputs.append((y.cpu().numpy(), y_hat.cpu().numpy()))

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)
        mae = torch.mean(torch.abs(y_hat - y))
        mape = torch.mean(torch.abs((y_hat - y) / y)) * 100

        self.log("test_loss", loss)
        self.log("test_MAE", mae)
        self.log("test_MAPE", mape)
        self.test_step_outputs.append((y.cpu().numpy(), y_hat.cpu().numpy()))

    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        y_hat = np.concatenate([output[1] for output in all_outputs])

        mae = np.mean(np.abs(y_hat - y))
        mape = np.mean(np.abs((y_hat - y) / y)) * 100
        mse = np.mean((y - y_hat) ** 2)
        self.log("valid_MAE", mae)
        self.log("valid_MAPE", mape)
        self.log("valid_loss", mse)
        # print('valid_mae:', mae, 'y:', y[0], 'valid y_hat:', y_hat[0])

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        all_outputs = self.test_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        y_hat = np.concatenate([output[1] for output in all_outputs])

        mae = np.mean(np.abs(y_hat - y))
        mape = np.mean(np.abs((y_hat - y) / y))
        mse = np.mean((y - y_hat) ** 2)
        self.log("test_MAE", mae)
        self.log("test_MAPE", mape)
        self.log("test_loss", mse)
        # print('test_mae:', mae, y_hat, y)

        self.test_step_outputs.clear()
        return mae, mape

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        # return torch.optim.SGD(self.parameters(), lr=self.lr)


def weights_init(network):
    for m in network:
        classname = m.__class__.__name__
        # print(classname)
        if classname.find("Linear") != -1:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()
