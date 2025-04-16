import numpy as np
import os
import glob
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch import seed_everything

from src.model.models_eval import (
    AudioClassifier,
    AudioClassifierAudioMAE,
    AudioClassifierCLAP,
    LinearHead,
)
from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model
from src.benchmark.other_eval.finetuning import AudioDataset, get_wandb_name
from src.benchmark.linear_eval import FeatureDataset


def evaluate_linear_head(
    seed,
    metrics,
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
    y_set = np.load(feature_dir + "train_test_split.npy")
    y_label = np.load(feature_dir + labels_filename)
    x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()

    feat_dim = x_data.shape[1]
    n_cls = len(set(y_label))

    x_data_test = x_data[y_set == "test"]
    y_label_test = y_label[y_set == "test"]

    test_data = FeatureDataset((x_data_test, y_label_test))

    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=1
    )

    ck_path_dir = (
        f"cks/linear/{dataset_name}_{task}/" if task else f"cks/linear/{dataset_name}"
    )
    ck_prefix = "_".join(
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

    # Search for the checkpoint file(s) that match
    ck_files = glob.glob(os.path.join(ck_path_dir, f"{ck_prefix}*.ckpt"))

    # Pick the first match (or handle multiple matches if needed)
    if not ck_files:
        raise FileNotFoundError(
            f"No checkpoint file found starting with {ck_prefix} in {ck_path_dir}"
        )

    ckpt_path = ck_files[0]
    print(f"Found checkpoint: {ckpt_path}")

    model = LinearHead.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        feat_dim=feat_dim,
        classes=n_cls,
        l2_strength=l2_strength,
        head=head,
        metrics=metrics,
        dataset=dataset_name,
        task=task,
    )

    wandb_logger = WandbLogger(
        project="Heart-Sound-Analysis",
        name=get_wandb_name(use_feature, f"{dataset_name}-{task}", head),
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
            "eval_only": True,
        }
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
    )

    test_res = trainer.test(model=model, dataloaders=test_loader, ckpt_path=None)

    auc = test_res[0]["test_auc"]

    wandb.finish()
    return auc


def evaluate_finetuned_model(
    seed,
    metrics,
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
    ck_path_dir = (
        f"cks/finetune/{dataset_name}_{task}/"
        if task
        else f"cks/finetune/{dataset_name}"
    )
    ck_prefix = "_".join(
        [
            "finetuning",
            head,
            pretrain,
            str(batch_size),
            str(lr),
            str(epochs),
            str(l2_strength),
            str(seed),
        ]
    )

    # Search for the checkpoint file(s) that match
    ck_files = glob.glob(os.path.join(ck_path_dir, f"{ck_prefix}*.ckpt"))

    # Pick the first match (or handle multiple matches if needed)
    if not ck_files:
        raise FileNotFoundError(
            f"No checkpoint file found starting with {ck_prefix} in {ck_path_dir}"
        )

    ckpt_path = ck_files[0]
    print(f"Found checkpoint: {ckpt_path}")

    wandb_logger = WandbLogger(
        project="Heart-Sound-Analysis-FT",
        name=get_wandb_name(pretrain, f"{dataset_name}-{task}", head),
        log_model=False,
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
            "eval_only": True,
        }
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

        model = AudioClassifierAudioMAE.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            net=net,
            head=head,
            classes=n_cls,
            lr=lr,
            l2_strength=l2_strength,
            feat_dim=feat_dim,
            metrics=metrics,
            dataset=dataset_name,
            task=task,
        )

    elif pretrain == "clap":
        from src.benchmark.baseline.msclap import CLAP

        audio_files = np.load(feature_dir + "sound_dir_loc.npy")
        x_data = np.array(audio_files)
        clap_model = CLAP(version="2022", use_cuda=True)
        net = clap_model.clap.audio_encoder
        model = AudioClassifierCLAP.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            net=net,
            head=head,
            feat_dim=feat_dim,
            classes=n_cls,
            lr=lr,
            l2_strength=l2_strength,
            metrics=metrics,
            dataset=dataset_name,
            task=task,
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
            model = AudioClassifierAudioMAE.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                net=pretrained_model,
                classes=n_cls,
                lr=lr,
                l2_strength=l2_strength,
                feat_dim=feat_dim,
                metrics=metrics,
                dataset=dataset_name,
                task=task,
            )
        else:
            freeze_encoder = "early" if pretrain == "operaCE" else "none"
            net = pretrained_model.encoder
            model = AudioClassifier.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                net=net,
                head=head,
                classes=n_cls,
                lr=lr,
                l2_strength=l2_strength,
                feat_dim=feat_dim,
                freeze_encoder=freeze_encoder,
                metrics=metrics,
                dataset=dataset_name,
                task=task,
            )

    wandb_logger.experiment.config.update({"freeze_encoder": freeze_encoder})

    y_set = np.load(feature_dir + "train_test_split.npy")
    y_label = np.load(feature_dir + labels_filename)

    x_data_test = x_data[y_set == "test"]
    y_label_test = y_label[y_set == "test"]

    test_data = AudioDataset(
        (x_data_test, y_label_test), augment=False, max_len=False, from_audio=from_audio
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=2
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
    )

    test_res = trainer.test(model=model, dataloaders=test_loader, ckpt_path=None)

    auc = test_res[0]["test_auc"]

    wandb.finish()
    return auc


def evaluate_model(cfg, seed):
    eval_args = dict(
        seed=seed,
        metrics=cfg.metrics,
        epochs=64,
        l2_strength=cfg.l2_strength,
    )

    if cfg.task == "physionet16":
        dataset_name = "physionet16"
        task = ""
        labels_filename = "labels.npy"
        feature_dir = f"feature/{cfg.task}_eval/"
    elif "zchsound" in cfg.task:
        parts = cfg.task.split("_")
        if "murmurs" in parts:  # new dataset / task notation for zchsound murmurs
            dataset_name = f"{parts[0]}_{parts[1]}"
            task = parts[2]
            labels_filename = f"{task}.npy"
            feature_dir = f"feature/{dataset_name}_eval/"
        else:
            dataset_name = parts[0]
            task = parts[1]
            labels_filename = "outcomes.npy"
            feature_dir=f"feature/{cfg.task}_eval/"
    elif "circor" in cfg.task:
        dataset_name = "circor"
        task = cfg.task.split("_")[1]
        labels_filename = f"{task}.npy"
        feature_dir = "feature/circor_eval/"
    elif "pascal" in cfg.task:
        dataset_name = "pascal"
        task = cfg.task.split("_")[1]
        labels_filename = "labels.npy"
        feature_dir = f"feature/{cfg.task}_eval/"
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    eval_args.update(
        dataset_name=dataset_name,
        task=task,
        labels_filename=labels_filename,
        feature_dir=feature_dir,
    )

    if cfg.head_only:
        return evaluate_linear_head(
            use_feature=cfg.pretrain, head="linear", **eval_args
        )
    else:
        return evaluate_finetuned_model(
            pretrain=cfg.pretrain, head="linear", feat_dim=cfg.dim, **eval_args
        )


@hydra.main(config_path="../configs", config_name="eval_config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if not cfg.LOOCV:
        auc_scores = []
        for seed in range(cfg.n_run):
            seed_everything(seed, workers=True)
            auc = evaluate_model(cfg, seed)
            auc_scores.append(auc)
        print("=" * 48)
        print(auc_scores)
        print(
            f"Five times mean task {cfg.task} finetuning from {cfg.pretrain} results: auc mean {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}"
        )
        print("=" * 48)

if __name__ == '__main__':
    main()