import collections
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

from src.model.models_eval import AudioClassifier, AudioClassifierAudioMAE, AudioClassifierCLAP
from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model
from src.benchmark.other_eval.finetuning import AudioDataset, get_wandb_name


def evaluate_model(
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
    metrics = [
        "circor_weighted_murmur_acc",
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
        raise FileNotFoundError(f"No checkpoint file found starting with {ck_prefix} in {ck_path_dir}")
        
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
    print(f"Label distribution: {collections.Counter(y_label)}")
    print(f"Unique labels: {collections.Counter(y_set)}")

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

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
    )

    trainer.test(model=model, dataloaders=train_loader, ckpt_path=None) # logging overwritten by test
    trainer.test(model=model, dataloaders=val_loader, ckpt_path=None) # logging overwritten by test
    test_res = trainer.test(model=model, dataloaders=test_loader, ckpt_path=None)
    
    auc = test_res[0]["test_auc"]
    
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
            if cfg.task == "circor_murmurs" or cfg.task == "circor_outcomes":
                task = cfg.task.split("_")[1]
                auc = evaluate_model(
                    pretrain=cfg.pretrain,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    feat_dim=cfg.dim,
                    dataset_name="circor",
                    task=task,
                    feature_dir="feature/circor_eval/",
                    labels_filename=f"{task}.npy",
                    seed=seed,
                )
            elif cfg.task == "zchsound_clean" or cfg.task == "zchsound_noisy": # ZCHSound outcomes
                task = cfg.task.split("_")[1]
                auc = evaluate_model(
                    pretrain=cfg.pretrain,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    feat_dim=cfg.dim,
                    dataset_name="zchsound",
                    task=task,
                    feature_dir=f"feature/{cfg.task}_eval/",
                    labels_filename="labels.npy",
                    seed=seed,
                )
            elif cfg.task == "zchsound_clean_murmurs" or cfg.task == "zchsound_noisy_murmurs": # ZCHSound murmurs
                data_task_list = cfg.task.split("_")
                dataset_name = f"{data_task_list[0]}_{data_task_list[1]}"
                task = data_task_list[2]
                auc = evaluate_model(
                    pretrain=cfg.pretrain,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    feat_dim=cfg.dim,
                    dataset_name=dataset_name,
                    task=task,
                    feature_dir=f"feature/{dataset_name}_eval/",
                    labels_filename=f"{task}.npy",
                    seed=seed,
                )
            elif cfg.task == "pascal_A" or cfg.task == "pascal_B":
                task = cfg.task.split("_")[1]
                auc = evaluate_model(
                    pretrain=cfg.pretrain,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    feat_dim=cfg.dim,
                    dataset_name="pascal",
                    task=task,
                    feature_dir=f"feature/{cfg.task}_eval/",
                    labels_filename="labels.npy",
                    seed=seed,
                )
            elif cfg.task == "physionet16":
                auc = evaluate_model(
                    pretrain=cfg.pretrain,
                    epochs=64,
                    l2_strength=cfg.l2_strength,
                    feat_dim=cfg.dim,
                    dataset_name="physionet16",
                    task="",
                    feature_dir=f"feature/{cfg.task}_eval/",
                    labels_filename="labels.npy",
                    seed=seed,
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