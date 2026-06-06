from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    TextClassificationDataset,
    apply_configured_text_features,
    get_label_names,
    group_overlap,
    read_excel_dataset,
    split_train_valid,
    summarize_dataframe,
)
from src.losses import FocalLoss, ProtoSupConLoss, SupConLoss  # noqa: E402
from src.model import RobertaSupConClassifier  # noqa: E402
from src.utils import FGM, compute_metrics, ensure_dir, load_config, save_json, set_seed  # noqa: E402


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_eval(model, dataloader, ce_loss_fn, supcon_loss_fn, device, lambda_contrast):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_contrast_loss = 0.0
    total_examples = 0
    total_valid_anchors = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"]
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                location_priority_ids=batch.get("location_priority_ids"),
            )
            cls_loss = ce_loss_fn(outputs["logits"], labels)
            contrast_loss, valid_anchors = supcon_loss_fn(outputs["contrastive_embedding"], labels)
            loss = cls_loss + lambda_contrast * contrast_loss

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_cls_loss += float(cls_loss.item()) * batch_size
            total_contrast_loss += float(contrast_loss.item()) * batch_size
            total_examples += batch_size
            total_valid_anchors += valid_anchors
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(outputs["logits"].argmax(dim=1).detach().cpu().tolist())

    return {
        "loss": total_loss / max(total_examples, 1),
        "classification_loss": total_cls_loss / max(total_examples, 1),
        "contrastive_loss": total_contrast_loss / max(total_examples, 1),
        "valid_contrastive_anchors": int(total_valid_anchors),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))

    output_dir = ensure_dir(cfg["output_dir"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    logs_dir = ensure_dir(output_dir / "logs")
    prediction_dir = ensure_dir(output_dir / "predictions")

    train_df = read_excel_dataset(cfg["train_path"])
    test_df = read_excel_dataset(cfg["test_path"])
    train_df = apply_configured_text_features(train_df, cfg)
    test_df = apply_configured_text_features(test_df, cfg)
    label_names = get_label_names(train_df)
    split = split_train_valid(
        train_df,
        validation_split=float(cfg["validation_split"]),
        seed=int(cfg["seed"]),
        strategy=str(cfg["split_strategy"]),
    )

    data_summary = {
        "train_all": summarize_dataframe(train_df, label_names),
        "test": summarize_dataframe(test_df, label_names),
        "train_split": summarize_dataframe(split.train_df, label_names),
        "valid_split": summarize_dataframe(split.valid_df, label_names),
        "split_method": split.split_method,
        "group_overlap": group_overlap(split.train_df, split.valid_df),
    }
    save_json(data_summary, logs_dir / "data_summary.json")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    train_dataset = TextClassificationDataset(split.train_df, tokenizer, int(cfg["max_length"]), has_labels=True)
    valid_dataset = TextClassificationDataset(split.valid_df, tokenizer, int(cfg["max_length"]), has_labels=True)
    test_dataset = TextClassificationDataset(test_df, tokenizer, int(cfg["max_length"]), has_labels=True)

    # ---- 平衡采样（小样本 / 长尾场景推荐开启） ----
    if bool(cfg.get("balance_sampling", False)):
        train_labels = split.train_df["label_id"].astype(int).values
        class_counts = np.bincount(train_labels)
        sample_weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights).float(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg["batch_size"]),
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaSupConClassifier(
        model_name=cfg["model_name"],
        num_labels=len(label_names),
        dropout=float(cfg["dropout"]),
        contrastive_dim=int(cfg["contrastive_dim"]),
        pooling=str(cfg["pooling"]),
        use_location_priority_branch=bool(cfg.get("use_location_priority_branch", False)),
        location_priority_embedding_dim=int(cfg.get("location_priority_embedding_dim", 8)),
    ).to(device)

    cls_loss_type = str(cfg.get("classification_loss", "cross_entropy"))
    if cls_loss_type == "focal":
        ce_loss_fn = FocalLoss(
            gamma=float(cfg.get("focal_gamma", 2.0)),
            alpha=cfg.get("focal_alpha", None),
        )
    else:
        ce_loss_fn = nn.CrossEntropyLoss()
    contrastive_loss_type = str(cfg.get("contrastive_loss", "supervised_contrastive"))
    if contrastive_loss_type == "proto_supervised_contrastive":
        supcon_loss_fn = ProtoSupConLoss(
            num_classes=len(label_names),
            feat_dim=int(cfg["contrastive_dim"]),
            temperature=float(cfg["temperature"]),
            alpha=float(cfg.get("proto_alpha", 0.5)),
        )
    else:
        supcon_loss_fn = SupConLoss(temperature=float(cfg["temperature"]))
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    gradient_accumulation_steps = int(cfg["gradient_accumulation_steps"])
    update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_training_steps = update_steps_per_epoch * int(cfg["epochs"])
    warmup_steps = int(total_training_steps * float(cfg["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_metric_name = str(cfg["main_metric"])
    best_metric = -1.0
    best_epoch = -1
    patience = int(cfg["early_stopping_patience"])
    bad_epochs = 0
    history = []
    lambda_contrast = float(cfg["lambda_contrast"])

    use_fgm = bool(cfg.get("use_fgm", False))
    fgm_epsilon = float(cfg.get("fgm_epsilon", 0.5))
    fgm = FGM(model, epsilon=fgm_epsilon) if use_fgm else None

    print(f"device={device}")
    print(f"split_method={split.split_method}, train={len(split.train_df)}, valid={len(split.valid_df)}, test={len(test_df)}")
    print(f"labels={label_names}")
    print(f"classification_loss={cls_loss_type}, contrastive_loss={contrastive_loss_type}")
    print(f"temperature={cfg['temperature']}, lambda_contrast={lambda_contrast}")
    if contrastive_loss_type == "proto_supervised_contrastive":
        print(f"proto_alpha={cfg.get('proto_alpha', 0.5)}")
    if bool(cfg.get("balance_sampling", False)):
        print("balance_sampling=True")
    if use_fgm:
        print(f"fgm=True, fgm_epsilon={fgm_epsilon}")

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_cls_loss = 0.0
        train_contrast_loss = 0.0
        train_examples = 0
        train_valid_anchors = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"]
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                location_priority_ids=batch.get("location_priority_ids"),
            )
            cls_loss = ce_loss_fn(outputs["logits"], labels)
            contrast_loss, valid_anchors = supcon_loss_fn(outputs["contrastive_embedding"], labels)
            loss = cls_loss + lambda_contrast * contrast_loss
            (loss / gradient_accumulation_steps).backward()

            # FGM 对抗训练：在 embedding 上沿梯度方向加扰动后重算 loss
            if use_fgm:
                fgm.attack()
                outputs_adv = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    location_priority_ids=batch.get("location_priority_ids"),
                )
                cls_loss_adv = ce_loss_fn(outputs_adv["logits"], labels)
                contrast_loss_adv, _ = supcon_loss_fn(outputs_adv["contrastive_embedding"], labels)
                loss_adv = cls_loss_adv + lambda_contrast * contrast_loss_adv
                (loss_adv / gradient_accumulation_steps).backward()
                fgm.restore()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            train_loss += float(loss.item()) * batch_size
            train_cls_loss += float(cls_loss.item()) * batch_size
            train_contrast_loss += float(contrast_loss.item()) * batch_size
            train_examples += batch_size
            train_valid_anchors += valid_anchors
            progress.set_postfix(loss=f"{train_loss / max(train_examples, 1):.4f}")

        valid_eval = run_eval(model, valid_loader, ce_loss_fn, supcon_loss_fn, device, lambda_contrast)
        valid_metrics = compute_metrics(valid_eval.pop("y_true"), valid_eval.pop("y_pred"), label_names)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_examples, 1),
            "train_classification_loss": train_cls_loss / max(train_examples, 1),
            "train_contrastive_loss": train_contrast_loss / max(train_examples, 1),
            "train_valid_contrastive_anchors": int(train_valid_anchors),
            "valid_eval": valid_eval,
            "valid_metrics": valid_metrics,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_record)

        current_metric = float(valid_metrics[best_metric_name])
        print(
            f"epoch={epoch} train_loss={epoch_record['train_loss']:.4f} "
            f"valid_loss={valid_eval['loss']:.4f} valid_{best_metric_name}={current_metric:.4f} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f}"
        )

        should_stop = False
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            bad_epochs = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "label_names": label_names,
                "best_epoch": best_epoch,
                "best_metric_name": best_metric_name,
                "best_metric": best_metric,
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch}")
                should_stop = True

        save_json({"history": history, "best_epoch": best_epoch, "best_metric": best_metric}, logs_dir / "training_history.json")
        if should_stop:
            break

    best_checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_eval = run_eval(model, test_loader, ce_loss_fn, supcon_loss_fn, device, lambda_contrast)
    test_y_true = test_eval.pop("y_true")
    test_y_pred = test_eval.pop("y_pred")
    test_metrics = compute_metrics(test_y_true, test_y_pred, label_names)
    test_report = {
        "best_epoch": best_epoch,
        "best_valid_metric_name": best_metric_name,
        "best_valid_metric": best_metric,
        "test_eval": test_eval,
        "test_metrics": test_metrics,
    }
    save_json(test_report, logs_dir / "test_metrics.json")

    pred_df = test_df.copy()
    pred_df["pred_label_id"] = test_y_pred
    pred_df["pred_label"] = [label_names[i] for i in test_y_pred]
    pred_df.to_excel(prediction_dir / "roberta_supcon_test_predictions.xlsx", index=False)
    pred_df.to_csv(prediction_dir / "roberta_supcon_test_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"best_epoch={best_epoch}, best_valid_{best_metric_name}={best_metric:.4f}")
    print(f"test_accuracy={test_metrics['accuracy']:.4f}, test_macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"saved outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/roberta_supcon_cls.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
