import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def compute_metrics(y_true, y_pred, label_names: list[str]) -> dict:
    labels = list(range(len(label_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class FGM:
    """Fast Gradient Method 对抗训练。

    在 embedding 层沿梯度方向加扰动，让模型在"最坏情况"下仍能正确分类。
    等价于免费的虚拟数据增强，特别适合小样本场景。

    用法：
        fgm = FGM(model, epsilon=0.5)
        loss.backward()           # 正常反向传播
        fgm.attack()              # 在 embedding 上加扰动
        loss_adv = model(...)     # 用扰动后的 embedding 再算一次 loss
        loss_adv.backward()       # 对抗梯度
        fgm.restore()             # 恢复原始 embedding
        optimizer.step()          # 用叠加了对抗梯度的方向更新参数
    """

    def __init__(self, model: torch.nn.Module, epsilon: float = 0.5, emb_name: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self._backup = {}

    def attack(self):
        """沿梯度方向给 embedding 加扰动。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self._backup[name] = param.data.clone()
                norm = param.grad.norm(p=2)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        """恢复原始 embedding。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self._backup:
                    param.data = self._backup[name]
        self._backup.clear()
