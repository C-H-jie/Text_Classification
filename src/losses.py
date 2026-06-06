from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss：自动降低已分对的简单样本的 loss 权重，让模型聚焦于难样本。

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=0 时退化为标准 CrossEntropyLoss（若 alpha 为 None）。
    推荐起点：gamma=2.0, alpha=None（无需额外调参）。

    适用场景：
    - 类别边界模糊（如本任务中"安全"与"其他"频繁互混）
    - 小样本下模型在简单样本上快速过拟合，难样本始终分不对
    """

    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] 未归一化的分类 logits
            targets: [B] 整数标签
        Returns:
            scalar loss
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # p_t: 模型对正确类别的预测概率

        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]
            ce_loss = alpha_t * ce_loss

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class SupConLoss(nn.Module):
    """原始监督对比损失：实例-实例对比。

    同类样本拉近，异类样本推远。batch 内每个样本以同类其余样本为正、
    异类样本为负，计算多正样本的 InfoNCE 损失。
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
        features = F.normalize(features, p=2, dim=1)
        labels = labels.view(-1, 1)
        batch_size = features.size(0)

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = torch.eq(labels, labels.T) & ~self_mask
        valid_anchor_mask = positive_mask.sum(dim=1) > 0
        valid_anchor_count = int(valid_anchor_mask.sum().item())

        if valid_anchor_count == 0:
            return features.new_tensor(0.0), 0

        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
        loss = -mean_log_prob_pos[valid_anchor_mask].mean()
        return loss, valid_anchor_count


class ProtoSupConLoss(nn.Module):
    """原型监督对比损失：实例-实例 + 实例-原型 双支路。

    支路 1（实例-实例）：同类样本拉近 + 异类推远，与 SupConLoss 相同。
    支路 2（实例-原型）：每个类别维护一个可学习原型向量，实例与原型做交叉熵，
    提供稳定的全局分类信号。

    双支路通过 alpha 加权混合。alpha 越大，越依赖原型支路。
    小样本/小 batch 场景推荐 alpha >= 0.5，利用原型弥补 batch 内正样本不足的问题。
    """

    def __init__(self, num_classes: int, feat_dim: int, temperature: float = 0.1, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        # 可学习原型向量：每个类别一个，初始随机 + L2 归一化
        self.prototypes = nn.Parameter(F.normalize(torch.randn(num_classes, feat_dim), dim=1))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Args:
            features: 已经 L2-normalized 的对比嵌入 [B, D]
            labels: 标签 [B]
        Returns:
            (loss, valid_anchor_count)
        """
        # features 已经过 projection_head 的 F.normalize，这里再归一化一次无害
        features = F.normalize(features, p=2, dim=1)

        # 确保原型向量与输入在同一设备（训练时 GPU，保存/加载后自动对齐）
        if self.prototypes.device != features.device:
            self.prototypes = nn.Parameter(self.prototypes.data.to(features.device))
        labels_v = labels.view(-1, 1)
        batch_size = features.size(0)

        # ---- 支路 1: 实例-实例对比（与 SupConLoss 一致） ----
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = torch.eq(labels_v, labels_v.T) & ~self_mask
        valid_anchor = positive_mask.sum(dim=1) > 0
        n_valid = int(valid_anchor.sum().item())

        if n_valid == 0:
            loss_inst = features.new_tensor(0.0)
        else:
            exp_logits = torch.exp(logits) * (~self_mask).float()
            log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
            mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
            loss_inst = -mean_log_prob_pos[valid_anchor].mean()

        # ---- 支路 2: 实例-原型对比 ----
        proto_logits = torch.matmul(features, self.prototypes.T) / self.temperature
        loss_proto = F.cross_entropy(proto_logits, labels)

        # ---- 加权混合 ----
        loss = (1.0 - self.alpha) * loss_inst + self.alpha * loss_proto

        return loss, n_valid
