"""
LLM few-shot 快速验证：把全部训练数据塞进 prompt 当记忆，在验证集和测试集上评估。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    apply_configured_text_features,
    get_label_names,
    group_overlap,
    read_excel_dataset,
    split_train_valid,
    summarize_dataframe,
)
from src.utils import compute_metrics, load_config, save_json

API_URL = "http://10.130.71.10:30571/v1/chat/completions"
MODEL_NAME = "qwen3.6-27b"  # API 自动选择
MAX_RETRIES = 3
RETRY_DELAY = 2


def build_few_shot_prompt(train_texts: list[str], train_labels: list[str], test_text: str) -> str:
    """把所有训练样本当 few-shot 示例，最后放待分类文本。"""
    lines = [
        "你是一个文本分类助手。请根据以下示例，将最后一条文本分类为以下三类之一：安全、质量、其他。",
        "只输出类别名称，不要输出任何其他内容。",
        "",
        "示例：",
    ]

    for i, (text, label) in enumerate(zip(train_texts, train_labels), 1):
        lines.append(f"{i}. 文本：{text}")
        lines.append(f"   类别：{label}")

    lines.append("")
    lines.append(f"请分类以下文本（只输出\"安全\"、\"质量\"或\"其他\"）：")
    lines.append(f"文本：{test_text}")
    lines.append("类别：")

    return "\n".join(lines)


def call_api(prompt: str, temperature: float = 0.0) -> str | None:
    """调用 LLM API，返回预测类别。"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, json=payload, timeout=120)
            if resp.status_code == 200:
                result = resp.json()
                msg = result["choices"][0]["message"]
                # 推理模型可能把最终答案放在 content 或 reasoning_content 里
                answer = (msg.get("content") or msg.get("reasoning_content") or "").strip()
                # 归一化输出：匹配第一个出现的类别名
                for label in ["安全", "质量", "其他"]:
                    if label in answer:
                        return label
                # 都没匹配到，返回原始输出的最后一行
                return answer.split("\n")[-1].strip() if answer else None
            else:
                print(f"  API error (status={resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            print(f"  Request failed (attempt {attempt + 1}): {e}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)

    return None


def run_llm_eval(train_texts: list[str], train_labels: list[str],
                  eval_texts: list[str], eval_labels: list[str],
                  label_names: list[str], desc: str = "eval") -> dict:
    """在给定数据集上运行 LLM 评估。"""
    y_true = []
    y_pred = []
    errors = []

    total = len(eval_texts)
    print(f"\n{'='*50}")
    print(f"{desc}: {total} samples")
    print(f"{'='*50}")

    for i, (text, label) in enumerate(zip(eval_texts, eval_labels)):
        prompt = build_few_shot_prompt(train_texts, train_labels, text)
        pred_text = call_api(prompt)

        true_name = label_names[label]
        if pred_text is None:
            pred_name = "__api_error__"
        else:
            pred_name = pred_text

        y_true.append(label)
        if pred_name in label_names:
            y_pred.append(label_names.index(pred_name))
        else:
            # 无法映射 → 归为"其他"
            y_pred.append(label_names.index("其他"))
            if pred_text is not None:
                print(f"  [{i+1}/{total}] unmapped output '{pred_text}' → 其他 (true={true_name})")

        if y_true[-1] != y_pred[-1]:
            errors.append({
                "idx": i,
                "text": text[:120],
                "true": true_name,
                "pred": pred_name,
            })

        if (i + 1) % 10 == 0:
            acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            print(f"  [{i+1}/{total}] running accuracy={acc:.4f}")

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1)
    print(f"  final accuracy={acc:.4f}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": acc,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/roberta_supcon_cls.yaml")
    parser.add_argument("--skip-valid", action="store_true", help="跳过验证集，只跑测试集")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 加载数据
    train_df = read_excel_dataset(cfg["train_path"])
    test_df = read_excel_dataset(cfg["test_path"])
    train_df = apply_configured_text_features(train_df, cfg)
    test_df = apply_configured_text_features(test_df, cfg)
    label_names = get_label_names(train_df)

    # 划分训练/验证
    split = split_train_valid(
        train_df,
        validation_split=float(cfg["validation_split"]),
        seed=int(cfg["seed"]),
        strategy=str(cfg["split_strategy"]),
    )

    print(f"Train: {len(split.train_df)}, Valid: {len(split.valid_df)}, Test: {len(test_df)}")
    print(f"Labels: {label_names}")

    train_texts = split.train_df["text"].astype(str).tolist()
    train_labels = split.train_df["label"].astype(str).tolist()

    results = {}

    # 验证集
    if not args.skip_valid:
        valid_texts = split.valid_df["text"].astype(str).tolist()
        valid_labels = split.valid_df["label_id"].astype(int).tolist()
        valid_result = run_llm_eval(
            train_texts, train_labels,
            valid_texts, valid_labels, label_names, desc="Valid",
        )
        valid_metrics = compute_metrics(
            valid_result.pop("y_true"), valid_result.pop("y_pred"), label_names,
        )
        results["valid"] = {**valid_result, "metrics": valid_metrics}

    # 测试集
    test_texts = test_df["text"].astype(str).tolist()
    test_labels = test_df["label_id"].astype(int).tolist()
    test_result = run_llm_eval(
        train_texts, train_labels,
        test_texts, test_labels, label_names, desc="Test",
    )
    test_metrics = compute_metrics(
        test_result.pop("y_true"), test_result.pop("y_pred"), label_names,
    )
    results["test"] = {**test_result, "metrics": test_metrics}

    # 输出
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    for split_name in ["valid", "test"]:
        if split_name not in results:
            continue
        m = results[split_name]["metrics"]
        print(f"\n--- {split_name} ---")
        print(f"Accuracy:  {m['accuracy']:.4f}")
        print(f"Macro-F1:  {m['macro_f1']:.4f}")
        print(f"Weighted-F1: {m['weighted_f1']:.4f}")
        print("Per-class:")
        for name in label_names:
            c = m["classification_report"][name]
            print(f"  {name}: P={c['precision']:.4f} R={c['recall']:.4f} F1={c['f1-score']:.4f} (support={c['support']})")

    # 保存
    output_dir = Path(cfg.get("output_dir", "outputs/llm_test"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, output_dir / "llm_test_results.json")
    print(f"\nSaved to {output_dir / 'llm_test_results.json'}")


if __name__ == "__main__":
    main()
