"""
Baseline: 结巴分词 + TF-IDF + 余弦相似度。
高频跨类词自动降权，低频特征词提权，比纯词袋命中率靠谱得多。
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import jieba
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_label_names, read_excel_dataset, split_train_valid
from src.utils import compute_metrics, load_config


def tokenize(text: str) -> list[str]:
    return [w.strip() for w in jieba.cut(text) if len(w.strip()) > 1]


def build_tfidf_class_vectors(texts: list[str], labels: list[int], num_classes: int):
    """
    为每个类别构建 TF-IDF 向量。
    返回: (vocab, idf, class_vectors)
      vocab: {word: idx}
      idf: np.array [V]
      class_vectors: np.array [C, V]  每行是一个类别的 TF-IDF 向量
    """
    # 每类文档合并为一个大文本
    class_docs = [[] for _ in range(num_classes)]
    for text, label in zip(texts, labels):
        class_docs[label].extend(tokenize(text))

    # 每个类的词频
    class_tfs = [Counter(doc) for doc in class_docs]

    # 全局词汇表
    vocab = {}
    for bag in class_tfs:
        for w in bag:
            if w not in vocab:
                vocab[w] = len(vocab)
    V = len(vocab)

    # 每类 TF 向量
    tf_vectors = np.zeros((num_classes, V))
    for c, bag in enumerate(class_tfs):
        total = sum(bag.values()) or 1
        for w, cnt in bag.items():
            tf_vectors[c, vocab[w]] = cnt / total

    # IDF: log(类数 / 出现该词的类数) + 1
    df = np.zeros(V)
    for c, bag in enumerate(class_tfs):
        for w in bag:
            df[vocab[w]] += 1
    idf = np.log((num_classes + 1) / (df + 1)) + 1

    # TF-IDF 向量
    tfidf_vectors = tf_vectors * idf

    # L2 归一化
    norms = np.linalg.norm(tfidf_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf_vectors = tfidf_vectors / norms

    return vocab, idf, tfidf_vectors


def classify_by_tfidf(text: str, vocab: dict, idf: np.ndarray,
                      class_vectors: np.ndarray) -> int:
    """测试文本 → TF-IDF 向量 → 找余弦相似度最高的类别。"""
    words = tokenize(text)
    if not words:
        return 0

    V = len(vocab)
    tf = np.zeros(V)
    word_counts = Counter(words)
    total = sum(word_counts.values()) or 1
    for w, cnt in word_counts.items():
        if w in vocab:
            tf[vocab[w]] = cnt / total

    query_vec = tf * idf
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    sims = class_vectors @ query_vec
    return int(np.argmax(sims))


def main():
    cfg = load_config("configs/roberta_supcon_cls.yaml")
    train_df = read_excel_dataset(cfg["train_path"])
    test_df = read_excel_dataset(cfg["test_path"])
    label_names = get_label_names(train_df)

    split = split_train_valid(train_df, validation_split=0.2, seed=42,
                              strategy=cfg["split_strategy"])

    print(f"Train: {len(split.train_df)}, Valid: {len(split.valid_df)}, Test: {len(test_df)}")
    print(f"Labels: {label_names}")

    train_texts = split.train_df["text"].astype(str).tolist()
    train_labels = split.train_df["label_id"].astype(int).tolist()

    vocab, idf, class_vectors = build_tfidf_class_vectors(
        train_texts, train_labels, len(label_names))
    print(f"Vocabulary size: {len(vocab)}")

    # 每类最高 TF-IDF 词 → 看特征词质量
    for c, name in enumerate(label_names):
        top_idx = np.argsort(class_vectors[c])[::-1][:15]
        idx_to_word = {v: k for k, v in vocab.items()}
        top_words = [(idx_to_word[i], class_vectors[c, i]) for i in top_idx]
        print(f"  [{name}] top TF-IDF: {[(w, f'{s:.3f}') for w, s in top_words[:10]]}")

    for desc, df in [("Valid", split.valid_df), ("Test", test_df)]:
        texts = df["text"].astype(str).tolist()
        labels = df["label_id"].astype(int).tolist()

        y_pred = [classify_by_tfidf(t, vocab, idf, class_vectors) for t in texts]
        metrics = compute_metrics(labels, y_pred, label_names)

        print(f"\n--- {desc} ---")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
        for name in label_names:
            c = metrics["classification_report"][name]
            print(f"  {name}: P={c['precision']:.4f} R={c['recall']:.4f} F1={c['f1-score']:.4f}")
        print("Confusion:")
        for row in metrics["confusion_matrix"]:
            print(f"  {row}")


if __name__ == "__main__":
    main()
