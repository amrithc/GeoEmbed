"""Retrieval evaluation metrics."""

import numpy as np


def precision_at_k(retrieved_labels: list[str], query_label: str, k: int) -> float:
    """Proportion of top-K results with the same label as the query."""
    top_k = retrieved_labels[:k]
    return sum(1 for l in top_k if l == query_label) / k


def mean_precision_at_k(
    all_retrieved_labels: list[list[str]],
    all_query_labels: list[str],
    k: int,
) -> float:
    """Mean precision@K across all queries."""
    scores = [
        precision_at_k(retrieved, query, k)
        for retrieved, query in zip(all_retrieved_labels, all_query_labels)
    ]
    return float(np.mean(scores))
