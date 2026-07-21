import csv
from pathlib import Path

import numpy as np

from app.services.text_vectorizer import HASHING_DIMENSIONS, vectorize_text, vectorize_texts


ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.18


def load_project_documents() -> tuple[list[dict[str, str]], np.ndarray]:
    with (ROOT / "seed/internal_projects.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    documents = [
        f"{row['customer_name']} {row['project_name']} {row['description']}" for row in rows
    ]
    return rows, np.asarray(vectorize_texts(documents))


def test_hash_vectors_are_fixed_size_normalized_and_deterministic() -> None:
    first = np.asarray(vectorize_text("新能源 储能"))
    second = np.asarray(vectorize_text("新能源 储能"))

    assert first.shape == (HASHING_DIMENSIONS,)
    assert np.isclose(np.linalg.norm(first), 1.0)
    assert np.array_equal(first, second)


def test_calibrated_threshold_keeps_relevant_and_rejects_irrelevant_query() -> None:
    rows, documents = load_project_documents()
    relevant_scores = documents @ np.asarray(vectorize_text("新能源 储能"))
    irrelevant_scores = documents @ np.asarray(vectorize_text("火星矿业 深海养殖"))
    p002_index = next(index for index, row in enumerate(rows) if row["project_id"] == "P002")

    assert relevant_scores[p002_index] >= THRESHOLD
    assert irrelevant_scores.max() < THRESHOLD
