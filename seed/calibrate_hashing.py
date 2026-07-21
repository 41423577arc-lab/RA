import csv
from pathlib import Path

import numpy as np

from app.services.text_vectorizer import vectorize_text, vectorize_texts


def main() -> None:
    csv_path = Path(__file__).with_name("internal_projects.csv")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    documents = [
        f"{row['customer_name']} {row['project_name']} {row['description']}" for row in rows
    ]
    document_vectors = np.asarray(vectorize_texts(documents))
    for query in ("新能源 储能", "比亚迪股份有限公司", "火星矿业 深海养殖"):
        scores = document_vectors @ np.asarray(vectorize_text(query))
        indices = np.argsort(-scores)[:5]
        values = [(rows[index]["project_id"], round(float(scores[index]), 4)) for index in indices]
        print(f"{query}: {values}")


if __name__ == "__main__":
    main()
