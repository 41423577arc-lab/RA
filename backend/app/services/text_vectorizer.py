from functools import lru_cache

from sklearn.feature_extraction.text import HashingVectorizer


HASHING_DIMENSIONS = 512


@lru_cache(maxsize=1)
def get_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        n_features=HASHING_DIMENSIONS,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
    )


def vectorize_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    matrix = get_vectorizer().transform(texts)
    return matrix.toarray().astype("float32").tolist()


def vectorize_text(text: str) -> list[float]:
    return vectorize_texts([text])[0]

