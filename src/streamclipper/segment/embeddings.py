"""Sentence embeddings for the semantic-drift boundary signal.

Vectors are sparse maps of ``index -> weight``, L2-normalised. That one
representation serves both backends: a TF-IDF vector is naturally sparse, and
a 384-dimensional dense embedding is small enough that storing it as a map
costs little and keeps every downstream operation (cosine, window mean)
written once.

Two backends:

* ``SentenceTransformerEmbedder`` -- the default. Real semantic vectors, GPU
  when one is available, and it catches paraphrase.
* ``TfidfEmbedder`` -- pure-Python fallback with no third-party dependencies.
  Lexical only, but it keeps the pipeline and the test suite runnable without
  torch.

The TF-IDF backend fits its vocabulary on the batch it is asked to encode,
which is the right corpus: boundaries are relative to *this* stream, so the
word that is rare on this stream is the informative one.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from typing import Mapping, Protocol, Sequence

from ..config import EmbeddingsConfig
from ..logging_setup import get_logger

log = get_logger(__name__)

Vector = Mapping[int, float]
"""A sparse, L2-normalised vector."""

_TOKEN = re.compile(r"[a-z0-9']+")

# Length at which a token is also indexed by its prefix. Crude stemming, but
# it makes "polling"/"pollsters" and "cooking"/"cooks" share a feature, which
# matters a lot when each sentence only contributes a handful of terms.
_STEM_LENGTH = 5

# Words that appear everywhere and carry no topical signal. Kept small on
# purpose -- IDF handles most of the work.
_STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have he her his i if in
    is it its just like me my not of on or our she so than that the their them then
    there they this to was we were what when which who will with you your yeah
    okay right gonna wanna dont im its really actually literally""".split()
)


class Embedder(Protocol):
    """Anything that turns sentences into sparse unit-length vectors."""

    def encode(self, texts: Sequence[str]) -> list[Vector]:
        ...


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


def features(text: str) -> list[str]:
    """Indexable features: content words plus their stems."""
    tokens = tokenize(text)
    return tokens + [t[:_STEM_LENGTH] for t in tokens if len(t) > _STEM_LENGTH]


def normalise(vector: dict[int, float]) -> dict[int, float]:
    norm = math.sqrt(sum(v * v for v in vector.values()))
    if norm <= 1e-12:
        return vector
    return {k: v / norm for k, v in vector.items()}


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity. Iterates the smaller side."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def mean(vectors: Sequence[Vector]) -> dict[int, float]:
    """Centroid of a block of vectors."""
    if not vectors:
        return {}
    accumulator: dict[int, float] = {}
    for vector in vectors:
        for key, value in vector.items():
            accumulator[key] = accumulator.get(key, 0.0) + value
    count = float(len(vectors))
    return {k: v / count for k, v in accumulator.items()}


class TfidfEmbedder:
    """Exact TF-IDF over the batch's own vocabulary.

    An earlier version hashed features into a fixed-width dense vector. That
    is cheaper, but livestream sentences carry only a handful of content words
    each, and at that sparsity hash collisions produced more similarity than
    the text did -- boundaries disappeared into the noise. An exact vocabulary
    costs a dict per sentence and is worth it.
    """

    def encode(self, texts: Sequence[str]) -> list[Vector]:
        docs = [features(text) for text in texts]
        n = len(docs)
        if n == 0:
            return []

        document_freq: Counter[str] = Counter()
        for tokens in docs:
            document_freq.update(set(tokens))
        vocabulary = {term: index for index, term in enumerate(sorted(document_freq))}

        vectors: list[Vector] = []
        for tokens in docs:
            vector: dict[int, float] = {}
            if tokens:
                counts = Counter(tokens)
                most_common = max(counts.values())
                for term, count in counts.items():
                    tf = 0.5 + 0.5 * count / most_common      # augmented tf
                    idf = math.log((n + 1) / (document_freq[term] + 1)) + 1.0
                    vector[vocabulary[term]] = tf * idf
            vectors.append(normalise(vector))
        return vectors


class SentenceTransformerEmbedder:
    """sentence-transformers backend, loaded lazily and shared across threads."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        self.config = config
        self._model = None
        self._lock = threading.Lock()

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            device = self.config.device
            if device == "auto":
                try:
                    import torch  # noqa: PLC0415

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            log.info(
                "embeddings.loading",
                extra={"model": self.config.model, "device": device},
            )
            self._model = SentenceTransformer(self.config.model, device=device)
            return self._model

    def encode(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        model = self._load()
        rows = model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [
            {index: float(value) for index, value in enumerate(row) if value}
            for row in rows
        ]


def get_embedder(config: EmbeddingsConfig) -> Embedder:
    """Build the configured embedder, falling back when the extra is absent.

    An explicit ``backend: tfidf`` is honoured; the default backend degrades
    with a warning rather than failing the run, because semantic boundaries
    are one of two signals and the LLM pass can carry the segmentation alone.
    """
    if config.backend == "tfidf":
        return TfidfEmbedder()

    try:
        import sentence_transformers  # noqa: F401,PLC0415

        return SentenceTransformerEmbedder(config)
    except ImportError:
        log.warning(
            "embeddings.fallback",
            extra={
                "requested": config.backend,
                "using": "tfidf",
                "hint": 'pip install "stream-clipper[embeddings]" for semantic vectors',
            },
        )
        return TfidfEmbedder()
