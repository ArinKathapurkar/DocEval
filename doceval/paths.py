"""paths.py — Canonical filesystem layout."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
QASPER = {
    "train": DATA / "qasper-train-v0.3.json",
    "dev": DATA / "qasper-dev-v0.3.json",
}

ARTIFACTS = ROOT / "artifacts"
CHUNKS = ARTIFACTS / "chunks.parquet"
QUESTIONS = ARTIFACTS / "questions.parquet"
EMBEDDINGS = ARTIFACTS / "chunk_emb.npy"
RESULTS = ARTIFACTS / "results"
GOLDEN = ROOT / "golden"          # hand-labeled judge-validation set (version controlled)


def ensure_dirs() -> None:
    for d in (DATA, ARTIFACTS, RESULTS, GOLDEN):
        d.mkdir(parents=True, exist_ok=True)
