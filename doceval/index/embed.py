"""
embed.py — Embed every chunk once with a local sentence-transformers model.

Run after ingest:
    python -m doceval.index.embed [--model BAAI/bge-small-en-v1.5]

Produces:
    artifacts/chunk_emb.npy   (N, D) float32, L2-normalized, row-aligned to chunks.parquet

Runs locally on MPS, so the retrieval side of this project costs nothing to
iterate on. API spend is reserved for generation and judging, where it actually
buys something.
"""

import argparse
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from doceval import paths

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# bge models expect an instruction prefix on the QUERY side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def pick_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(name, device=pick_device())


def encode_chunks(model: SentenceTransformer, texts: list[str], batch: int = 128) -> np.ndarray:
    return model.encode(
        texts, batch_size=batch, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )


def encode_queries(model: SentenceTransformer, queries: list[str], batch: int = 128) -> np.ndarray:
    return model.encode(
        [QUERY_PREFIX + q for q in queries], batch_size=batch,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    paths.ensure_dirs()
    chunks = pd.read_parquet(paths.CHUNKS)
    model = load_model(args.model)
    print(f"{args.model} on {model.device} | {len(chunks):,} chunks")

    t0 = time.perf_counter()
    emb = encode_chunks(model, chunks.text.tolist(), args.batch)
    dt = time.perf_counter() - t0

    np.save(paths.EMBEDDINGS, emb)
    print(f"\n{emb.shape} float32 ({emb.nbytes / 1e6:.1f} MB) in {dt:.0f}s "
          f"({len(chunks) / dt:.0f} chunks/s)")
