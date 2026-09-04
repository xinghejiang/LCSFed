#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault('OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'YES')
os.environ.setdefault('no_proxy', '*')

API_FULL_DIM = 1536

API_KEY = 'Your-API-Key'
BASE_URL = 'https://api.openai.com/v1'


def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x if norm == 0 else x / norm
    norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
    return np.where(norm == 0, x, x / norm)


def cut_emb(x, cut_len):
    return normalize_l2(x[:cut_len])


def strip_profile_prefix(text: str) -> str:
    for prefix in ('Movie profile:', 'User profile:'):
        if text.startswith(prefix):
            return text[len(prefix):].lstrip()
    return text


def read_texts(path: Path, limit_rows: int | None = None) -> list[str]:
    df = pd.read_csv(
        path, sep='\t', header=None, names=['id', 'text'],
        encoding='utf-8', nrows=limit_rows,
    )
    if df.shape[1] != 2:
        raise ValueError(f'Expected 2 columns in {path}')
    texts = (
        df['text'].astype(str)
        .map(strip_profile_prefix)
        .str.lower()
        .tolist()
    )
    if not texts:
        raise ValueError(f'No valid rows in {path}')
    return texts


def get_embedding(text, *, max_tokens, model_name, dimension, api_key, base_url):
    import tiktoken
    from openai import OpenAI

    encoder = tiktoken.encoding_for_model(model_name)
    clean_text = encoder.decode(encoder.encode(text)[:max_tokens])

    client_params = {'api_key': api_key}
    if base_url:
        client_params['base_url'] = base_url

    response = OpenAI(**client_params).embeddings.create(
        model=model_name,
        input=clean_text,
        dimensions=dimension,
    )
    return response.data[0].embedding


def get_embedding_with_retry(text, *, max_retries, max_tokens, model_name, dimension, api_key, base_url):
    for attempt in range(max_retries):
        try:
            return get_embedding(
                text,
                max_tokens=max_tokens,
                model_name=model_name,
                dimension=dimension,
                api_key=api_key,
                base_url=base_url,
            )
        except Exception as e:
            wait = (2 ** attempt) * 0.5
            print(f'Retry {attempt + 1}/{max_retries}: {e} (wait {wait:.1f}s)', file=sys.stderr)
            time.sleep(wait)
    print(f'Failed, zero vector: {text[:50]}...', file=sys.stderr)
    return np.zeros(dimension)


def embed_texts(texts, *, max_workers, max_retries, max_tokens, model_name, api_key, base_url):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                get_embedding_with_retry, t,
                max_retries=max_retries,
                max_tokens=max_tokens,
                model_name=model_name,
                dimension=API_FULL_DIM,
                api_key=api_key,
                base_url=base_url,
            )
            for t in texts
        ]
        for fut in tqdm(futures, total=len(texts), desc='Embeddings'):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f'Worker failed: {e}', file=sys.stderr)
                results.append(np.zeros(API_FULL_DIM))
    return results


def to_output_embeddings(raw_embeddings, out_dim: int) -> np.ndarray:
    if out_dim == API_FULL_DIM:
        return np.stack(raw_embeddings).astype(np.float32)
    return np.array([cut_emb(x, out_dim) for x in raw_embeddings], dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description='Text -> LLM embedding')
    p.add_argument('--input', required=True, help='Input TSV: id<TAB>text')
    p.add_argument('--out-dir', required=True, help='Output directory')
    p.add_argument('--dim', type=int, default=1024)
    p.add_argument('--model', default='text-embedding-3-large')
    p.add_argument('--max-tokens', type=int, default=8190)
    p.add_argument('--max-workers', type=int, default=5)
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--limit-rows', type=int, default=None)
    args = p.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = read_texts(input_path, limit_rows=args.limit_rows)

    raw = embed_texts(
        texts,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        model_name=args.model,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    embeddings = to_output_embeddings(raw, args.dim)

    out_path = out_dir / f'llm_{args.dim}.npy'
    np.save(out_path, embeddings)

    norms = np.linalg.norm(embeddings, axis=1)
    print(f'Saved {out_path}  shape={embeddings.shape}  dtype={embeddings.dtype}')
    print(f'L2 norm  mean={norms.mean():.4f}  range=[{embeddings.min():.4f}, {embeddings.max():.4f}]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
