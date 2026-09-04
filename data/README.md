# Data

Bundled **ML_1M** split under `ML_1M/`.

## Layout

```
ML_1M/
  train.csv  vali.csv  test.csv
  uid_info.txt  iid_info.txt
  uid/llm_1024.npy
  iid/llm_1024.npy
```

## Files

| File | Description |
|------|-------------|
| `train/vali/test.csv` | Interactions (`uid`, `iid`, `rating`) |
| `uid_info.txt` / `iid_info.txt` | Side info, TSV `id<TAB>text`, row order = id `0 … N-1` |
| `uid/llm_1024.npy` | Bundled user semantic embeddings (6039×1024, float32) |
| `iid/llm_1024.npy` | Bundled item semantic embeddings (3550×1024, float32) |

## Semantic embeddings

The bundled matrices use the first 1024 dimensions of Qwen3-Embedding-4B outputs and apply row-wise L2 normalization.

MovieLens 1M: [GroupLens terms](https://grouplens.org/datasets/movielens/1m/).
