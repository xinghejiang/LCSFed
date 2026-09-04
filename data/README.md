# Data

Bundled **ML_1M** split under `ML_1M/`.

## Layout

```
ML_1M/
  train.csv  vali.csv  test.csv
  uid_info.txt  iid_info.txt
  uid/llm_1024.npy   # required for training; generate if missing
  iid/llm_1024.npy
```

## Files

| File | Description |
|------|-------------|
| `train/vali/test.csv` | Interactions (`uid`, `iid`, `rating`) |
| `uid_info.txt` / `iid_info.txt` | Side info, TSV `id<TAB>text`, row order = id `0 … N-1` |
| `uid/llm_1024.npy` | User LLM embeddings (6039×1024) |
| `iid/llm_1024.npy` | Item LLM embeddings (3550×1024) |

## Generate embeddings

If `llm_1024.npy` is not present, run from the project root:

```bash
export OPENAI_API_KEY=your_key
python scripts/text2emb.py --input data/ML_1M/uid_info.txt --out-dir data/ML_1M/uid --dim 1024
python scripts/text2emb.py --input data/ML_1M/iid_info.txt --out-dir data/ML_1M/iid --dim 1024
```

MovieLens 1M: [GroupLens terms](https://grouplens.org/datasets/movielens/1m/).
