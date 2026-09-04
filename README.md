# LCSFed

This folder provides the training code for LCSFed under cold-start federated recommendation.

## Layout

```
LCSFed/
├── main.py
├── model/          # client/server model and federated engine
├── utils/          # metrics, logging, MovieLens-style data loader
├── data/           # processed ML-1M data and semantic embeddings
├── requirements.txt
└── README.md
```

## Requirements

Python 3.8+. Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU builds of PyTorch, follow [pytorch.org](https://pytorch.org/get-started/locally/) (CUDA wheels use their extra index URL).

## Running

```bash
cd LCSFed
python main.py --data_dir data --dataset_name ML_1M
```

All arguments: `python main.py -h`. Logs are written under `logs/`.
