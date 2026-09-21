# 🧭 EvGNN

<p align="center">
  <img alt="Paper" src="https://img.shields.io/badge/Paper-Artificial%20Intelligence-8E24AA">
</p>

Official implementation for **EvGNN**, the evidence-guided graph neural network framework for learning with noisy and sparse node labels.

This repository is a cleaned research release of the code for:

> Evidence-guided Learning against Noisy and Sparse Labels on Graphs

EvGNN trains a multi-expert graph neural network under sparse and noisy supervision. It estimates evidence and uncertainty with subjective logic, fuses expert decisions with Dempster-Shafer evidence theory, and uses consistency learning plus pseudo-labeling to improve supervision from unlabeled nodes.

## ✨ Highlights

- Robust semi-supervised node classification with sparse and noisy labels.
- Multi-view expert architecture with evidence-aware uncertainty estimation.
- Decision fusion for conflicting experts.
- Consistency and pseudo-label regularization for unlabeled nodes.
- Reproducible experiment presets for Citeseer, PubMed, DBLP, Coauthor CS, Amazon Computers, and Amazon Photo.

## 🗂️ Repository Structure

```text
.
|-- arguments.py          # Command-line arguments
|-- configs.py            # Experiment presets
|-- train.py              # Main training and evaluation loop
|-- run.py                # Batch experiment launcher
|-- model.py              # 🧭 EvGNN model, losses, evidence fusion
|-- deepergnn.py          # GNN backbone components
|-- aug.py                # Graph augmentation utilities
|-- functional.py         # Graph drop/weight helpers
|-- datasets.py           # PyG dataset loaders
|-- dataset_process.py    # Planetoid-style raw data processing
|-- utils_process.py      # Splits, noise injection, sparse helpers
|-- data/                 # Lightweight raw data files included here
`-- save_models/          # Checkpoints are written here at runtime
```

The release intentionally excludes generated files such as Python caches, local IDE settings, previous result logs, trained checkpoints, t-SNE artifacts, and PyG `processed/` caches.

## ⚙️ Installation

Create a fresh environment with Python 3.8 or later.

```bash
conda create -n evgnn python=3.8
conda activate evgnn
pip install -r requirements.txt
```

Install PyTorch Geometric packages according to your PyTorch and CUDA versions if the generic installation does not match your machine:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```

See the official PyTorch Geometric installation guide for CUDA-specific wheels.

## 📊 Data

The code supports six datasets used in the paper:

- Citation networks: `citeseer`, `pubmed`
- Full citation network: `dblp`
- Coauthor network: `cs`
- Amazon co-purchase networks: `computers`, `photo`

Lightweight raw files for Citeseer, PubMed, Cora, and DBLP are included under `data/`. PyTorch Geometric datasets such as Coauthor CS and Amazon Computers/Photo are downloaded and processed automatically under `data_test/` on first use.

## 🚀 Training

Run a single experiment from the command line:

```bash
python train.py --preset dblp_uniform
```

Command-line options can still override preset values:

```bash
python train.py --preset dblp_uniform --seed 3 --ptb_rate 0.4
```

Run the batch settings encoded in the experiment launcher:

```bash
python run.py
```

Checkpoints are saved to `save_models/<dataset>.pkl`.

## 📚 Citation

BibTeX:

```bibtex
@article{yi2026evidence,
  title={Evidence-guided Learning against Noisy and Sparse Labels on Graphs},
  author={Yi, Siyu and Zhang, Wei and Mao, Zhengyang and Zhou, Yongdao and Qiao, Ziyue and Shen, Li and Tao, Dacheng and Lv, Jiancheng and Ju, Wei},
  journal={Artificial Intelligence},
  pages={104608},
  year={2026},
  publisher={Elsevier}
}
```

