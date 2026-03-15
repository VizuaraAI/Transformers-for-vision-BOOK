# Ch 02 — BERT from Scratch

A miniature BERT model built entirely from scratch in PyTorch and trained on the IMDb sentiment dataset.

## What's Inside

The notebook walks through every layer of BERT — embeddings, multi-head self-attention, feed-forward blocks, encoder stack — and wires them into a working sequence classifier.

**Architecture**

```
Token + Position + Segment Embeddings
        ↓
Transformer Encoder × N
  ├─ Multi-Head Self-Attention
  ├─ Layer Norm + Residual
  ├─ Feed-Forward Network
  └─ Layer Norm + Residual
        ↓
  [CLS] → Classification Head → Sentiment
```

**Config** — `dim=256`, `heads=4`, `layers=4`, `max_len=256`, tokenizer: GPT-2 BPE via `tiktoken`

## Quick Start

```bash
pip install torch datasets tiktoken tqdm scikit-learn
```

Open `BERT_from_scratch_on_IDMB_Dataset.ipynb` and run all cells.

## Notebook Outline

| # | Section | Description |
|---|---------|-------------|
| 1.1–1.6 | Setup & Tokenization | Dataset loading, BPE tokenizer, special tokens (`[CLS]`, `[SEP]`, `[PAD]`) |
| 1.7–1.10 | Data Pipeline | Encoding, attention masks, `Dataset` & `DataLoader` |
| 1.11–1.16 | Model | Embeddings → Attention → FFN → Encoder → Classification head |
| 1.17–1.19 | Training | AdamW optimizer, cross-entropy loss, 100 epochs |
| 1.20–1.22 | Eval & Inference | Test accuracy, save/load model, predict on custom text |

<!-- ## Reference

 -->
