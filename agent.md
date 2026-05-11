---
name: CLAP Project Guide
description: Guide for the CLAP (Contrastive Language-Audio Pretraining) codebase — architecture, key files, usage patterns, and conventions.
---

# CLAP (Contrastive Language-Audio Pretraining)

CLAP learns acoustic concepts from natural language supervision and enables zero-shot audio classification, retrieval, and captioning. Based on Microsoft's msclap library (v1.3.3).

## Project Structure

```
msclap/
├── __init__.py              # Re-exports CLAPWrapper as CLAP
├── CLAPWrapper.py           # Main entry point — model loading, preprocessing, inference
├── models/
│   ├── clap.py              # CLAP model: AudioEncoder + TextEncoder + contrastive head
│   ├── audio.py             # Audio encoders: Cnn14 (CNN-based), HTSAT (transformer-based)
│   ├── htsat.py             # HTS-AT: hierarchical token-semantic audio transformer (941 lines)
│   ├── mapper.py            # ClapCaptionModel: audio captioning with GPT-2 decoder
│   ├── config.py            # HTSAT training config (not used for inference)
│   ├── pytorch_utils.py     # Utilities for HTSAT
│   └── utils.py             # Config reader helper
├── configs/
│   ├── config_2022.yml      # 2022 model config (Cnn14 audio encoder)
│   ├── config_2023.yml      # 2023 model config (HTSAT audio encoder, GPT-2 text encoder)
│   └── config_clapcap.yml   # Captioning model config
examples/
├── zero_shot_classification.py
├── zero_shot_predictions.py
├── audio_captioning.py
└── esc50_dataset.py
```

## Model Versions

| Version  | Audio Encoder | Text Encoder | Purpose |
|----------|--------------|--------------|---------|
| `'2022'` | Cnn14        | RoBERTa      | Zero-shot classification & retrieval |
| `'2023'` | HTSAT        | GPT-2        | Zero-shot classification & retrieval (better performance) |
| `'clapcap'` | HTSAT     | GPT-2 (+decoder) | Audio captioning |

## Key Architecture (`models/clap.py`)

- **`AudioEncoder`**: CNN14 or HTSAT → feature → `Projection` (MLP with residual + LayerNorm)
- **`TextEncoder`**: HuggingFace transformer (GPT-2/RoBERTa/CLIP/BERT) → pooled output → `Projection`
- **`CLAP`**: Combines both encoders + learns `logit_scale` parameter for contrastive loss
- **`Projection`**: Shared projection block: Linear → GeLU → Dropout → Linear → +residual → LayerNorm

## Audio Captioning (`models/mapper.py`)

- **`ClapCaptionModel`**: Audio → CLAP encoder → MLP or TransformerMapper → GPT-2 decoder
- **`TransformerMapper`**: Prefix tuning — learns prefix tokens prepended to GPT-2 embeddings
- Supports beam search decoding in `CLAPWrapper.generate_caption()`

## Main Entry Point (`msclap/__init__.py` → `CLAPWrapper`)

```python
from msclap import CLAP

# Load model (weights auto-downloaded from HuggingFace)
clap = CLAP(version='2023', use_cuda=False)

# Get embeddings
text_emb = clap.get_text_embeddings(["dog barking", "rain"])
audio_emb = clap.get_audio_embeddings(["audio.wav"])

# Contrastive similarity
sim = clap.compute_similarity(audio_emb, text_emb)

# Captioning (version='clapcap')
captions = clap.generate_caption(["audio.wav"])
```

## Key Design Details

- **Weight download**: Auto-downloaded from HuggingFace (`microsoft/msclap` repo), cached locally
- **Config loading**: YAML configs parsed to `argparse.Namespace`, used for model construction
- **Audio preprocessing**: Loads via `torchaudio`, resamples to config's `sampling_rate`, pads/truncates to `duration` seconds
- **DDP unwrapping**: Checkpoints may be saved with DataParallel wrapper — weights are unwrapped before `load_state_dict`
- **Inference only**: All forward passes wrapped in `torch.no_grad()`
- **`_generic_batch_inference`**: Generator that yields batch results; used for batched embedding extraction

## Dependencies (pyproject.toml)

- Python ≥ 3.8
- PyTorch ≥ 2.1, torchaudio ≥ 2.1
- transformers ≥ 4.34
- librosa ≥ 0.10.1, torchlibrosa ≥ 0.1.0
- numpy, pandas, scikit-learn, pyyaml, tqdm
- Weights auto-downloaded from HuggingFace via `huggingface_hub`

## Common Tasks

- **Zero-shot classification**: `clap.get_text_embeddings(class_labels)` → `clap.get_audio_embeddings(audio_files)` → `clap.compute_similarity()`
- **Batched processing**: Use `get_audio_embeddings_per_batch(audio_files, batch_size)` for memory efficiency
- **Adding a new version**: Add config YAML + model weight key to `CLAPWrapper.model_name` dict
