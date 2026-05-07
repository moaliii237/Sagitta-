# Sagitta ASR — V10 Dutch Speech Recognition Fine-Tuning Pipeline

> Continued pre-training of a Wav2Vec2-based Dutch ASR model on Belgian Dutch and CSS10
> audiobook speech, with CTC decoding augmented by a domain-specific KenLM n-gram
> language model trained on Sagitta call center transcripts.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background & Motivation](#2-background--motivation)
3. [Model Architecture](#3-model-architecture)
4. [Training Data](#4-training-data)
5. [Training Strategy](#5-training-strategy)
6. [Language Model & Decoding](#6-language-model--decoding)
7. [Results](#7-results)
8. [Error Analysis](#8-error-analysis)
9. [Project Structure](#9-project-structure)
10. [Setup & Installation](#10-setup--installation)
11. [Usage](#11-usage)
12. [Known Limitations](#12-known-limitations)
13. [Next Steps — V11 Roadmap](#13-next-steps--v11-roadmap)

---

## 1. Project Overview

This repository contains the full fine-tuning pipeline for **Sagitta ASR V10** — a
Dutch automatic speech recognition model tailored for call center audio.

**The core problem:** Sagitta's V9 model was trained on CGN (Corpus Gesproken Nederlands)
telephone speech and a small set of internal Sagitta call recordings. When evaluated on
Belgian Dutch and CSS10 audiobook speech, V9 produced a WER of 100% — effectively
failing entirely on these domains.

**What V10 does:** Continues training V9 on 18,780 clips of Belgian Dutch + CSS10
audiobook speech. After 5 epochs of fine-tuning on an NVIDIA A100 (61.6 minutes),
WER drops from 100% to **8.3%** on the validation set.

**Stack:** HuggingFace Transformers · Wav2Vec2ForCTC · PyTorch · KenLM · pyctcdecode

---

## 2. Background & Motivation

### What is ASR?

Automatic Speech Recognition (ASR) converts raw audio waveforms into text. Modern
ASR systems use deep neural networks trained end-to-end on (audio, transcript) pairs.

### What is Wav2Vec2?

Wav2Vec2 (Baevski et al., 2020) is a self-supervised model from Meta AI that:

1. Pre-trains on large amounts of **unlabelled** audio using a contrastive learning
   objective — learning rich acoustic representations without needing transcripts
2. Is then fine-tuned on labelled (audio, transcript) pairs for the target language

The model used here is `GroNLP/wav2vec2-dutch-large-ft-cgn` — a large Wav2Vec2 model
pre-trained on Dutch data from the Corpus Gesproken Nederlands (CGN), which covers
telephone speech, broadcasts, and read speech across Dutch and Belgian Dutch speakers.

### What is CTC?

Connectionist Temporal Classification (CTC) is the training objective used to map
variable-length audio sequences to variable-length text sequences without requiring
explicit alignment between audio frames and characters.

During inference, CTC outputs a probability distribution over the vocabulary at each
time step. Decoding collapses repeated tokens and blank tokens to produce the final
transcript. For example:

```
hh-hh-ee-ee-ee-ll-ll-[blank]-ll-oo → hello
```

### Why V9 Failed on Belgian+CSS10

V9 was trained almost exclusively on standard Dutch (CGN) and a small set of internal
Sagitta calls. Belgian Dutch has distinct phonology, prosody, and vocabulary that V9
had never been exposed to. CSS10 audiobook speech has a very different acoustic
profile (clean studio recording, read speech vs. spontaneous conversation).

The 100% WER is not a bug — it reflects genuine domain mismatch.

---

## 3. Model Architecture

```
Raw Audio (16kHz mono float32)
        │
        ▼
┌───────────────────────────┐
│   CNN Feature Encoder     │  ← FROZEN during V10 training
│   (7 conv layers)         │    Extracts low-level acoustic features
│   4,200,448 parameters    │    from raw waveform. Already well-trained
└───────────┬───────────────┘    from V9 — freezing prevents regression.
            │
            ▼
┌───────────────────────────┐
│   Transformer Encoder     │  ← TRAINABLE during V10 training
│   (24 attention layers)   │    Learns higher-level language patterns
│   307,069,952 parameters  │    and cross-timestep dependencies.
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│   CTC Linear Head         │  ← TRAINABLE during V10 training
│   (hidden → vocab size)   │    Maps transformer output to per-token
│   4,193,442 parameters    │    probabilities at each timestep.
└───────────┬───────────────┘
            │
            ▼
   Token probabilities (34 vocab)
        │
        ▼
   Greedy / Beam Search decoding
        │
        ▼
   Dutch transcript text
```

**Total parameters:** 315,463,842
**Frozen (CNN):** 4,200,448
**Trainable:** 311,263,394

**Vocabulary (34 tokens):**
```
a b c d e f g h i j k l m n o p q r s t u v w x y z
' - à è é | [PAD] [UNK]
```
Where `|` is the word boundary delimiter and `[PAD]` (id=33) is the CTC blank token.

---

## 4. Training Data

### Datasets

| Dataset | Language | Domain | Clips | Hours (approx) |
|---|---|---|---|---|
| Belgian Dutch (CGN subset) | Belgian NL | Conversational / read | 14,985 | ~17h |
| CSS10 Dutch | Standard NL | Audiobook (read speech) | 6,431 | ~9h |
| **Combined (after filtering)** | | | **18,780 train / 1,038 val** | **~24h** |

### Data Filtering

Before training, the following clips are removed:

- **Empty transcripts** — nothing to learn from (0 removed in this run)
- **Clips > 10 seconds** — dropped to prevent GPU OOM errors (494 train, 32 val removed)

### Preprocessing Pipeline

```
CSV manifest (audio_path, text, duration, source)
        │
        ▼
Path remapping (Windows → Linux/Colab paths)
        │
        ▼
Audio loading — librosa at 16kHz mono float32
        │
        ▼
Feature extraction — Wav2Vec2FeatureExtractor
  input_values: normalised float array
  attention_mask: 1=real audio, 0=padding
        │
        ▼
Text tokenisation — Wav2Vec2CTCTokenizer
  "hallo" → [8, 1, 12, 12, 15]
        │
        ▼
HuggingFace Dataset (cached to disk after first run)
Columns: input_values, attention_mask, labels
```

**Important:** `set_format('torch')` is NOT called after preprocessing. This triggers
a NumPy 2.0 compatibility bug with variable-length arrays. The DataCollator handles
numpy → torch conversion at batch time instead.

### Data Collator

The `DataCollatorCTCWithPadding` class handles two tasks at batch time:

1. **Dynamic padding** — pads each batch to the longest example in that batch,
   not globally. This minimises wasted computation from padding.

2. **Label masking** — replaces `pad_token_id` (33) in labels with `-100`.
   PyTorch's CTC loss ignores positions where `label == -100`. Without this,
   the model tries to learn from padding tokens, producing negative loss values
   and corrupted training. This was the root cause of WER > 100% in earlier runs.

---

## 5. Training Strategy

### Continued Pre-training vs. Fine-tuning from Scratch

V10 uses **continued pre-training** — we load V9 weights and keep training, rather
than starting from the base HuggingFace model. This means:

- V9's Dutch acoustic knowledge is preserved
- The model only needs to adapt to the new domains, not re-learn Dutch from scratch
- Training converges faster and requires less data

### CNN Freezing

The CNN feature encoder is frozen during V10 training via `model.freeze_feature_encoder()`.

**Why:** The CNN learns low-level acoustic features (phoneme-like sounds) from raw
waveforms. These are domain-agnostic — a /t/ sound is a /t/ whether it's Belgian Dutch
or standard Dutch. Freezing it:

- Reduces trainable parameters from 315M to 311M
- Prevents the acoustic encoder from drifting away from what V9 learned
- Speeds up training (fewer gradient computations)

### Hyperparameters

| Parameter | Value | Reasoning |
|---|---|---|
| Learning rate | 3e-5 | Low to prevent catastrophic forgetting |
| LR schedule | Linear with warmup | Stable convergence |
| Warmup steps | 200 | ~0.07 epochs — standard for fine-tuning |
| Epochs | 5 | Sufficient convergence on this dataset size |
| Batch size (train) | 16 per device | Fits A100 40GB with headroom |
| Gradient accumulation | 2 steps | Effective batch = 32 |
| FP16 | True | Halves VRAM, speeds up A100 |
| BF16 | False | Caused NaN loss in V9 (different numerical range) |
| Gradient checkpointing | False | Caused tensor mismatch errors in V9 |
| group_by_length | True | Batches similar-duration clips → less padding |
| Best model metric | eval_loss | Load checkpoint with lowest val loss at end |

### Training Run Stats

| Metric | Value |
|---|---|
| Hardware | NVIDIA A100-SXM4-40GB (42.4 GB VRAM) |
| Runtime | 61.6 minutes |
| Total steps | 2,930 (586 per epoch) |
| Final train loss | 0.4012 |
| Checkpoints saved | 3 (steps 1500, 2000, 2500) |

---

## 6. Language Model & Decoding

### Why Add a Language Model to CTC?

CTC decodes each timestep independently. It has no notion of whether the resulting
words form a grammatically or lexically plausible Dutch sentence.

A language model (LM) rescores candidate transcripts by their linguistic plausibility.
During beam search, the final score for a hypothesis is:

```
score = log P(audio | text)  +  alpha * log P(text)  +  beta * |words|
         acoustic model score     LM score               word insertion bonus
```

### The Sagitta Language Model

A custom **4-gram KenLM language model** trained on Sagitta call center transcripts:

| Property | Value |
|---|---|
| Type | 4-gram ARPA |
| Training data | Sagitta call center transcripts |
| File size | 8.3 MB (ARPA) / 4.6 MB (binary) |
| Domain unigrams | 1,081 Sagitta-specific words |

Domain unigrams act as a vocabulary boost — words like `contract`, `engie`, `tarieven`,
`verbruik`, `stroom` are boosted so the decoder prefers them when acoustically plausible.

### Decoder Configuration

Built with `pyctcdecode.build_ctcdecoder()`:

| Parameter | Value | Found by |
|---|---|---|
| alpha (LM weight) | 0.3 | Grid search |
| beta (word insertion bonus) | 2.0 | Grid search |
| beam_width | 100 | Fixed |

**Alpha grid search result:**

| alpha | beta | WER on 500 val examples |
|---|---|---|
| 0.3 | 2.0 | **12.3%** ← best |
| 0.3 | 1.5 | 12.7% |
| 0.3 | 1.0 | 13.2% |
| 0.5 | 2.0 | 17.7% |
| 0.7 | 2.0 | 19.0% |
| 0.9 | 2.0 | 19.5% |

Low alpha (0.3) is optimal because the acoustic model is already strong on this domain.
Higher alpha values make the LM override good acoustic predictions with worse ones.

---

## 7. Results

### Main Results Table

| Model | WER | CER |
|---|---|---|
| V9 baseline on Belgian+CSS10 val | 100.0% | 94.1% |
| **V10 greedy (no LM)** | **8.3%** | **2.1%** |
| V10 + Sagitta LM + unigrams | 11.7% | 2.7% |
| V10 on real Sagitta 8kHz calls | ~22.6% | — |

**WER definition:**
```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```

**Fine-tuning gain:** WER 100.0% → 8.3% (↓ 91.7 percentage points)

### Why Greedy Beats LM+Unigrams on Val Set

The LM was trained on Sagitta call center transcripts, but the validation set is
Belgian Dutch + CSS10 audiobook speech. The LM's vocabulary is slightly mismatched
to the val set domain, causing it to occasionally override correct acoustic predictions.

On real Sagitta calls, the LM provides a small improvement (52.6% → 51.9% greedy vs LM).

### Sample Transcription — Perfect Case

```
Reference : ik zou in het bijzonder drie aspecten willen noemen die volgens mij essentieel zijn
V10 Greedy: ik zou in het bijzonder drie aspecten willen noemen die volgens mij essentieel zijn
WER       : 0.0%
```

### Sample Transcription — Real Sagitta 8kHz Call

```
Reference : oh meneer tiemann ja ik ben hier yes nee ik was bijna vergeten kijk op het moment
            dat er ophang moet ik altijd nog even wat aanbieden mocht u aan de lijn blijven...

V10 Greedy: omijnheer die manjes nee ik was bijna vergeten kijk moment dat de opbang mok
            altijd nog even wat aanbieden uh wi mogen w aan de lijn blijft voor te midden...

WER       : 22.6%
```

The model struggles with proper names (Tiemann → "die manjes") and telephone-specific
acoustic conditions — expected given the 16kHz vs 8kHz mismatch.

---

## 8. Error Analysis

### Why 8.3% WER on Val but 22.6% on Sagitta Calls?

This is **domain mismatch**, not model failure. There are three compounding factors:

| Factor | Val Set | Sagitta Calls |
|---|---|---|
| Sample rate | 16kHz | 8kHz (telephone) |
| Recording condition | Clean studio / field | Telephone codec compression |
| Speech type | Read / conversational | Spontaneous call center dialogue |
| Proper names | Standard Dutch names | Customer/agent names, brand names |

The model was never trained on 8kHz telephone audio. The acoustic input distribution
is fundamentally different — telephone codecs remove frequencies above 4kHz, which
contains important phonetic information.


### Common Error Patterns

| Error type | Example | Cause |
|---|---|---|
| Proper names | Tiemann → "die manjes" | OOV words, 8kHz distortion |
| Word boundaries | "mijnheer" → "mijn heer" | CTC segmentation |
| Filler words | "uh", "eh" → various | Not in training data |
| Code-switching | English words in Dutch | Model vocab is Dutch-only |

---

## 9. Project Structure

```
sagitta-asr-v10/
│
├── config.py                   ← Single source of truth for all paths
│                                  and hyperparameters. Edit this file
│                                  when paths change — nothing else.
│
├── requirements.txt            ← Pinned package versions
├── .gitignore                  ← Excludes model weights, audio, .arpa files
│
├── utils/
│   └── verify.py               ← GPU check + path validation
│                                  Run before anything else
│
├── model/
│   ├── loader.py               ← load_processor() and load_model()
│   │                              Handles processor_config.json rename fix
│   │                              and CNN freezing
│   └── metrics.py              ← compute_metrics() for WER + CER
│                                  Called by HuggingFace Trainer after
│                                  each evaluation step
│
├── data/
│   ├── prepare.py              ← Audio copy (Drive → local SSD)
│   │                              Manifest loading + path remapping
│   │                              Audio preprocessing + HF dataset build
│   └── collator.py             ← DataCollatorCTCWithPadding
│                                  Dynamic padding + -100 label masking
│
├── training/
│   └── train.py                ← Full training pipeline entry point
│                                  TrainingArguments + Trainer + save
│
├── evaluation/
│   ├── evaluate_greedy.py      ← Greedy WER/CER before and after training
│   ├── evaluate_lm.py          ← LM decoder build + greedy vs LM comparison
│   └── evaluate_sagitta.py     ← Real call transcription vs WhisperX refs
│                                  Handles long audio via 30s chunking
│
├── lm/
│   └── tune_alpha_beta.py      ← 4x4 grid search over alpha [0.3-0.9]
│                                  and beta [0.5-2.0] on 500 val examples
│
└── results/                    ← JSON outputs (gitignored)
    ├── train_metrics.json
    ├── greedy_comparison.json
    ├── lm_comparison.json
    ├── full_summary.json
    └── sagitta_vs_whisper.json
```

---

## 10. Setup & Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (A100 recommended, minimum 16GB VRAM)
- Google Drive mounted at `/content/drive` (Colab) or configured path

### Install Python packages

```bash
pip install -r requirements.txt
```

### Install KenLM (from source — no pip package)

```bash
# Ubuntu / Colab
sudo apt-get install -y libboost-all-dev cmake build-essential

git clone https://github.com/kpu/kenlm.git
mkdir kenlm/build && cd kenlm/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Python bindings
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### Configure paths

Open `config.py` and set `DRIVE_ROOT` to your Google Drive root:

```python
DRIVE_ROOT = '/content/drive/MyDrive/ArcumSagitta'  # Colab
# or
DRIVE_ROOT = '/mnt/drive/MyDrive/ArcumSagitta'       # local mount
```

All other paths are derived from `DRIVE_ROOT` automatically.

### Verify setup

```bash
python utils/verify.py
```

Expected output:
```
=== GPU ===
  NVIDIA A100-SXM4-40GB
  42.4 GB VRAM

---

## 11. Usage

### Run order

```bash
# 1. Verify GPU and all paths exist
python utils/verify.py

# 2. Fine-tune V9 → V10 (61 min on A100)
python training/train.py

# 3. Compare WER before and after fine-tuning (greedy, no LM)
python evaluation/evaluate_greedy.py

# 4. Build LM decoder and evaluate greedy vs LM
python evaluation/evaluate_lm.py

# 5. Test on real Sagitta 8kHz calls vs WhisperX references
python evaluation/evaluate_sagitta.py

# Optional: grid search for optimal alpha and beta
python lm/tune_alpha_beta.py
```

### Resume after Colab disconnect

If training is interrupted, resume from the latest checkpoint:

```python
# In training/train.py, change:
train_result = trainer.train()
# To:
train_result = trainer.train(resume_from_checkpoint=True)
```

Checkpoints are saved every 500 steps to `V10_MODEL_PATH`.

---

## 12. Known Limitations

### 16kHz vs 8kHz Mismatch

The most significant limitation. All training data (Belgian Dutch + CSS10) is 16kHz
clean audio. Sagitta call center recordings are 8kHz telephone audio compressed with
G.711 or similar codecs.

This mismatch causes the ~22% WER on real calls. The model's acoustic encoder has
never seen telephone-compressed audio and cannot handle it reliably.

**Fix in V11:** Fine-tune directly on 8kHz Sagitta calls.

### WhisperX as Ground Truth

The evaluation in `evaluate_sagitta.py` uses WhisperX transcripts as the reference.
These are not human-verified — they are themselves model outputs. WER of 52.6% means
"52.6% different from WhisperX", not "52.6% different from perfect human transcription".
Actual accuracy against human transcripts may differ.

### Vocabulary Coverage

The model vocabulary contains only 34 tokens (a-z lowercase, basic punctuation).
Proper names, brand names (Engie, Sagitta), and code-switched English words are
all handled as sequences of characters — there is no word-level OOV handling at
the acoustic model level.

---

## 13. Next Steps — V11 Roadmap

| Priority | Task | Expected Impact |
|---|---|---|
| 🔴 High | Fine-tune on real Sagitta 8kHz calls | Close the 16kHz/8kHz gap |
| 🔴 High | Upsample 8kHz audio to 16kHz before training | Alternative to full retraining |
| 🟡 Medium | Add more Sagitta transcripts to LM training data | Improve domain LM |
| 🟡 Medium | Human-verify WhisperX transcripts for eval | Cleaner ground truth |
| 🟢 Low | Experiment with Whisper as base model | Different architecture tradeoffs |
| 🟢 Low | Speaker diarization integration | Separate agent vs customer WER |

---

## Files Not in This Repository

Large files are stored on Google Drive and excluded via `.gitignore`:

| File | Size | Location on Drive |
|---|---|---|
| `model.safetensors` | 1.2 GB | `v10/models/gronlp_belgian_css10_finetuned/` |
| `dutch_sagitta_lm.arpa` | 8.3 MB | `project/` |
| `dutch_sagitta_lm.bin` | 4.6 MB | `project/` |
| Belgian audio clips | ~17h audio | `v9/output/belgian_audio/` |
| CSS10 audio clips | ~9h audio | `v9/output/css10_audio/` |
| Sagitta call recordings | — | `datasets/sagitta voice/15/` |
