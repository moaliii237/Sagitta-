# Sagitta ASR V10, Dutch speech recognition fine-tuning pipeline

This is my training pipeline for Sagitta ASR V10, a Dutch speech recognition model
for call center audio. I continue training a Wav2Vec2 Dutch model on Belgian Dutch
and CSS10 audiobook speech, and I decode with CTC plus a 4-gram KenLM language model
trained on Dutch call center transcripts.

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Background and motivation](#2-background-and-motivation)
3. [Model architecture](#3-model-architecture)
4. [Training data](#4-training-data)
5. [Training strategy](#5-training-strategy)
6. [Language model and decoding](#6-language-model-and-decoding)
7. [Results](#7-results)
8. [Error analysis](#8-error-analysis)
9. [Project structure](#9-project-structure)
10. [Setup and installation](#10-setup-and-installation)
11. [Usage](#11-usage)
12. [Known limitations](#12-known-limitations)
13. [Next steps, V11 roadmap](#13-next-steps-v11-roadmap)

## 1. Project overview

This repo holds the full fine-tuning pipeline for Sagitta ASR V10, a Dutch automatic
speech recognition model for call center audio.

The problem I started from: the V9 model was trained on CGN (Corpus Gesproken
Nederlands) telephone speech plus a small set of internal call recordings. When I
tested V9 on Belgian Dutch and CSS10 audiobook speech, the WER was 100%. It basically
failed on those domains.

What V10 does: I keep training V9 on 18,780 clips of Belgian Dutch and CSS10 audiobook
speech. After 5 epochs on an NVIDIA A100 (61.6 minutes), the WER drops from 100% to
8.3% on the validation set.

Stack: HuggingFace Transformers, Wav2Vec2ForCTC, PyTorch, KenLM, pyctcdecode.

## 2. Background and motivation

### What is ASR?

Automatic Speech Recognition (ASR) turns raw audio into text. Modern ASR systems use
deep neural networks trained end to end on (audio, transcript) pairs.

### What is Wav2Vec2?

Wav2Vec2 (Baevski et al., 2020) is a self-supervised model from Meta AI. It works in
two stages:

1. It pre-trains on a large amount of unlabelled audio with a contrastive objective,
   so it learns acoustic representations without needing transcripts.
2. It is then fine-tuned on labelled (audio, transcript) pairs for the target language.

The model I use here is `GroNLP/wav2vec2-dutch-large-ft-cgn`, a large Wav2Vec2 model
pre-trained on Dutch data from CGN, which covers telephone speech, broadcasts, and
read speech across Dutch and Belgian Dutch speakers.

### What is CTC?

Connectionist Temporal Classification (CTC) is the training objective that maps a
variable-length audio sequence to a variable-length text sequence without needing an
explicit alignment between audio frames and characters.

At inference time CTC gives a probability over the vocabulary at each time step.
Decoding collapses repeated tokens and blank tokens into the final transcript.
For example:

```
hh-hh-ee-ee-ee-ll-ll-[blank]-ll-oo  becomes  hello
```

### Why V9 failed on Belgian and CSS10

V9 was trained almost only on standard Dutch (CGN) and a small set of internal calls.
Belgian Dutch has different phonology, prosody, and vocabulary that V9 never saw.
CSS10 audiobook speech also has a very different acoustic profile (clean studio
recording, read speech instead of spontaneous conversation).

So the 100% WER is not a bug. It is a real domain mismatch.

## 3. Model architecture

The model is a stack of three parts. The CNN feature encoder is frozen during V10
training and the rest is trainable:

```
Raw audio (16kHz mono float32)
    |
    v
CNN feature encoder       FROZEN during V10
  7 conv layers           extracts low-level acoustic features from the
  4,200,448 params        waveform. Already well trained from V9, so I
                          freeze it to prevent regression.
    |
    v
Transformer encoder       TRAINABLE during V10
  24 attention layers     learns higher-level language patterns and
  307,069,952 params      dependencies across time steps.
    |
    v
CTC linear head           TRAINABLE during V10
  hidden to vocab size    maps the transformer output to per-token
  4,193,442 params        probabilities at each time step.
    |
    v
Token probabilities (34 vocab)
    |
    v
Greedy or beam search decoding
    |
    v
Dutch transcript text
```

Total parameters: 315,463,842. Frozen (CNN): 4,200,448. Trainable: 311,263,394.

Vocabulary (34 tokens):

```
a b c d e f g h i j k l m n o p q r s t u v w x y z
' - a-grave e-grave e-acute | [PAD] [UNK]
```

Here `|` is the word boundary delimiter and `[PAD]` (id=33) is the CTC blank token.

## 4. Training data

### Datasets

| Dataset | Language | Domain | Clips | Hours (approx) |
|---|---|---|---|---|
| Belgian Dutch (CGN subset) | Belgian NL | Conversational and read | 14,985 | ~17h |
| CSS10 Dutch | Standard NL | Audiobook (read speech) | 6,431 | ~9h |
| Combined (after filtering) | | | 18,780 train / 1,038 val | ~24h |

### Data filtering

Before training I remove these clips:

- Empty transcripts. Nothing to learn from (0 removed in this run).
- Clips longer than 10 seconds. Dropped to avoid GPU OOM (494 train, 32 val removed).

### Preprocessing pipeline

```
CSV manifest (audio_path, text, duration, source)
    |
    v
Path remapping (Windows to Linux/Colab paths)
    |
    v
Audio loading, librosa at 16kHz mono float32
    |
    v
Feature extraction, Wav2Vec2FeatureExtractor
  input_values: normalised float array
  attention_mask: 1 for real audio, 0 for padding
    |
    v
Text tokenisation, Wav2Vec2CTCTokenizer
  "hallo" becomes [8, 1, 12, 12, 15]
    |
    v
HuggingFace Dataset (cached to disk after first run)
Columns: input_values, attention_mask, labels
```

One thing to watch: I do NOT call `set_format('torch')` after preprocessing. That
triggers a NumPy 2.0 compatibility bug with variable-length arrays. The data collator
does the numpy to torch conversion at batch time instead.

### Data collator

The `DataCollatorCTCWithPadding` class does two things at batch time:

1. Dynamic padding. It pads each batch to the longest example in that batch, not the
   longest in the whole dataset. This wastes less computation on padding.

2. Label masking. It replaces `pad_token_id` (33) in the labels with `-100`. The
   PyTorch CTC loss ignores positions where `label == -100`. Without this the model
   tries to learn from padding tokens, which gives negative loss values and corrupts
   training. This was the root cause of WER above 100% in my earlier runs.

## 5. Training strategy

### Continued training instead of training from scratch

V10 continues training from V9 weights instead of starting from the base HuggingFace
model. This way:

- V9's Dutch acoustic knowledge is kept.
- The model only has to adapt to the new domains, it does not relearn Dutch.
- Training converges faster and needs less data.

### Freezing the CNN

I freeze the CNN feature encoder during V10 training with
`model.freeze_feature_encoder()`.

The reason: the CNN learns low-level acoustic features (phoneme-like sounds) from the
raw waveform. These are domain independent. A /t/ sound is a /t/ whether it is Belgian
Dutch or standard Dutch. Freezing it:

- Reduces trainable parameters from 315M to 311M.
- Keeps the acoustic encoder from drifting away from what V9 learned.
- Speeds up training, fewer gradients to compute.

### Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| Learning rate | 3e-5 | Low, to prevent catastrophic forgetting |
| LR schedule | Linear with warmup | Stable convergence |
| Warmup steps | 200 | ~0.07 epochs, standard for fine-tuning |
| Epochs | 5 | Enough convergence for this dataset size |
| Batch size (train) | 16 per device | Fits A100 40GB with headroom |
| Gradient accumulation | 2 steps | Effective batch = 32 |
| FP16 | True | Halves VRAM, faster on A100 |
| BF16 | False | Caused NaN loss in V9 (different numerical range) |
| Gradient checkpointing | False | Caused tensor mismatch errors in V9 |
| group_by_length | True | Batches similar-duration clips, less padding |
| Best model metric | eval_loss | Load the checkpoint with the lowest val loss |

### Training run stats

| Metric | Value |
|---|---|
| Hardware | NVIDIA A100-SXM4-40GB (42.4 GB VRAM) |
| Runtime | 61.6 minutes |
| Total steps | 2,930 (586 per epoch) |
| Final train loss | 0.4012 |
| Checkpoints saved | 3 (steps 1500, 2000, 2500) |

## 6. Language model and decoding

### Why add a language model to CTC?

CTC decodes each time step on its own. It has no idea whether the words it produces
form a plausible Dutch sentence.

A language model rescores candidate transcripts by how likely they are as Dutch text.
During beam search the score for a hypothesis is:

```
score = log P(audio | text)  +  alpha * log P(text)  +  beta * number_of_words
        acoustic model score     language model score    word insertion bonus
```

### The call center language model

I trained a custom 4-gram KenLM language model on Dutch call center transcripts:

| Property | Value |
|---|---|
| Type | 4-gram ARPA |
| Training data | Dutch call center transcripts |
| File size | 8.3 MB (ARPA) / 4.6 MB (binary) |
| Domain unigrams | 1,081 domain-specific words |

The domain unigrams act as a vocabulary boost. Common call center words (contract,
energy supplier names, tariff and consumption terms) are boosted so the decoder
prefers them when they are acoustically plausible.

### Decoder configuration

Built with `pyctcdecode.build_ctcdecoder()`:

| Parameter | Value | Found by |
|---|---|---|
| alpha (LM weight) | 0.3 | Grid search |
| beta (word insertion bonus) | 2.0 | Grid search |
| beam_width | 100 | Fixed |

Alpha grid search result:

| alpha | beta | WER on 500 val examples |
|---|---|---|
| 0.3 | 2.0 | 12.3% (best) |
| 0.3 | 1.5 | 12.7% |
| 0.3 | 1.0 | 13.2% |
| 0.5 | 2.0 | 17.7% |
| 0.7 | 2.0 | 19.0% |
| 0.9 | 2.0 | 19.5% |

Low alpha (0.3) works best because the acoustic model is already strong on this
domain. Higher alpha lets the language model override good acoustic predictions with
worse ones.

## 7. Results

### Main results

| Model | WER | CER |
|---|---|---|
| V9 baseline on Belgian and CSS10 val | 100.0% | 94.1% |
| V10 greedy (no LM) | 8.3% | 2.1% |
| V10 + LM + unigrams | 11.7% | 2.7% |
| V10 on real 8kHz calls | ~22.6% | n/a |

WER definition:

```
WER = (substitutions + deletions + insertions) / total reference words
```

Fine-tuning gain: WER 100.0% to 8.3%, down 91.7 percentage points.

### Why greedy beats LM + unigrams on the val set

The language model was trained on call center transcripts, but the validation set is
Belgian Dutch plus CSS10 audiobook speech. The LM vocabulary is a bit mismatched to
the val set domain, so it sometimes overrides correct acoustic predictions.

On the real call domain the LM gives a small improvement instead (greedy a little
worse than LM).

### Sample transcription, perfect case

This one is from the Belgian Dutch validation set:

```
Reference : ik zou in het bijzonder drie aspecten willen noemen die volgens mij essentieel zijn
V10 Greedy: ik zou in het bijzonder drie aspecten willen noemen die volgens mij essentieel zijn
WER       : 0.0%
```

On the harder 8kHz telephone domain the model still makes mistakes on proper names and
telephone-specific acoustic conditions. That is expected given the 16kHz vs 8kHz
mismatch, see the error analysis below.

## 8. Error analysis

### Why 8.3% WER on val but about 22.6% on real calls?

This is domain mismatch, not a broken model. There are three factors stacking up:

| Factor | Val set | Real calls |
|---|---|---|
| Sample rate | 16kHz | 8kHz (telephone) |
| Recording condition | Clean studio or field | Telephone codec compression |
| Speech type | Read or conversational | Spontaneous call center dialogue |
| Proper names | Standard Dutch names | Customer and agent names, brand names |

The model was never trained on 8kHz telephone audio. The acoustic input distribution
is just different. Telephone codecs remove frequencies above 4kHz, and those carry
important phonetic information.

### Common error patterns

| Error type | Cause |
|---|---|
| Proper names | Out-of-vocabulary words, 8kHz distortion |
| Word boundaries | CTC segmentation splits or joins words |
| Filler words ("uh", "eh") | Not in the training data |
| Code-switching (English words in Dutch) | The model vocabulary is Dutch only |

## 9. Project structure

```
sagitta-asr-v10/
  config.py            Single source of truth for all paths and
                       hyperparameters. Edit this file when paths
                       change, nothing else.

  requirements.txt     Pinned package versions
  .gitignore           Excludes model weights, audio, .arpa files

  utils/
    verify.py          GPU check and path validation. Run this first.

  model/
    loader.py          load_processor() and load_model(). Handles the
                       processor_config.json rename fix and CNN freezing.
    metrics.py         compute_metrics() for WER and CER. Called by the
                       HuggingFace Trainer after each eval step.

  data/
    prepare.py         Audio copy (Drive to local SSD), manifest loading
                       and path remapping, audio preprocessing and HF
                       dataset build.
    collator.py        DataCollatorCTCWithPadding. Dynamic padding and
                       -100 label masking.

  training/
    train.py           Full training pipeline entry point.
                       TrainingArguments, Trainer, and save.

  evaluation/
    evaluate_greedy.py    Greedy WER and CER before and after training.
    evaluate_lm.py        LM decoder build and greedy vs LM comparison.
    evaluate_sagitta.py   Real call transcription vs WhisperX references.
                          Handles long audio with 30s chunking.

  lm/
    tune_alpha_beta.py    4x4 grid search over alpha [0.3-0.9] and
                          beta [0.5-2.0] on 500 val examples.

  results/             JSON outputs (gitignored)
    train_metrics.json
    greedy_comparison.json
    lm_comparison.json
    full_summary.json
    sagitta_vs_whisper.json
```

## 10. Setup and installation

### Requirements

- Python 3.10+
- A CUDA GPU (A100 recommended, minimum 16GB VRAM)
- Google Drive mounted at `/content/drive` (Colab) or a configured path

### Install Python packages

```bash
pip install -r requirements.txt
```

### Install KenLM (from source, there is no pip package)

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

It prints the GPU name and VRAM, then checks that every required path exists.

## 11. Usage

### Run order

```bash
# 1. Verify GPU and that all paths exist
python utils/verify.py

# 2. Fine-tune V9 to V10 (about 61 min on A100)
python training/train.py

# 3. Compare WER before and after fine-tuning (greedy, no LM)
python evaluation/evaluate_greedy.py

# 4. Build the LM decoder and evaluate greedy vs LM
python evaluation/evaluate_lm.py

# 5. Test on real 8kHz calls vs WhisperX references
python evaluation/evaluate_sagitta.py

# Optional: grid search for the best alpha and beta
python lm/tune_alpha_beta.py
```

### Resume after a Colab disconnect

If training stops, resume from the latest checkpoint:

```python
# In training/train.py, change:
train_result = trainer.train()
# to:
train_result = trainer.train(resume_from_checkpoint=True)
```

Checkpoints are saved every 500 steps to `V10_MODEL_PATH`.

## 12. Known limitations

### 16kHz vs 8kHz mismatch

This is the biggest limitation. All my training data (Belgian Dutch and CSS10) is
16kHz clean audio. The real call recordings are 8kHz telephone audio compressed with
G.711 or a similar codec.

This mismatch is the reason for the ~22% WER on real calls. The acoustic encoder has
never seen telephone-compressed audio, so it cannot handle it reliably.

Fix planned for V11: fine-tune directly on 8kHz calls.

### WhisperX as ground truth

The evaluation in `evaluate_sagitta.py` uses WhisperX transcripts as the reference.
These are not checked by a human, they are model output themselves. So a WER number
here means "different from WhisperX", not "different from a perfect human
transcription". The real accuracy against human transcripts can be different.

### Vocabulary coverage

The model vocabulary has only 34 tokens (a-z lowercase, basic punctuation). Proper
names, brand names, and code-switched English words are all handled as character
sequences. There is no word-level out-of-vocabulary handling at the acoustic model
level.

## 13. Next steps, V11 roadmap

| Priority | Task | Expected impact |
|---|---|---|
| High | Fine-tune on real 8kHz calls | Close the 16kHz/8kHz gap |
| High | Upsample 8kHz audio to 16kHz before training | Alternative to full retraining |
| Medium | Add more transcripts to the LM training data | Better domain LM |
| Medium | Human-verify the WhisperX transcripts for eval | Cleaner ground truth |
| Low | Try Whisper as the base model | Different architecture tradeoffs |
| Low | Add speaker diarization | Separate agent vs customer WER |

## Files not in this repository

Large files live on Google Drive and are excluded with `.gitignore`:

| File | Size | Location on Drive |
|---|---|---|
| `model.safetensors` | 1.2 GB | `v10/models/gronlp_belgian_css10_finetuned/` |
| `dutch_sagitta_lm.arpa` | 8.3 MB | `project/` |
| `dutch_sagitta_lm.bin` | 4.6 MB | `project/` |
| Belgian audio clips | ~17h audio | `v9/output/belgian_audio/` |
| CSS10 audio clips | ~9h audio | `v9/output/css10_audio/` |
| Real call recordings | n/a | `datasets/sagitta voice/15/` |
