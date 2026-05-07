# Sagitta ASR — V10 Fine-Tuning Pipeline

Fine-tuning pipeline for Dutch speech recognition at Sagitta call centers.
Extends the V9 Wav2Vec2 model with Belgian Dutch and CSS10 audiobook data,
then evaluates against real call center recordings.

---

## Results

| Model | WER | CER |
|---|---|---|
| V9 baseline (on Belgian+CSS10 val) | 100.0% | 94.1% |
| **V10 greedy (after fine-tuning)** | **8.3%** | **2.1%** |
| V10 + Sagitta LM + unigrams | 11.7% | 2.7% |
| V10 on real Sagitta 8kHz calls | ~22.6% | — |

**Fine-tune gain: WER 100% → 8.3% (↓ 91.7%)**

The 22.6% WER on real Sagitta calls is expected — the model was trained
on 16kHz clean audio while Sagitta calls are 8kHz telephone audio.
Next step (V11): fine-tune directly on 8kHz Sagitta calls.

---

## Project Structure

```
sagitta-asr-v10/
├── config.py                        ← All paths & hyperparameters (edit this)
├── requirements.txt
├── .gitignore
│
├── utils/
│   └── verify.py                    ← Check GPU + paths before training
│
├── model/
│   ├── loader.py                    ← Load processor + model from V9
│   └── metrics.py                   ← WER/CER compute function
│
├── data/
│   ├── prepare.py                   ← Audio copy, manifest loading, preprocessing
│   └── collator.py                  ← DataCollatorCTCWithPadding
│
├── training/
│   └── train.py                     ← TrainingArguments + Trainer + save
│
├── evaluation/
│   ├── evaluate_greedy.py           ← Baseline + after fine-tuning comparison
│   ├── evaluate_lm.py               ← LM decoder build + full LM eval
│   └── evaluate_sagitta.py          ← Real call transcription vs WhisperX
│
├── lm/
│   └── tune_alpha_beta.py           ← Grid search alpha & beta (optional)
│
└── results/                         ← JSON outputs saved here
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

KenLM must be built from source (required for the LM decoder):

```bash
# On Ubuntu/Colab:
apt-get install -y libboost-all-dev
git clone https://github.com/kpu/kenlm.git
mkdir kenlm/build && cd kenlm/build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### 2. Configure paths

Edit `config.py` — set `DRIVE_ROOT` to wherever your Google Drive is mounted
and verify all paths point to your V9 model, manifests, and audio folders.

### 3. Verify setup

```bash
python utils/verify.py
```

---

## Usage

Run scripts in this order:

```bash
# 1. Check GPU and paths
python utils/verify.py

# 2. Fine-tune V9 → V10
python training/train.py

# 3. Evaluate greedy (before vs after)
python evaluation/evaluate_greedy.py

# 4. Evaluate with language model
python evaluation/evaluate_lm.py

# 5. Test on real Sagitta calls
python evaluation/evaluate_sagitta.py

# Optional: tune LM alpha/beta parameters
python lm/tune_alpha_beta.py
```

---

## Model Architecture

**Base model**: `GroNLP/wav2vec2-dutch-large-ft-cgn` (Wav2Vec2ForCTC)

**Training data for V10**:
- Belgian Dutch: 14,985 clips
- CSS10 Dutch audiobook: 6,431 clips
- Total after filtering: 18,780 train / 1,038 val

**Key training decisions**:
- Learning rate: 3e-5 (low to prevent catastrophic forgetting)
- CNN feature encoder frozen (preserves V9 acoustic learning)
- FP16 enabled, BF16 disabled (BF16 caused NaN loss in V9)
- Gradient checkpointing disabled (caused tensor mismatch errors in V9)
- Effective batch size: 32 (16 × 2 accumulation steps)
- 5 epochs, 61.6 min on A100

**Language model**: Custom Sagitta 4-gram ARPA built with KenLM
- Alpha: 0.3 (low — acoustic model is stronger than LM on this domain)
- Beta: 2.0 (word insertion bonus helps Dutch compound words)
- 1,081 domain unigrams (Sagitta call center vocabulary)

---

## Files NOT in this repo

These are stored on Google Drive (too large for GitHub):

| File | Size | Location |
|---|---|---|
| `model.safetensors` | 1.2 GB | `Drive/v10/models/gronlp_belgian_css10_finetuned/` |
| `dutch_sagitta_lm.arpa` | 8.3 MB | `Drive/project/` |
| Belgian audio | ~GB | `Drive/v9/output/belgian_audio/` |
| CSS10 audio | ~GB | `Drive/v9/output/css10_audio/` |

---

## Next Steps (V11)

1. Fine-tune on real Sagitta 8kHz telephone calls
2. Add more Sagitta transcripts to the language model
3. Retrain LM on combined Belgian Dutch + Sagitta call text
