# =============================================================
# config.py — All paths and hyperparameters for Sagitta V10
# This is the ONLY file you need to edit if paths change.
# =============================================================
import os

# ── Root folder on Google Drive ──────────────────────────────
# When running in Colab, Drive is mounted at /content/drive
# When running locally, point this to wherever you synced Drive
DRIVE_ROOT = '/content/drive/MyDrive/ArcumSagitta'

# ── Input: already-trained V9 model ──────────────────────────
# V9 was trained on CGN telephone speech + Sagitta calls
V9_MODEL_PATH     = f'{DRIVE_ROOT}/v9/models/gronlp'
V9_PROCESSOR_PATH = f'{DRIVE_ROOT}/v9/processor/gronlp'

# ── Output: where V10 will be saved ──────────────────────────
V10_MODEL_PATH   = f'{DRIVE_ROOT}/v10/models/gronlp_belgian_css10_finetuned'
V10_LOGS_PATH    = f'{DRIVE_ROOT}/v10/logs'
V10_RESULTS_PATH = f'{DRIVE_ROOT}/v10/results'

# ── Language model paths ──────────────────────────────────────
LM_DIR       = f'{DRIVE_ROOT}/lm'
LM_PATH      = f'{LM_DIR}/4gram.arpa'
SAGITTA_ARPA = f'{DRIVE_ROOT}/project/dutch_sagitta_lm.arpa'
SAGITTA_BIN  = f'{DRIVE_ROOT}/project/dutch_sagitta_lm.bin'
HOT_WORDS_PATH = f'{DRIVE_ROOT}/project/sagitta_hot_words.txt'

# ── Manifests ─────────────────────────────────────────────────
# combined_train.csv and combined_val.csv merge:
# Belgian Dutch (14,985 clips) + CSS10 Dutch (6,431 clips)
MANIFEST_DIR   = f'{DRIVE_ROOT}/v9/output/manifests'
TRAIN_MANIFEST = f'{MANIFEST_DIR}/combined_train.csv'
VAL_MANIFEST   = f'{MANIFEST_DIR}/combined_val.csv'

# ── Audio folders on Drive ────────────────────────────────────
BELGIAN_AUDIO_DRIVE = f'{DRIVE_ROOT}/v9/output/belgian_audio'
CSS10_AUDIO_DRIVE   = f'{DRIVE_ROOT}/v9/output/css10_audio'

# ── Local SSD copies (faster than Drive during training) ──────
LOCAL_BELGIAN = '/content/belgian_audio'
LOCAL_CSS10   = '/content/css10_audio'

# ── Windows → local path remapping ───────────────────────────
# Manifest CSVs were created on Windows — paths look like:
# D:\dutch_asr_prep\output\belgian_audio\clip.wav
# We remap them to local SSD paths after copying from Drive
WINDOWS_PATH_MAP = {
    r'D:\dutch_asr_prep\output\belgian_audio': LOCAL_BELGIAN,
    r'D:\dutch_asr_prep\output\css10_audio'  : LOCAL_CSS10,
}

# ── Sagitta real call audio ───────────────────────────────────
SAGITTA_AUDIO_PATH      = f'{DRIVE_ROOT}/datasets/sagitta voice/15'
SAGITTA_TRANSCRIPT_PATH = f'{DRIVE_ROOT}/datasets/sagitta voice/transcripts'

# ── Audio settings ────────────────────────────────────────────
TARGET_SR     = 16_000   # model expects 16kHz mono
MAX_AUDIO_SEC = 10.0     # clips longer than this are dropped (OOM protection)
MAX_CHUNK_SEC = 30       # max chunk for long-audio inference

# ── Training hyperparameters ──────────────────────────────────
# Low LR (3e-5): model already knows Dutch from V9.
# High LR would cause catastrophic forgetting.
LEARNING_RATE      = 3e-5
NUM_TRAIN_EPOCHS   = 5
BATCH_SIZE_TRAIN   = 16
BATCH_SIZE_EVAL    = 8
WARMUP_STEPS       = 200
SAVE_STEPS         = 500
EVAL_STEPS         = 500
LOGGING_STEPS      = 50
DATALOADER_WORKERS = 4

# ── LM decoder settings (best found via grid search) ─────────
LM_ALPHA = 0.3   # LM weight — low because acoustic model is strong
LM_BETA  = 2.0   # word insertion bonus — helps Dutch compound words

# ── Create output directories ─────────────────────────────────
for p in [V10_MODEL_PATH, V10_LOGS_PATH, V10_RESULTS_PATH, LM_DIR]:
    os.makedirs(p, exist_ok=True)
