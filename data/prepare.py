# =============================================================
# data/prepare.py — Audio copy, manifest loading, preprocessing
# Usage: python data/prepare.py
# =============================================================
import os
import shutil
import librosa
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import Wav2Vec2Processor
from config import (
    BELGIAN_AUDIO_DRIVE, CSS10_AUDIO_DRIVE,
    LOCAL_BELGIAN, LOCAL_CSS10,
    WINDOWS_PATH_MAP,
    TRAIN_MANIFEST, VAL_MANIFEST,
    TARGET_SR, MAX_AUDIO_SEC,
)


# ── Step 1: Copy audio from Drive to local SSD ────────────────
def copy_audio_to_local():
    """
    Copy audio from Google Drive to local Colab SSD.
    Local disk is much faster than Drive for repeated reads.
    Skips copy if folder already exists (safe to re-run).
    """
    if not os.path.exists(LOCAL_BELGIAN):
        print('Copying Belgian audio to local disk...')
        shutil.copytree(BELGIAN_AUDIO_DRIVE, LOCAL_BELGIAN)
        print('✅ Belgian audio done')
    else:
        print('✅ Belgian audio already on local disk')

    if not os.path.exists(LOCAL_CSS10):
        print('Copying CSS10 audio to local disk...')
        shutil.copytree(CSS10_AUDIO_DRIVE, LOCAL_CSS10)
        print('✅ CSS10 audio done')
    else:
        print('✅ CSS10 audio already on local disk')


# ── Step 2: Load and clean manifests ─────────────────────────
def remap_path(windows_path: str) -> str:
    """
    Remap Windows-style paths from the manifest CSVs to
    local SSD paths. Manifest was created on Windows so paths
    look like: D:\dutch_asr_prep\output\belgian_audio\clip.wav
    """
    p = str(windows_path).replace('\\', '/')
    for win_prefix, local_prefix in WINDOWS_PATH_MAP.items():
        win_norm = win_prefix.replace('\\', '/')
        if p.startswith(win_norm):
            return local_prefix + p[len(win_norm):]
    return p  # already a local path or unrecognised


def load_manifest(csv_path: str, split: str) -> pd.DataFrame:
    """
    Load a CSV manifest, remap paths, clean text, drop bad rows.
    Drops: empty transcripts, clips longer than MAX_AUDIO_SEC.
    """
    df = pd.read_csv(csv_path)
    print(f'  {split}: {len(df):,} rows loaded')

    df['audio_path'] = df['audio_path'].apply(remap_path)
    df['text']       = df['text'].astype(str).str.lower().str.strip()

    before = len(df)
    df = df[df['text'].str.len() > 0].copy()
    print(f'  {split}: dropped {before - len(df)} empty-text rows')

    if 'duration' in df.columns:
        before = len(df)
        df = df[df['duration'] <= MAX_AUDIO_SEC].copy()
        print(f'  {split}: dropped {before - len(df)} clips > {MAX_AUDIO_SEC}s')

    exist = df['audio_path'].iloc[:5].apply(os.path.exists).tolist()
    print(f'  {split}: first 5 files exist? {exist}')

    return df.reset_index(drop=True)


# ── Step 3: Preprocess audio + text into model-ready arrays ──
def load_audio(path: str) -> np.ndarray:
    """Load audio at 16kHz mono float32."""
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


def df_to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict({
        'audio_path': df['audio_path'].tolist(),
        'text'      : df['text'].tolist(),
    })


def make_preprocess_fn(processor: Wav2Vec2Processor):
    """
    Returns a batch preprocessing function bound to the processor.

    For audio: produces input_values (normalised float array)
    and attention_mask (1=real audio, 0=padding).

    For text: tokenises Dutch text into integer IDs.
    e.g. 'hallo' → [8, 1, 12, 12, 15]

    NOTE: Do NOT call set_format('torch') after this.
    NumPy 2.0 has a bug with variable-length arrays in that mode.
    The collator handles numpy → torch conversion instead.
    """
    def preprocess_batch(batch):
        speech = [load_audio(p) for p in batch['audio_path']]

        inputs = processor(
            speech,
            sampling_rate=TARGET_SR,
            return_tensors='np',
            padding=True,
            return_attention_mask=True,
        )
        labels = processor(
            text=batch['text'],
            return_tensors='np',
            padding=True,
        )

        batch['input_values']   = inputs.input_values
        batch['attention_mask'] = inputs.attention_mask
        batch['labels']         = labels.input_ids
        return batch

    return preprocess_batch


def build_datasets(df_train: pd.DataFrame, df_val: pd.DataFrame,
                   processor: Wav2Vec2Processor):
    """
    Convert dataframes into preprocessed HuggingFace Datasets.
    Results are cached to disk — re-running is instant.
    First run takes ~45 min for 18k clips.
    """
    print('Building datasets (cached after first run)...\n')
    preprocess_batch = make_preprocess_fn(processor)

    train_dataset = df_to_dataset(df_train).map(
        preprocess_batch,
        batched=True,
        batch_size=32,
        num_proc=1,   # keep at 1 — num_proc=2 crashes Colab
        remove_columns=['audio_path', 'text'],
        desc='Preprocessing train',
    )
    val_dataset = df_to_dataset(df_val).map(
        preprocess_batch,
        batched=True,
        batch_size=32,
        num_proc=1,
        remove_columns=['audio_path', 'text'],
        desc='Preprocessing val',
    )

    print(f'\n✅ Train: {len(train_dataset):,} | Val: {len(val_dataset):,}')
    print(f'   Columns: {train_dataset.column_names}')
    return train_dataset, val_dataset


if __name__ == '__main__':
    copy_audio_to_local()
    print('\nLoading manifests...')
    df_train = load_manifest(TRAIN_MANIFEST, 'TRAIN')
    df_val   = load_manifest(VAL_MANIFEST,   'VAL')
    print(f'\n✅ Train: {len(df_train):,} | Val: {len(df_val):,}')
