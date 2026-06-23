# utils/verify.py: check the GPU and that all required paths exist.
# Run this first, before anything else.
# Usage: python utils/verify.py
import os
import torch
from config import (
    V9_MODEL_PATH, V9_PROCESSOR_PATH,
    TRAIN_MANIFEST, VAL_MANIFEST,
    BELGIAN_AUDIO_DRIVE, CSS10_AUDIO_DRIVE,
)


def verify_gpu():
    print('=== GPU ===')
    if torch.cuda.is_available():
        print(f'  {torch.cuda.get_device_name(0)}')
        print(f'  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM')
    else:
        print('  No GPU detected, training will be very slow on CPU.')


def verify_paths():
    print('\n=== PATHS ===')
    checks = {
        'V9 model'      : V9_MODEL_PATH,
        'V9 processor'  : V9_PROCESSOR_PATH,
        'Train manifest': TRAIN_MANIFEST,
        'Val manifest'  : VAL_MANIFEST,
        'Belgian audio' : BELGIAN_AUDIO_DRIVE,
        'CSS10 audio'   : CSS10_AUDIO_DRIVE,
    }
    all_ok = True
    for label, path in checks.items():
        ok = os.path.exists(path)
        status = 'OK     ' if ok else 'MISSING'
        if not ok:
            all_ok = False
        print(f'  [{status}]  {label}: {path}')

    if all_ok:
        print('\nAll paths verified, ready to train.')
    else:
        print('\nFix the missing paths in config.py before continuing.')


if __name__ == '__main__':
    verify_gpu()
    verify_paths()
