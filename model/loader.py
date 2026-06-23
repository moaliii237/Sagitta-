# model/loader.py: load the processor and V9 model weights.
import os
import shutil
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
from config import V9_MODEL_PATH, V9_PROCESSOR_PATH


def load_processor() -> Wav2Vec2Processor:
    """
    Load the Wav2Vec2 processor from the V9 checkpoint.

    I build it manually from two parts instead of using
    Wav2Vec2Processor.from_pretrained() because the V9
    processor_config.json has a duplicate feature_extractor key that
    causes a TypeError when it loads automatically.

    I also fix the filename: HuggingFace 4.40 expects
    preprocessor_config.json but V9 saved processor_config.json.
    """
    proc_path = V9_PROCESSOR_PATH

    # Fix the filename if needed.
    src = f'{proc_path}/processor_config.json'
    dst = f'{proc_path}/preprocessor_config.json'
    if not os.path.exists(dst) and os.path.exists(src):
        shutil.copy(src, dst)
        print('Copied processor_config.json to preprocessor_config.json')

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(proc_path)
    tokenizer         = Wav2Vec2CTCTokenizer.from_pretrained(proc_path)
    processor         = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    print(f'Processor loaded')
    print(f'   Vocab size    : {tokenizer.vocab_size}')
    print(f'   Sampling rate : {feature_extractor.sampling_rate}')
    print(f'   pad_token_id  : {tokenizer.pad_token_id}')
    return processor


def load_model() -> Wav2Vec2ForCTC:
    """
    Load the V9 model weights into Wav2Vec2ForCTC and freeze the CNN.

    The architecture has three parts:
    1. CNN feature encoder (frozen), low-level acoustic features
    2. Transformer layers (trainable), higher-level language patterns
    3. CTC head (trainable), token probability at each time step

    Why freeze the CNN:
    - Fewer parameters, so faster training
    - The CNN acoustic learning from V9 is kept
    - The Belgian and CSS10 data only needs to improve language patterns
    """
    print('Loading V9 model...')
    model = Wav2Vec2ForCTC.from_pretrained(V9_MODEL_PATH)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params    : {total:,}')
    print(f'  Trainable params: {trainable:,}')

    # Freeze the CNN feature extractor.
    model.freeze_feature_encoder()
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Frozen  (CNN)   : {frozen:,}')
    print(f'  Trainable now   : {trainable:,}')
    print('\nModel ready')
    return model
