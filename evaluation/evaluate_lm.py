# =============================================================
# evaluation/evaluate_lm.py — Build LM decoder + run LM eval
# Usage: python evaluation/evaluate_lm.py
# =============================================================
import os
import json
import torch
import numpy as np
import librosa
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from jiwer import wer as jwer
from pyctcdecode import build_ctcdecoder

from config import (
    SAGITTA_ARPA, HOT_WORDS_PATH, V10_RESULTS_PATH,
    TARGET_SR, BATCH_SIZE_EVAL, LM_ALPHA, LM_BETA,
    TRAIN_MANIFEST, VAL_MANIFEST,
)
from model.loader  import load_processor, load_model
from model.metrics import wer_metric, cer_metric
from data.prepare  import copy_audio_to_local, load_manifest, build_datasets
from data.collator import DataCollatorCTCWithPadding

EVAL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def check_lm_files():
    """Verify the ARPA language model file exists."""
    print('=== Checking Sagitta LM files ===')
    for path in [SAGITTA_ARPA]:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1e6
            print(f'  ✅ {os.path.basename(path)}: {size:.1f} MB')
        else:
            print(f'  ❌ MISSING: {path}')
            print(f'     Run the KenLM training steps first')
            raise FileNotFoundError(f'LM file not found: {path}')


def build_decoder(processor):
    """
    Build the CTC beam search decoder with Sagitta LM.

    Alpha (LM weight): how much to trust the LM vs acoustic model.
      Best found via grid search: 0.3 (low — acoustic model is strong)

    Beta (word insertion bonus): rewards longer, complete words.
      Helps Dutch compounds like 'warmtepomp'.
      Best found via grid search: 2.0

    Unigrams = domain vocabulary (1081 Sagitta call center words).
    This is the hotwords mechanism — decoder prefers these words.

    Vocab must be ordered by token ID with | replaced by space
    and [PAD] replaced by empty string.
    """
    # Load domain hot words
    with open(HOT_WORDS_PATH, 'r') as f:
        hot_words = [line.strip() for line in f if line.strip()]
    print(f'Loaded {len(hot_words)} domain words as unigrams')
    print(f'Top 10: {hot_words[:10]}')

    # Build vocab ordered by token ID
    vocab_dict        = processor.tokenizer.get_vocab()
    sorted_vocab      = sorted(vocab_dict.items(), key=lambda x: x[1])
    vocab_list        = [token for token, _ in sorted_vocab]
    vocab_for_decoder = [
        ' '  if token == '|'
        else ''   if token == '[PAD]'
        else token
        for token in vocab_list
    ]

    decoder = build_ctcdecoder(
        labels           = vocab_for_decoder,
        kenlm_model_path = SAGITTA_ARPA,
        unigrams         = hot_words,
        alpha            = LM_ALPHA,
        beta             = LM_BETA,
    )

    print(f'\n✅ Decoder ready')
    print(f'   LM       : {os.path.basename(SAGITTA_ARPA)}')
    print(f'   Unigrams : {len(hot_words)} domain words')
    print(f'   Alpha    : {LM_ALPHA}')
    print(f'   Beta     : {LM_BETA}')
    return decoder, vocab_for_decoder, hot_words


def quick_lm_test(model, processor, decoder, df_val):
    """Single-example sanity check: greedy vs LM on one val clip."""
    model.eval().to(EVAL_DEVICE)
    row      = df_val.iloc[0]
    ref_text = row['text']
    speech   = librosa.load(row['audio_path'], sr=TARGET_SR, mono=True)[0].astype('float32')

    inputs = processor(
        speech,
        sampling_rate=TARGET_SR,
        return_tensors='pt',
        return_attention_mask=True,
    )

    with torch.no_grad():
        logits = model(
            inputs.input_values.to(EVAL_DEVICE),
            attention_mask=inputs.attention_mask.to(EVAL_DEVICE),
        ).logits

    greedy_ids  = torch.argmax(logits, dim=-1)
    greedy_text = processor.batch_decode(greedy_ids)[0]

    logits_np = logits.squeeze(0).cpu().numpy()
    lm_text   = decoder.decode(logits_np, beam_width=100)

    print(f'\nReference  : {ref_text}')
    print(f'Greedy     : {greedy_text}')
    print(f'LM + Uni   : {lm_text}')
    print(f'\nGreedy WER : {jwer(ref_text, greedy_text)*100:.1f}%')
    print(f'LM WER     : {jwer(ref_text, lm_text)*100:.1f}%')


def run_full_lm_eval(model, val_dataset, data_collator, processor, decoder):
    """Full val set greedy vs LM evaluation."""
    model.eval().to(EVAL_DEVICE)
    all_greedy, all_lm, all_refs = [], [], []

    loader = DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=data_collator,
        num_workers=0,
    )

    for batch in tqdm(loader, desc='Greedy + LM eval'):
        iv = batch['input_values'].to(EVAL_DEVICE)
        am = batch['attention_mask'].to(EVAL_DEVICE)

        with torch.no_grad():
            logits = model(iv, attention_mask=am).logits

        # Greedy
        greedy_ids  = torch.argmax(logits, dim=-1)
        greedy_strs = processor.batch_decode(greedy_ids)
        all_greedy.extend(greedy_strs)

        # LM + unigrams — one example at a time
        logits_np = logits.cpu().numpy()
        for i in range(logits_np.shape[0]):
            all_lm.append(decoder.decode(logits_np[i], beam_width=100))

        # References
        lbl = batch['labels'].clone()
        lbl[lbl == -100] = processor.tokenizer.pad_token_id
        all_refs.extend(processor.batch_decode(lbl, group_tokens=False))

    g_wer = wer_metric.compute(predictions=all_greedy, references=all_refs) * 100
    g_cer = cer_metric.compute(predictions=all_greedy, references=all_refs) * 100
    l_wer = wer_metric.compute(predictions=all_lm,     references=all_refs) * 100
    l_cer = cer_metric.compute(predictions=all_lm,     references=all_refs) * 100

    print('\n' + '='*58)
    print('  Greedy vs LM + Unigrams — V10 Val Set')
    print('='*58)
    print(f'  {"":14} {"WER":>10} {"CER":>10}')
    print('-'*58)
    print(f'  {"Greedy":14} {g_wer:>9.1f}% {g_cer:>9.1f}%')
    print(f'  {"LM + Unigrams":14} {l_wer:>9.1f}% {l_cer:>9.1f}%')
    print(f'  {"Δ":14} {l_wer-g_wer:>+9.1f}% {l_cer-g_cer:>+9.1f}%')
    print('='*58)

    return g_wer, g_cer, l_wer, l_cer


def main():
    check_lm_files()

    processor = load_processor()
    model     = load_model()

    copy_audio_to_local()
    print('\nLoading manifests...')
    df_train = load_manifest(TRAIN_MANIFEST, 'TRAIN')
    df_val   = load_manifest(VAL_MANIFEST,   'VAL')
    _, val_dataset = build_datasets(df_train, df_val, processor)

    data_collator = DataCollatorCTCWithPadding(processor=processor)
    decoder, _, _ = build_decoder(processor)

    # Quick single-example test
    print('\n--- Quick LM test on one val clip ---')
    quick_lm_test(model, processor, decoder, df_val)

    # Full val set evaluation
    print('\n--- Full val set evaluation ---')
    g_wer, g_cer, l_wer, l_cer = run_full_lm_eval(
        model, val_dataset, data_collator, processor, decoder
    )

    # Save results
    with open(f'{V10_RESULTS_PATH}/lm_comparison.json', 'w') as f:
        json.dump({
            'greedy'        : {'WER': g_wer, 'CER': g_cer},
            'lm_sagitta_uni': {'WER': l_wer, 'CER': l_cer},
        }, f, indent=2)
    print(f'\n✅ Results saved to {V10_RESULTS_PATH}/lm_comparison.json')


if __name__ == '__main__':
    main()
