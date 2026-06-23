# evaluation/evaluate_greedy.py: greedy evaluation.
# Runs the baseline (V9) and final (V10) greedy WER/CER comparison.
# Usage: python evaluation/evaluate_greedy.py
import json
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from functools import partial

from config import (
    BATCH_SIZE_EVAL, V10_RESULTS_PATH,
    TRAIN_MANIFEST, VAL_MANIFEST,
)
from model.loader  import load_processor, load_model
from model.metrics import wer_metric, cer_metric
from data.prepare  import copy_audio_to_local, load_manifest, build_datasets
from data.collator import DataCollatorCTCWithPadding

EVAL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_model(model, dataset, data_collator, processor, label: str) -> dict:
    """
    Run greedy decoding on the full dataset and compute WER + CER.
    Greedy = argmax at each timestep, no language model.
    """
    model.eval().to(EVAL_DEVICE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_EVAL,
        collate_fn=data_collator,
        num_workers=0,
    )
    all_preds, all_refs = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=label):
            iv = batch['input_values'].to(EVAL_DEVICE)
            am = batch['attention_mask'].to(EVAL_DEVICE)

            logits   = model(iv, attention_mask=am).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_str = processor.batch_decode(pred_ids)

            lbl = batch['labels'].clone()
            lbl[lbl == -100] = processor.tokenizer.pad_token_id
            ref_str = processor.batch_decode(lbl, group_tokens=False)

            all_preds.extend(pred_str)
            all_refs.extend(ref_str)

    wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    cer = cer_metric.compute(predictions=all_preds, references=all_refs)
    return {'WER': round(wer * 100, 2), 'CER': round(cer * 100, 2)}


def print_comparison(baseline: dict, final_greedy: dict):
    print('\n' + '='*52)
    print('  BEFORE vs AFTER FINE-TUNING (Greedy, No LM)')
    print('='*52)
    print(f'  {"Metric":<8} {"Before (V9)":>14} {"After (V10)":>14} {"Delta":>8}')
    print('-'*52)
    for m in ['WER', 'CER']:
        b     = baseline[m]
        a     = final_greedy[m]
        delta = a - b
        sign  = '-' if delta < 0 else '+'
        print(f'  {m:<8} {b:>13.1f}% {a:>13.1f}% {sign}{abs(delta):>5.1f}%')
    print('='*52)


def main():
    processor = load_processor()
    model     = load_model()

    copy_audio_to_local()
    print('\nLoading manifests...')
    df_train = load_manifest(TRAIN_MANIFEST, 'TRAIN')
    df_val   = load_manifest(VAL_MANIFEST,   'VAL')
    _, val_dataset = build_datasets(df_train, df_val, processor)

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Baseline: V9 on the Belgian and CSS10 val set.
    print('\nRunning baseline evaluation (V9 before fine-tuning)...')
    baseline = evaluate_model(model, val_dataset, data_collator, processor,
                              'Baseline (V9)')
    print(f'\nBASELINE, V9 on Belgian and CSS10 val:')
    print(f'   WER: {baseline["WER"]}%')
    print(f'   CER: {baseline["CER"]}%')
    print(f'   (V9 own-domain reference: WER 74.5%, CER 14.1%)')

    # Final: V10 after fine-tuning.
    print('\nRunning final evaluation (V10 after fine-tuning)...')
    final_greedy = evaluate_model(model, val_dataset, data_collator, processor,
                                  'Final (V10 greedy)')

    print_comparison(baseline, final_greedy)

    # Full summary.
    print('\n' + '='*62)
    print('  FULL RESULTS SUMMARY, Sagitta V10')
    print('='*62)
    print(f'  {"Model":<28} {"WER":>10} {"CER":>10}')
    print('-'*62)
    print(f'  {"V9 baseline (greedy)":<28} {baseline["WER"]:>9.1f}% {baseline["CER"]:>9.1f}%')
    print(f'  {"V10 greedy":<28} {final_greedy["WER"]:>9.1f}% {final_greedy["CER"]:>9.1f}%')
    print('='*62)

    ft_gain = final_greedy['WER'] - baseline['WER']
    print(f'\n  Fine-tune gain : WER {ft_gain:+.1f}%')

    # Save results.
    with open(f'{V10_RESULTS_PATH}/greedy_comparison.json', 'w') as f:
        json.dump({
            'baseline_v9'  : baseline,
            'finetuned_v10': final_greedy,
        }, f, indent=2)

    print(f'\nResults saved to {V10_RESULTS_PATH}/greedy_comparison.json')


if __name__ == '__main__':
    main()
