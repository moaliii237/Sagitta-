# lm/tune_alpha_beta.py: grid search for the best alpha and beta.
# Run this after training to find the best LM decoder settings.
# Usage: python lm/tune_alpha_beta.py
# Result I got: alpha=0.3, beta=2.0, WER 12.3%
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder

from config import (
    SAGITTA_ARPA, HOT_WORDS_PATH, V10_RESULTS_PATH,
    TRAIN_MANIFEST, VAL_MANIFEST,
)
from model.loader  import load_processor, load_model
from model.metrics import wer_metric
from data.prepare  import copy_audio_to_local, load_manifest, build_datasets
from data.collator import DataCollatorCTCWithPadding
from evaluation.evaluate_lm import build_decoder

EVAL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def collect_logits(model, val_dataset, data_collator, processor, n=500):
    """
    Collect logits from the first N val examples.
    Stores each example separately (not stacked) because
    clips have different lengths.
    """
    loader = DataLoader(
        val_dataset.select(range(min(n, len(val_dataset)))),
        batch_size=4,
        collate_fn=data_collator,
        num_workers=0,
    )

    print(f'Collecting logits on {n} val examples...')
    model.eval().to(EVAL_DEVICE)

    subset_logits = []
    subset_refs   = []

    for batch in tqdm(loader, desc='Collecting logits'):
        iv = batch['input_values'].to(EVAL_DEVICE)
        am = batch['attention_mask'].to(EVAL_DEVICE)
        with torch.no_grad():
            logits = model(iv, attention_mask=am).logits

        logits_np = logits.cpu().numpy()
        for i in range(logits_np.shape[0]):
            subset_logits.append(logits_np[i])

        lbl = batch['labels'].clone()
        lbl[lbl == -100] = processor.tokenizer.pad_token_id
        subset_refs.extend(processor.batch_decode(lbl, group_tokens=False))

    print(f'Collected {len(subset_logits)} examples')
    return subset_logits, subset_refs


def run_grid_search(subset_logits, subset_refs, vocab_for_decoder, hot_words):
    """Grid search over alpha and beta values."""
    results = []
    print('\nRunning alpha/beta grid search...\n')

    for alpha in [0.3, 0.5, 0.7, 0.9]:
        for beta in [0.5, 1.0, 1.5, 2.0]:
            d = build_ctcdecoder(
                labels           = vocab_for_decoder,
                kenlm_model_path = SAGITTA_ARPA,
                unigrams         = hot_words,
                alpha            = alpha,
                beta             = beta,
            )
            preds = [d.decode(subset_logits[i], beam_width=50)
                     for i in range(len(subset_logits))]
            w = wer_metric.compute(predictions=preds, references=subset_refs) * 100
            results.append((alpha, beta, w))
            print(f'  alpha={alpha}  beta={beta}  WER={w:.1f}%')

    best = min(results, key=lambda x: x[2])
    print(f'\nBest: alpha={best[0]}  beta={best[1]}  WER={best[2]:.1f}%')
    print('\nUpdate LM_ALPHA and LM_BETA in config.py with these values.')
    print('Then re-run evaluation/evaluate_lm.py for final results.')
    return results, best


def main():
    processor = load_processor()
    model     = load_model()

    copy_audio_to_local()
    print('\nLoading manifests...')
    df_train = load_manifest(TRAIN_MANIFEST, 'TRAIN')
    df_val   = load_manifest(VAL_MANIFEST,   'VAL')
    _, val_dataset = build_datasets(df_train, df_val, processor)

    data_collator = DataCollatorCTCWithPadding(processor=processor)
    _, vocab_for_decoder, hot_words = build_decoder(processor)

    subset_logits, subset_refs = collect_logits(
        model, val_dataset, data_collator, processor, n=500
    )

    results, best = run_grid_search(
        subset_logits, subset_refs, vocab_for_decoder, hot_words
    )


if __name__ == '__main__':
    main()
