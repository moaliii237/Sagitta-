# =============================================================
# model/metrics.py — WER and CER metrics for the Trainer
# =============================================================
import evaluate
import numpy as np

wer_metric = evaluate.load('wer')
cer_metric = evaluate.load('cer')


def compute_metrics(pred, processor):
    """
    Called by the HuggingFace Trainer after each eval step.

    WER (Word Error Rate): fraction of words wrong.
    CER (Character Error Rate): fraction of characters wrong.
    CER is more informative for Dutch compound words like
    'warmtepomp' — one wrong char makes the whole word wrong in WER.

    Decoding steps:
    1. argmax on logits → highest-probability token per timestep
    2. batch_decode → CTC collapse (removes repeated + blank tokens)
    3. Replace -100 labels with pad_token_id before decoding refs
    """
    pred_ids  = np.argmax(pred.predictions, axis=-1)
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id before decoding references
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {
        'wer': round(wer * 100, 2),
        'cer': round(cer * 100, 2),
    }
