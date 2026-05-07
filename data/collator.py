# =============================================================
# data/collator.py — Dynamic padding collator for CTC training
# =============================================================
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import Wav2Vec2Processor


@dataclass
class DataCollatorCTCWithPadding:
    """
    Collator for CTC training with dynamic padding.

    Does two things per batch:

    1. Dynamic padding
       Pads each batch to the longest example IN THAT BATCH
       (not globally). Saves memory vs padding to the longest
       clip in the entire dataset.

    2. Label masking with -100
       Replaces pad_token_id (33) in labels with -100.
       PyTorch CTC loss skips positions where label == -100.
       Without this, the model learns from padding tokens which
       produces negative loss and corrupts training.
       This fix resolved the WER > 100% bug in V9.

    Uses feature_extractor.pad() and tokenizer.pad() directly
    (not processor.pad()) to avoid the deprecated
    as_target_processor warning and ensure attention_mask
    is always returned.
    """
    processor : Wav2Vec2Processor
    padding   : Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_values': f['input_values']} for f in features]
        label_features = [{'input_ids'   : f['labels']}       for f in features]

        # Pad audio — always request attention_mask explicitly
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Pad labels using tokenizer directly
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors='pt',
        )

        # Replace pad_token_id (33) with -100 so CTC loss ignores padding
        pad_id = self.processor.tokenizer.pad_token_id
        labels = labels_batch['input_ids'].masked_fill(
            labels_batch['input_ids'] == pad_id, -100
        )
        batch['labels'] = labels
        return batch
