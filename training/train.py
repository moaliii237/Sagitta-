# training/train.py: fine-tune V9 into V10.
# Usage: python training/train.py
import json
import os
from functools import partial
from transformers import TrainingArguments, Trainer

from config import (
    V10_MODEL_PATH, V10_LOGS_PATH, V10_RESULTS_PATH,
    LEARNING_RATE, NUM_TRAIN_EPOCHS, BATCH_SIZE_TRAIN, BATCH_SIZE_EVAL,
    WARMUP_STEPS, SAVE_STEPS, EVAL_STEPS, LOGGING_STEPS, DATALOADER_WORKERS,
    TRAIN_MANIFEST, VAL_MANIFEST,
)
from model.loader     import load_processor, load_model
from model.metrics    import wer_metric, cer_metric, compute_metrics
from data.prepare     import copy_audio_to_local, load_manifest, build_datasets
from data.collator    import DataCollatorCTCWithPadding


def get_training_args() -> TrainingArguments:
    """
    TrainingArguments, the key decisions:

    learning_rate = 3e-5
        Low, to avoid catastrophic forgetting of V9's Dutch knowledge.

    fp16 = True, bf16 = False
        fp16 halves VRAM and is faster on the A100.
        bf16 caused NaN loss in V9, so it stays off.

    gradient_checkpointing = False
        Caused tensor mismatch errors in V9. The A100 has enough VRAM.

    gradient_accumulation_steps = 2
        Effective batch = 16 x 2 = 32 without loading 32 clips at once.

    group_by_length = True
        Batches similar-duration clips, so less padding waste and faster.

    load_best_model_at_end = True
        Loads the checkpoint with the lowest eval_loss, not the last one.
    """
    return TrainingArguments(
        output_dir                  = V10_MODEL_PATH,
        logging_dir                 = V10_LOGS_PATH,

        learning_rate               = LEARNING_RATE,
        num_train_epochs            = NUM_TRAIN_EPOCHS,
        warmup_steps                = WARMUP_STEPS,
        lr_scheduler_type           = 'linear',

        per_device_train_batch_size = BATCH_SIZE_TRAIN,
        per_device_eval_batch_size  = BATCH_SIZE_EVAL,
        gradient_accumulation_steps = 2,
        gradient_checkpointing      = False,
        dataloader_num_workers      = DATALOADER_WORKERS,

        fp16                        = True,
        bf16                        = False,

        evaluation_strategy         = 'steps',
        eval_steps                  = EVAL_STEPS,
        save_strategy               = 'steps',
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = 'eval_loss',
        greater_is_better           = False,

        logging_steps               = LOGGING_STEPS,
        report_to                   = ['none'],

        group_by_length             = True,
        seed                        = 42,
    )


def main():
    # Load processor and model.
    processor = load_processor()
    model     = load_model()

    # Prepare data.
    copy_audio_to_local()
    print('\nLoading manifests...')
    df_train = load_manifest(TRAIN_MANIFEST, 'TRAIN')
    df_val   = load_manifest(VAL_MANIFEST,   'VAL')
    train_dataset, val_dataset = build_datasets(df_train, df_val, processor)

    # Collator and metrics.
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    metrics_fn    = partial(compute_metrics, processor=processor)

    # Training arguments.
    training_args = get_training_args()

    # Build the trainer.
    # Note: tokenizer=processor.feature_extractor is intentional.
    # The Trainer uses this for padding during eval, and for audio
    # models that means the feature extractor, not the text tokenizer.
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        tokenizer       = processor.feature_extractor,
        data_collator   = data_collator,
        compute_metrics = metrics_fn,
    )

    steps_per_epoch = len(train_dataset) // (BATCH_SIZE_TRAIN * 2)
    print(f'\nTrainer ready')
    print(f'   Steps per epoch : {steps_per_epoch:,}')
    print(f'   Total steps     : {steps_per_epoch * NUM_TRAIN_EPOCHS:,}')
    print('\nStarting fine-tuning...')

    train_result = trainer.train()

    print('\nTraining complete')
    print(f'   Runtime : {train_result.metrics["train_runtime"] / 60:.1f} min')
    print(f'   Loss    : {train_result.metrics["train_loss"]:.4f}')

    # Save model and processor to Drive.
    trainer.save_model(V10_MODEL_PATH)
    processor.save_pretrained(V10_MODEL_PATH)

    with open(f'{V10_RESULTS_PATH}/train_metrics.json', 'w') as f:
        json.dump(train_result.metrics, f, indent=2)

    print(f'\nModel saved to: {V10_MODEL_PATH}')
    print('Files saved:')
    for fname in os.listdir(V10_MODEL_PATH):
        fpath = os.path.join(V10_MODEL_PATH, fname)
        size  = os.path.getsize(fpath) / 1e6
        print(f'  {fname:<40} {size:.1f} MB')


if __name__ == '__main__':
    main()
