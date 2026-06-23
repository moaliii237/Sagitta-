# evaluation/evaluate_sagitta.py: V10 vs WhisperX on real calls.
# Transcribes real 8kHz call center audio and compares it against
# the WhisperX reference transcripts.
# Usage: python evaluation/evaluate_sagitta.py
import os
import re
import json
import librosa
import torch
from jiwer import wer as jwer
from tqdm.auto import tqdm

from config import (
    SAGITTA_AUDIO_PATH, SAGITTA_TRANSCRIPT_PATH,
    TARGET_SR, MAX_CHUNK_SEC, V10_RESULTS_PATH,
)
from model.loader      import load_processor, load_model
from evaluation.evaluate_lm import build_decoder

EVAL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def find_matched_pairs():
    """
    Find all audio files that have a matching WhisperX transcript.
    Transcripts live at: transcripts/{id}/{id}_full.txt
    """
    pairs = []
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac']

    for audio_file in sorted(os.listdir(SAGITTA_AUDIO_PATH)):
        stem, ext = os.path.splitext(audio_file)
        if ext.lower() not in audio_extensions:
            continue
        transcript_path = os.path.join(
            SAGITTA_TRANSCRIPT_PATH, stem, f'{stem}_full.txt'
        )
        if os.path.exists(transcript_path):
            pairs.append({
                'id'        : stem,
                'audio'     : os.path.join(SAGITTA_AUDIO_PATH, audio_file),
                'transcript': transcript_path,
            })

    print(f'Matched {len(pairs)} audio + transcript pairs\n')
    return pairs


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def transcribe_audio(audio_path: str, model, processor, decoder,
                     use_lm: bool = True):
    """
    Transcribe a long audio file by splitting it into chunks.
    MAX_CHUNK_SEC = 30s is safe for wav2vec2 on the A100.
    The chunks are joined and cleaned after decoding.
    I free GPU memory between chunks to avoid OOM.
    """
    speech = librosa.load(audio_path, sr=TARGET_SR, mono=True)[0].astype('float32')

    chunk_size  = MAX_CHUNK_SEC * TARGET_SR
    num_chunks  = (len(speech) + chunk_size - 1) // chunk_size
    greedy_parts, lm_parts = [], []

    for i in range(num_chunks):
        chunk = speech[i * chunk_size : (i + 1) * chunk_size]

        inputs = processor(
            chunk,
            sampling_rate=TARGET_SR,
            return_tensors='pt',
            return_attention_mask=True,
        )

        with torch.no_grad():
            logits = model(
                inputs.input_values.to(EVAL_DEVICE),
                attention_mask=inputs.attention_mask.to(EVAL_DEVICE),
            ).logits

        # Greedy decode.
        greedy_ids = torch.argmax(logits, dim=-1)
        greedy_parts.append(processor.batch_decode(greedy_ids)[0])

        # LM decode.
        if use_lm:
            logits_np = logits.squeeze(0).cpu().numpy()
            lm_parts.append(decoder.decode(logits_np, beam_width=100))

        # Free GPU memory between chunks.
        del logits, inputs
        torch.cuda.empty_cache()

    greedy_full = clean_text(' '.join(greedy_parts))
    lm_full     = clean_text(' '.join(lm_parts)) if use_lm else greedy_full

    return greedy_full, lm_full


def main():
    processor           = load_processor()
    model               = load_model()
    decoder, _, _       = build_decoder(processor)
    model.eval().to(EVAL_DEVICE)

    pairs   = find_matched_pairs()
    results = []

    for pair in tqdm(pairs, desc='Transcribing Sagitta calls'):
        with open(pair['transcript'], 'r') as f:
            whisper_text = clean_text(f.read())

        try:
            greedy_text, lm_text = transcribe_audio(
                pair['audio'], model, processor, decoder
            )

            wer_greedy = jwer(whisper_text, greedy_text) * 100
            wer_lm     = jwer(whisper_text, lm_text)     * 100

            results.append({
                'id'         : pair['id'],
                'whisper_ref': whisper_text,
                'v10_greedy' : greedy_text,
                'v10_lm'     : lm_text,
                'wer_greedy' : wer_greedy,
                'wer_lm'     : wer_lm,
                'error'      : None,
            })

        except Exception as e:
            print(f'  Skipped {pair["id"][:40]}: {e}')
            torch.cuda.empty_cache()
            results.append({'id': pair['id'], 'error': str(e)})

    # Summary.
    valid      = [r for r in results if r['error'] is None]
    avg_greedy = sum(r['wer_greedy'] for r in valid) / len(valid)
    avg_lm     = sum(r['wer_lm']     for r in valid) / len(valid)

    print('\n' + '='*62)
    print('  V10 vs WhisperX, real 8kHz calls')
    print('='*62)
    print(f'  Calls evaluated : {len(valid)} / {len(pairs)}')
    print(f'  {"V10 Greedy":<20} avg WER vs WhisperX: {avg_greedy:.1f}%')
    print(f'  {"V10 + LM":<20} avg WER vs WhisperX: {avg_lm:.1f}%')
    print('='*62)

    # Per-call breakdown.
    print('\nPer-call breakdown:')
    print(f'  {"ID":36} {"Greedy":>8} {"LM":>8}')
    print('-'*56)
    for r in valid:
        short_id = r['id'][:32] + '..' if len(r['id']) > 32 else r['id']
        print(f'  {short_id:<36} {r["wer_greedy"]:>7.1f}% {r["wer_lm"]:>7.1f}%')

    # Sample transcriptions.
    print('\n\nSample transcriptions (first 3 calls):')
    print('='*62)
    for r in valid[:3]:
        print(f'\nID        : {r["id"][:50]}')
        print(f'WhisperX  : {r["whisper_ref"][:300]}')
        print(f'V10 Greedy: {r["v10_greedy"][:300]}')
        print(f'V10 LM    : {r["v10_lm"][:300]}')
        print(f'WER Greedy: {r["wer_greedy"]:.1f}%  |  WER LM: {r["wer_lm"]:.1f}%')
        print('-'*62)

    # Save.
    out_path = f'{V10_RESULTS_PATH}/sagitta_vs_whisper.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {out_path}')

    # Note on the results.
    print('\nNote on these results:')
    print('   A WER around 52% is expected, the model was trained on 16kHz clean audio.')
    print('   The calls are 8kHz telephone audio, which is a different acoustic domain.')
    print('   Next step: fine-tune on real 8kHz calls for V11.')


if __name__ == '__main__':
    main()
