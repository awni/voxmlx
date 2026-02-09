## Title
Bug: deterministic non-incremental vs incremental divergence at token 50 (encoder path mismatch)

## What this PR adds
- Adds a focused bug dossier with reproducible commands and artifacts:
  - `docs/bug_reports/encoder_divergence_nonincremental_vs_incremental.md`
- Adds original-author-commit reproduction artifacts proving issue predates local changes:
  - `perf/audio_runs/commit-compare-e6d193e-20260209T000000Z/comparison.json`
  - `perf/audio_runs/commit-compare-e6d193e-20s-20260209T000000Z/comparison.json`
- Adds a committed 20s fixture used by the report's token-50 claim:
  - `perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_20s.wav`

## Problem statement
On the same model/audio/settings, non-incremental file decode (`generate.py` with `model.encode`) diverges from incremental decode (`encode_step`) at output token index 50.
This index is documented against the included 20s fixture (and may differ for different audio).

## Repro
See:
- `docs/bug_reports/encoder_divergence_nonincremental_vs_incremental.md`

## Key evidence
- First divergence index pinned at 50 across decoder `sliding_window` sweep (128..16384).
- Mel parity and conv parity hold through early region; first conv mismatch occurs later (pos 134).
- Divergence localizes to encoder transformer full-vs-cached history path (layer 0 with history present).
- Same signature reproduced at original-author commit `e6d193e85e84e30f26e370c66973ce287b8a9d57`.

## Why this matters
This causes non-incremental transcription quality collapse to PAD-heavy/special-token-heavy output relative to incremental path, limiting usable clip length for file transcription.

## Proposed next step for maintainers
Treat incremental encoder path as correctness baseline and enforce full-vs-incremental equivalence checks while investigating encoder full-sequence vs cached-history attention semantics.
