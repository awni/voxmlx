# Bug Report: Non-incremental vs Incremental Encoder Divergence (first token mismatch at index 50)

## Summary
On Voxtral realtime MLX inference, non-incremental file decoding (`generate.py` path using `model.encode`) and incremental decoding (`encode_step` path) diverge deterministically at output token index **50** on the same audio/model/settings.

Important: the `50` index is tied to the included 20s fixture below and may differ for other audio.

This appears to be an **encoder transformer full-sequence vs cached-history semantic mismatch**, not a decoder sliding-window issue.

## Scope
- Model: `mlx-community/Voxtral-Mini-4B-Realtime-6bit`
- Audio fixture (included in this PR): `perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_20s.wav`
- Temperature: `0.0` (greedy)
- STFT backend: `dft` (for parity runs)

## Why this report is high confidence
The divergence was reproduced on both:
- current branch state (instrumented runs), and
- original-author commit `e6d193e85e84e30f26e370c66973ce287b8a9d57`.

So this is not introduced by recent local changes.

## Minimal Repro (self-contained, 20s fixture)
```bash
PYTHONPATH=. .venv313/bin/python - <<'PY'
from pathlib import Path
import mlx.core as mx
from voxmlx import load_model, _build_prompt_tokens
from voxmlx.audio import load_audio, pad_audio, log_mel_spectrogram, log_mel_spectrogram_step, SAMPLES_PER_TOKEN
from voxmlx.cache import RotatingKVCache

def first_diff(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y: return i
    return None if len(a)==len(b) else min(len(a),len(b))

def inc_embeds(model, audio_padded):
    at=ct1=ct2=ec=ds=None
    parts=[]
    for i in range(0,len(audio_padded),SAMPLES_PER_TOKEN):
        mel,at=log_mel_spectrogram_step(audio_padded[i:i+SAMPLES_PER_TOKEN],at)
        out,ct1,ct2,ec,ds=model.encode_step(mel,ct1,ct2,ec,ds)
        if out is not None and out.shape[0]>0: parts.append(out)
    return mx.concatenate(parts,axis=0)

def decode(model, embeds, prompt_tokens, n_delay, eos, sw=8192):
    t=model.time_embedding(mx.array([n_delay],dtype=mx.float32))
    pre=len(prompt_tokens)
    text=model.language_model.embed(mx.array([prompt_tokens]))[0]
    cache=[RotatingKVCache(sw) for _ in range(len(model.language_model.layers))]
    logits=model.decode((text+embeds[:pre])[None,:,:],t,"causal",cache)
    mx.eval(logits,*[x for c in cache for x in (c.keys,c.values)])
    y=mx.argmax(logits[0,-1:],axis=-1).squeeze()
    out=[]
    for pos in range(pre, embeds.shape[0]):
        tok=model.language_model.embed(y.reshape(1,1))[0,0]
        nlogits=model.decode((embeds[pos]+tok)[None,None,:],t,mask=None,cache=cache)
        y_next=mx.argmax(nlogits[0,-1:],axis=-1).squeeze()
        tid=int(y.item()); out.append(tid); y=y_next
    return out

model,sp,_=load_model("mlx-community/Voxtral-Mini-4B-Realtime-6bit")
audio=load_audio("perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_20s.wav")
audio=pad_audio(audio)
prompt,n_delay=_build_prompt_tokens(sp)
emb_non=model.encode(log_mel_spectrogram(audio))
emb_inc=inc_embeds(model,audio)
t_non=decode(model,emb_non,prompt,n_delay,sp.eos_id)
t_inc=decode(model,emb_inc,prompt,n_delay,sp.eos_id)
print("first_divergence", first_diff(t_non,t_inc))
PY
```

## Key Observations

### 1) Deterministic first divergence at token 50
- first token divergence index: `50`
- first meaningful divergence index: `50`
- non-incremental path becomes more `[STREAMING_PAD]`-heavy after divergence.

Artifacts:
- `perf/audio_runs/trace-noninc-vs-inc-20260209T151236Z/comparison.json`
- `perf/audio_runs/trace-noninc-vs-inc-20260209T154656Z/comparison.json`
- `perf/audio_runs/trace-noninc-vs-inc-20260209T154656Z/focus_pairwise.json`

### 2) Decoder sliding window is not primary cause
A sweep over decoder `sliding_window` (`128, 512, 2048, 8192, 16384`) keeps first divergence pinned at `50`.
Small windows degrade quality further, but do not move divergence onset.

### 3) Stage localization
- Mel parity: effectively equal.
- Conv parity: `forward_conv` vs concatenated `forward_conv_step` is exactly equal through position `133`.
  - first conv nonzero mismatch at `134` (later than divergence trigger region).
- Transformer path: full-sequence vs cached-history diverges from layer 0 with history present.

### 4) Decoder path appears consistent given input embeddings
`generate_nonincremental` == decode-from-nonincremental-embeds replay, while decode from incremental embeds differs with first divergence at token 50.

### 5) Reproduced on original-author commit
Commit `e6d193e85e84e30f26e370c66973ce287b8a9d57`:
- 120s: first divergence index `50`, token norm ~`0.254473`
- 20s: first divergence index `50`, token norm ~`0.065637`

Artifacts:
- `perf/audio_runs/commit-compare-e6d193e-20260209T000000Z/comparison.json`
- `perf/audio_runs/commit-compare-e6d193e-20s-20260209T000000Z/comparison.json`

## Ruled Out / Lower-Probability Causes
- Async scheduling race (`mx.async_eval`) as primary cause.
- STFT/mel mismatch as primary cause.
- Conv-step equivalence as primary cause.
- Decoder cache window wrap as primary cause for token-50 onset.

## Likely Root Cause Class
Encoder transformer full-sequence attention path (`encoder.__call__`) and cached incremental path (`forward_transformer(..., cache=...)`) are not numerically/semantically equivalent with history, causing early embedding drift and autoregressive argmax flip at token 50.

## Suggested Maintainer Actions
1. Treat incremental encoder path as correctness baseline for file decoding.
2. Add a regression check: first divergence index vs incremental reference on 120s fixture should be `None` (or within strict tolerance).
3. Investigate encoder full-vs-cached attention equivalence in layer 0 with history present (mask alignment / cached-history semantics in MLX SDPA).
4. Optionally reintroduce a faster batch path only if it matches incremental-reference behavior on the same trace metrics.
