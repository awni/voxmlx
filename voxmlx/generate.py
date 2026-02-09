import mlx.core as mx

from .audio import load_audio, log_mel_spectrogram, pad_audio
from .cache import RotatingKVCache
from .model import VoxtralRealtime


def generate(
    model: VoxtralRealtime,
    audio_path: str,
    prompt_tokens: list[int],
    n_delay_tokens: int,
    temperature: float = 0.0,
    eos_token_id: int = 2,
    sliding_window: int = 8192,
) -> list[int]:
    # 1. Load audio, pad for streaming, and compute mel spectrogram
    audio = load_audio(audio_path)
    audio = pad_audio(audio)
    mel = log_mel_spectrogram(audio)  # [n_mels, T]

    # 2. Encode audio
    audio_embeds = model.encode(mel)  # [N_audio, 3072]
    N_audio = audio_embeds.shape[0]

    # 3. Time conditioning (uses delay tokens only, not left pad)
    t_cond = model.time_embedding(mx.array([n_delay_tokens], dtype=mx.float32))  # [1, dim]

    # 4. Build prefix embeddings
    # prompt_tokens = [BOS] + [STREAMING_PAD] * (n_left_pad + n_delay)
    prefix_len = len(prompt_tokens)
    prompt_ids = mx.array([prompt_tokens])  # [1, prefix_len]
    text_embeds = model.language_model.embed(prompt_ids)[0]  # [prefix_len, 3072]

    # Each prefix position: tok_embed + audio_embed
    prefix_embeds = text_embeds + audio_embeds[:prefix_len]  # [prefix_len, 3072]
    prefix_embeds = prefix_embeds[None, :, :]  # [1, prefix_len, 3072]

    n_layers = len(model.language_model.layers)
    cache = [RotatingKVCache(sliding_window) for _ in range(n_layers)]

    def sample(logits):
        if temperature <= 0:
            return mx.argmax(logits[0, -1:], axis=-1).squeeze()
        return mx.random.categorical(logits[0, -1:] / temperature).squeeze()

    def step(token, pos):
        token_embed = model.language_model.embed(token.reshape(1, 1))[0, 0]
        step_embed = (audio_embeds[pos] + token_embed)[None, None, :]
        logits = model.decode(step_embed, t_cond, mask=None, cache=cache)
        return sample(logits)

    # 5. Prefill
    logits = model.decode(prefix_embeds, t_cond, "causal", cache)
    mx.eval(logits, *[x for c in cache for x in (c.keys, c.values)])

    # 6. Autoregressive loop with async_eval double buffering
    y = sample(logits)
    mx.async_eval(y)

    output_tokens = []
    for pos in range(prefix_len, N_audio):
        next_y = step(y, pos)
        mx.async_eval(next_y)

        token_id = y.item()
        if token_id == eos_token_id:
            break
        output_tokens.append(token_id)

        if pos % 256 == 0:
            mx.clear_cache()

        y = next_y

    # Flush last pending token when loop ends without EOS.
    token_id = y.item()
    if token_id != eos_token_id:
        output_tokens.append(token_id)

    return output_tokens
