import math

import mlx.core as mx
import mlx.nn as nn

from .cache import RotatingKVCache
from .encoder import CausalWhisperEncoder
from .language_model import LanguageModel


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = mx.exp(
            -math.log(theta) * mx.arange(dim // 2).astype(mx.float32) / (dim // 2)
        )
        self._inv_freq = inv_freq

    def __call__(self, t: mx.array) -> mx.array:
        # t: scalar or [B]
        t = t.reshape(-1, 1).astype(mx.float32)  # [B, 1]
        emb = t * self._inv_freq  # [B, dim//2]
        return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)  # [B, dim]


class AudioLanguageAdapter(nn.Module):
    def __init__(self, in_dim: int = 5120, out_dim: int = 3072):
        super().__init__()
        self.w_in = nn.Linear(in_dim, out_dim, bias=False)
        self.w_out = nn.Linear(out_dim, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_out(nn.gelu(self.w_in(x)))


class VoxtralRealtime(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        enc = config["multimodal"]["whisper_model_args"]["encoder_args"]
        audio_enc = enc["audio_encoding_args"]
        downsample = config["multimodal"]["whisper_model_args"]["downsample_args"]["downsample_factor"]

        self.encoder = CausalWhisperEncoder(
            in_channels=audio_enc["num_mel_bins"],
            dim=enc["dim"],
            n_layers=enc["n_layers"],
            n_heads=enc["n_heads"],
            head_dim=enc["head_dim"],
            hidden_dim=enc["hidden_dim"],
            rope_theta=enc["rope_theta"],
            sliding_window=enc["sliding_window"],
        )

        adapter_in = enc["dim"] * downsample
        self.adapter = AudioLanguageAdapter(adapter_in, config["dim"])

        cond_dim = config.get("ada_rms_norm_t_cond_dim", 32)
        self.language_model = LanguageModel(
            dim=config["dim"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            n_kv_heads=config["n_kv_heads"],
            head_dim=config["head_dim"],
            hidden_dim=config["hidden_dim"],
            vocab_size=config["vocab_size"],
            rope_theta=config["rope_theta"],
            cond_dim=cond_dim,
        )

        self.time_embedding = TimeEmbedding(dim=config["dim"])
        self.downsample_factor = downsample
        self._encoder_dim = enc["dim"]

    def encode(self, mel: mx.array) -> mx.array:
        # mel: [n_mels, T]
        # Truncate T to be even (for conv stride 2)
        T = mel.shape[1]
        if T % 2 != 0:
            mel = mel[:, 1:]

        x = self.encoder(mel)  # [1, T/2, encoder_dim]
        x = x[0]  # [T/2, encoder_dim]

        # Truncate to be divisible by downsample_factor
        L = x.shape[0]
        remainder = L % self.downsample_factor
        if remainder != 0:
            x = x[remainder:]
            L = x.shape[0]

        # Reshape: [T/2, 1280] -> [T/8, 5120]
        x = x.reshape(L // self.downsample_factor, -1)

        # Adapter: [T/8, 5120] -> [T/8, 3072]
        x = self.adapter(x)
        return x

    def encode_step(self, new_mel, conv1_tail, conv2_tail, encoder_cache, ds_buf):
        """Incrementally encode new mel frames.

        Args:
            new_mel: [n_mels, N_new] new mel frames
            conv1_tail, conv2_tail: conv state (None for first call)
            encoder_cache: list of RotatingKVCache (None for first call)
            ds_buf: leftover conv2 frames from previous downsample grouping (None initially)

        Returns:
            (new_audio_embeds or None, conv1_tail, conv2_tail, encoder_cache, ds_buf)
        """
        # Transpose mel for conv: [1, N, n_mels]
        x_mel = new_mel.T[None, :, :].astype(self.encoder.conv1.weight.dtype)

        # Incremental conv
        x, conv1_tail, conv2_tail = self.encoder.forward_conv_step(
            x_mel, conv1_tail, conv2_tail
        )

        # Create encoder cache on first call
        if encoder_cache is None:
            encoder_cache = [
                RotatingKVCache(self.encoder.sliding_window)
                for _ in range(len(self.encoder.layers))
            ]

        # Transformer with KV cache
        x = self.encoder.forward_transformer(x, cache=encoder_cache)
        x = x[0]  # [N_conv2, 1280]

        # Accumulate with downsample buffer
        if ds_buf is not None:
            x = mx.concatenate([ds_buf, x])
        n_complete = (x.shape[0] // self.downsample_factor) * self.downsample_factor
        if n_complete == 0:
            return None, conv1_tail, conv2_tail, encoder_cache, x

        ds_buf = x[n_complete:] if x.shape[0] > n_complete else None
        x = x[:n_complete]

        # Downsample + adapter
        x = x.reshape(n_complete // self.downsample_factor, -1)  # [N_tokens, 5120]
        x = self.adapter(x)  # [N_tokens, 3072]
        return x, conv1_tail, conv2_tail, encoder_cache, ds_buf

    def decode(
        self,
        embeddings: mx.array,
        t_cond: mx.array,
        mask=None,
        cache: list | None = None,
    ):
        return self.language_model(embeddings, t_cond, mask, cache)
