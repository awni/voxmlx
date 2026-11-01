import mlx.core as mx
import mlx.nn as nn


def create_causal_mask(N: int, offset: int = 0, window_size: int | None = None):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    return mask


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding_total = kernel_size - stride
        # MLX Conv1d: weight shape [out_channels, kernel_size, in_channels]
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, time, channels] (NHC layout)
        # Left-pad for causal convolution
        if self.padding_total > 0:
            x = mx.pad(x, [(0, 0), (self.padding_total, 0), (0, 0)])
        # MLX conv1d expects [batch, time, channels]
        return mx.conv1d(x, self.weight, stride=self.stride) + self.bias


class EncoderAttention(nn.Module):
    def __init__(self, dim: int = 1280, n_heads: int = 32, head_dim: int = 64, rope_theta: float = 1e6):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=True)

        self.rope_theta = rope_theta

    def __call__(self, x: mx.array, offset: int, mask: mx.array, cache=None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
        q = mx.fast.rope(q, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class EncoderSwiGLU(nn.Module):
    def __init__(self, dim: int = 1280, hidden_dim: int = 5120):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int = 1280, n_heads: int = 32, head_dim: int = 64, hidden_dim: int = 5120, rope_theta: float = 1e6):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.attention = EncoderAttention(dim, n_heads, head_dim, rope_theta)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.mlp = EncoderSwiGLU(dim, hidden_dim)

    def __call__(self, x: mx.array, offset: int, mask: mx.array, cache=None) -> mx.array:
        x = x + self.attention(self.attn_norm(x), offset, mask, cache=cache)
        x = x + self.mlp(self.ffn_norm(x))
        return x



class CausalWhisperEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        dim: int = 1280,
        n_layers: int = 32,
        n_heads: int = 32,
        head_dim: int = 64,
        hidden_dim: int = 5120,
        rope_theta: float = 1e6,
        sliding_window: int = 750,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, dim, kernel_size=3, stride=1)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=3, stride=2)
        self.layers = [
            EncoderLayer(dim, n_heads, head_dim, hidden_dim, rope_theta) for _ in range(n_layers)
        ]
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.sliding_window = sliding_window

    def forward_conv(self, mel: mx.array) -> mx.array:
        # mel: [n_mels, T]
        x = mel.T[None, :, :]  # [1, T, n_mels]
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        return x  # [1, T/2, dim]

    def forward_conv_step(self, new_mel, conv1_tail, conv2_tail):
        """Incremental conv: process new mel frames with cached tails.

        new_mel: [1, N, n_mels] (already transposed)
        conv1_tail: [1, 2, n_mels] or None (first call)
        conv2_tail: [1, 1, dim] or None (first call)

        Returns: (conv2_out, new_conv1_tail, new_conv2_tail)
        """
        # Conv1 (k=3, s=1, causal pad=2)
        if conv1_tail is not None:
            x = mx.concatenate([conv1_tail, new_mel], axis=1)
        else:
            x = mx.pad(new_mel, [(0, 0), (self.conv1.padding_total, 0), (0, 0)])
        new_conv1_tail = new_mel[:, -self.conv1.padding_total:, :]
        x = nn.gelu(
            mx.conv1d(x, self.conv1.weight, stride=self.conv1.stride)
            + self.conv1.bias
        )

        # Conv2 (k=3, s=2, causal pad=1)
        if conv2_tail is not None:
            x_in = mx.concatenate([conv2_tail, x], axis=1)
        else:
            x_in = mx.pad(x, [(0, 0), (self.conv2.padding_total, 0), (0, 0)])
        new_conv2_tail = x[:, -self.conv2.padding_total:, :]
        x = nn.gelu(
            mx.conv1d(x_in, self.conv2.weight, stride=self.conv2.stride)
            + self.conv2.bias
        )

        return x, new_conv1_tail, new_conv2_tail

    def _make_mask(self, N: int, cache=None):
        offset = (
            min(cache[0].max_size - 1, cache[0].offset)
            if cache is not None
            else 0
        )
        if offset + N > self.sliding_window:
            return create_causal_mask(N, offset, window_size=self.sliding_window)
        return "causal" if N > 1 else None

    def forward_transformer(self, x, cache=None):
        """Run transformer layers with optional KV cache."""
        mask = self._make_mask(x.shape[1], cache)
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, offset=0, mask=mask, cache=layer_cache)
        x = self.norm(x)
        return x

    def __call__(self, mel: mx.array) -> mx.array:
        x = self.forward_conv(mel.astype(self.conv1.weight.dtype))  # [1, T/2, dim]
        return self.forward_transformer(x)  # [1, T/2, dim]
