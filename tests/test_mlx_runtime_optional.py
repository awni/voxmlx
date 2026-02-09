import importlib.util
import os
import unittest
from unittest import mock


class MlxRuntimeOptionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enabled = os.getenv("VOXMLX_ENABLE_MLX_RUNTIME_TESTS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            raise unittest.SkipTest(
                "Set VOXMLX_ENABLE_MLX_RUNTIME_TESTS=1 to run MLX runtime optional tests"
            )

        if importlib.util.find_spec("mlx") is None:
            raise unittest.SkipTest("mlx is required for runtime optional tests")

    def test_rotating_kv_cache_concat_respects_max_size(self):
        import mlx.core as mx

        from voxmlx.cache import RotatingKVCache

        # Oversized first concat update should still honor max_size.
        first_big = RotatingKVCache(max_size=4)
        keys = mx.arange(7, dtype=mx.float32).reshape(1, 1, 7, 1)
        values = mx.arange(7, dtype=mx.float32).reshape(1, 1, 7, 1)
        k0, v0 = first_big.update_and_fetch(keys, values)
        self.assertEqual(k0.shape[2], 4)
        self.assertEqual(v0.shape[2], 4)

        cache = RotatingKVCache(max_size=8)

        for span in (3, 4, 5):
            keys = mx.arange(span, dtype=mx.float32).reshape(1, 1, span, 1)
            values = mx.arange(span, dtype=mx.float32).reshape(1, 1, span, 1)
            k, v = cache.update_and_fetch(keys, values)
            self.assertLessEqual(k.shape[2], 8)
            self.assertLessEqual(v.shape[2], 8)

    def test_rotating_kv_cache_concat_keeps_expected_tail_for_multi_token_append(self):
        import mlx.core as mx
        import numpy as np

        from voxmlx.cache import RotatingKVCache

        cache = RotatingKVCache(max_size=8)

        base = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8, 1)
        cache.update_and_fetch(base, base)

        add = mx.array([100, 101, 102, 103], dtype=mx.float32).reshape(1, 1, 4, 1)
        k, v = cache.update_and_fetch(add, add)

        expected = mx.array([4, 5, 6, 7, 100, 101, 102, 103], dtype=mx.float32).reshape(
            1, 1, 8, 1
        )
        self.assertEqual(k.shape[2], 8)
        self.assertEqual(v.shape[2], 8)
        np.testing.assert_allclose(np.array(k), np.array(expected), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.array(v), np.array(expected), atol=0.0, rtol=0.0)

    def test_offline_and_streaming_mel_match(self):
        import mlx.core as mx
        import numpy as np

        from voxmlx.audio import (
            SAMPLES_PER_TOKEN,
            log_mel_spectrogram,
            log_mel_spectrogram_step,
        )

        rng = np.random.default_rng(0)
        audio = rng.standard_normal(123456).astype(np.float32) * 0.01

        offline = log_mel_spectrogram(audio)

        tail = None
        chunks = []
        for i in range(0, len(audio), SAMPLES_PER_TOKEN):
            mel, tail = log_mel_spectrogram_step(audio[i : i + SAMPLES_PER_TOKEN], tail)
            chunks.append(mel)

        streaming = mx.concatenate(chunks, axis=1)
        self.assertEqual(tuple(offline.shape), tuple(streaming.shape))

        offline_np = np.array(offline)
        streaming_np = np.array(streaming)
        self.assertTrue(np.allclose(offline_np, streaming_np, atol=1e-5, rtol=1e-5))

    def test_generate_flushes_final_pending_token(self):
        import mlx.core as mx
        import numpy as np

        import voxmlx.generate as generate_module

        class _FakeLanguageModel:
            def __init__(self, hidden_dim: int, vocab_size: int):
                self.layers = [object()]
                self._hidden_dim = hidden_dim
                self._vocab_size = vocab_size

            def embed(self, input_ids):
                bsz, seq = input_ids.shape
                return mx.zeros((bsz, seq, self._hidden_dim), dtype=mx.float32)

        class _FakeModel:
            def __init__(self):
                self.hidden_dim = 8
                self.vocab_size = 128
                self.n_audio = 4  # prefix=2, loop runs 2 steps
                self._decode_tokens = [11, 12, 13]  # prefill + 2 steps
                self._decode_calls = 0
                self.language_model = _FakeLanguageModel(self.hidden_dim, self.vocab_size)

            def time_embedding(self, t):
                return mx.zeros((1, self.hidden_dim), dtype=mx.float32)

            def encode(self, mel):
                del mel
                return mx.zeros((self.n_audio, self.hidden_dim), dtype=mx.float32)

            def decode(self, embeddings, t_cond, mask=None, cache=None):
                del t_cond, mask, cache
                token = self._decode_tokens[self._decode_calls]
                self._decode_calls += 1
                seq = embeddings.shape[1]
                logits = np.full((1, seq, self.vocab_size), -1000.0, dtype=np.float32)
                logits[:, :, token] = 1000.0
                return mx.array(logits)

        model = _FakeModel()
        with mock.patch.object(generate_module, "load_audio", return_value=np.zeros(1, dtype=np.float32)), mock.patch.object(
            generate_module, "pad_audio", side_effect=lambda x: x
        ), mock.patch.object(
            generate_module, "log_mel_spectrogram", return_value=mx.zeros((128, 8), dtype=mx.float32)
        ):
            out = generate_module.generate(
                model=model,
                audio_path="dummy.wav",
                prompt_tokens=[1, 2],
                n_delay_tokens=0,
                temperature=0.0,
                eos_token_id=99,
                sliding_window=16,
            )

        self.assertEqual(out, [11, 12, 13])

    def test_encode_matches_incremental_encode_step(self):
        import mlx.core as mx
        import numpy as np

        from voxmlx.model import VoxtralRealtime

        config = {
            "dim": 16,
            "ada_rms_norm_t_cond_dim": 16,
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "head_dim": 8,
            "hidden_dim": 32,
            "vocab_size": 256,
            "rope_theta": 1e6,
            "multimodal": {
                "whisper_model_args": {
                    "encoder_args": {
                        "audio_encoding_args": {"num_mel_bins": 128},
                        "dim": 16,
                        "n_layers": 2,
                        "n_heads": 2,
                        "head_dim": 8,
                        "hidden_dim": 32,
                        "rope_theta": 1e6,
                        "sliding_window": 64,
                    },
                    "downsample_args": {"downsample_factor": 4},
                }
            },
        }
        model = VoxtralRealtime(config)

        rng = np.random.default_rng(42)
        mel = mx.array(rng.standard_normal((128, 241)).astype(np.float32) * 0.1)
        offline = model.encode(mel)

        conv1_tail = conv2_tail = encoder_cache = ds_buf = None
        parts = []
        # Use realistic streaming-like mel chunk sizes (>= conv overlap).
        spans = [8, 12, 4, 16]
        pos = 0
        i = 0
        while pos < mel.shape[1]:
            span = spans[i % len(spans)]
            chunk = mel[:, pos : pos + span]
            out, conv1_tail, conv2_tail, encoder_cache, ds_buf = model.encode_step(
                chunk,
                conv1_tail,
                conv2_tail,
                encoder_cache,
                ds_buf,
            )
            if out is not None and out.shape[0] > 0:
                parts.append(out)
            pos += span
            i += 1

        if parts:
            incremental = mx.concatenate(parts, axis=0)
        else:
            incremental = mx.zeros((0, config["dim"]), dtype=mx.float32)

        self.assertEqual(tuple(offline.shape), tuple(incremental.shape))
        self.assertTrue(
            np.allclose(np.array(offline), np.array(incremental), atol=1e-4, rtol=1e-4)
        )

    def test_encode_step_uses_encoder_sliding_window_for_cache_size(self):
        import mlx.core as mx

        from voxmlx.model import VoxtralRealtime

        sliding_window = 7
        config = {
            "dim": 16,
            "ada_rms_norm_t_cond_dim": 16,
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "head_dim": 8,
            "hidden_dim": 32,
            "vocab_size": 256,
            "rope_theta": 1e6,
            "multimodal": {
                "whisper_model_args": {
                    "encoder_args": {
                        "audio_encoding_args": {"num_mel_bins": 128},
                        "dim": 16,
                        "n_layers": 2,
                        "n_heads": 2,
                        "head_dim": 8,
                        "hidden_dim": 32,
                        "rope_theta": 1e6,
                        "sliding_window": sliding_window,
                    },
                    "downsample_args": {"downsample_factor": 4},
                }
            },
        }
        model = VoxtralRealtime(config)

        mel_chunk = mx.zeros((128, 16), dtype=mx.float32)
        _, _, _, encoder_cache, _ = model.encode_step(
            mel_chunk,
            conv1_tail=None,
            conv2_tail=None,
            encoder_cache=None,
            ds_buf=None,
        )

        self.assertIsNotNone(encoder_cache)
        self.assertEqual(len(encoder_cache), 2)
        for layer_cache in encoder_cache:
            self.assertEqual(layer_cache.max_size, sliding_window)

    def test_encode_trims_trailing_frames_not_leading_frames(self):
        import mlx.core as mx
        import numpy as np

        from voxmlx.model import VoxtralRealtime

        class _FakeEncoder:
            def __call__(self, mel):
                # Return one channel that directly tracks encoded frame index.
                length = int(mel.shape[1])
                return mx.arange(length, dtype=mx.float32).reshape(1, length, 1)

        class _FakeAdapter:
            def __call__(self, x):
                return x

        class _FakeModel:
            def __init__(self):
                self.encoder = _FakeEncoder()
                self.adapter = _FakeAdapter()
                self.downsample_factor = 3

        fake = _FakeModel()

        # Odd mel length triggers the first trim path. Then downsample remainder
        # triggers the second trim path. Correct behavior keeps earliest indices.
        mel = mx.zeros((128, 9), dtype=mx.float32)
        out = VoxtralRealtime.encode(fake, mel)
        expected = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        np.testing.assert_allclose(np.array(out), expected, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
