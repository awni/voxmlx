import importlib
import importlib.util
import os
import unittest
from unittest import mock


class GenerateFlushBugfixOptionalTests(unittest.TestCase):
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

    def test_generate_flushes_last_pending_non_eos_token(self):
        import mlx.core as mx
        import numpy as np

        generate_module = importlib.import_module("voxmlx.generate")

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


if __name__ == "__main__":
    unittest.main()
