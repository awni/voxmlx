import importlib.util
import os
import unittest


class EncoderWindowBugfixOptionalTests(unittest.TestCase):
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

    def test_encode_step_uses_encoder_sliding_window(self):
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
        self.assertEqual(len(encoder_cache), 2)
        for layer_cache in encoder_cache:
            self.assertEqual(layer_cache.max_size, sliding_window)


if __name__ == "__main__":
    unittest.main()
