import importlib.util
import os
import unittest


class EncodeTrimBugfixOptionalTests(unittest.TestCase):
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

    def test_encode_trims_trailing_frames_not_leading_frames(self):
        import mlx.core as mx
        import numpy as np

        from voxmlx.model import VoxtralRealtime

        class _FakeEncoder:
            def __call__(self, mel):
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
        mel = mx.zeros((128, 9), dtype=mx.float32)
        out = VoxtralRealtime.encode(fake, mel)

        # Expected: keep earliest 6 encoded frames (0..5), grouped by 3.
        expected = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        np.testing.assert_allclose(np.array(out), expected, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
