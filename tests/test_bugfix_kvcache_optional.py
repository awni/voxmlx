import importlib.util
import os
import unittest


class KVCacheBugfixOptionalTests(unittest.TestCase):
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

    def test_concat_append_respects_max_size(self):
        import mlx.core as mx

        from voxmlx.cache import RotatingKVCache

        cache = RotatingKVCache(max_size=8)
        base = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8, 1)
        add = mx.array([100, 101, 102, 103], dtype=mx.float32).reshape(1, 1, 4, 1)

        cache.update_and_fetch(base, base)
        k, v = cache.update_and_fetch(add, add)

        self.assertEqual(k.shape[2], 8)
        self.assertEqual(v.shape[2], 8)

    def test_oversized_first_update_trims_to_max_size(self):
        import mlx.core as mx
        import numpy as np

        from voxmlx.cache import RotatingKVCache

        cache = RotatingKVCache(max_size=4)
        keys = mx.arange(7, dtype=mx.float32).reshape(1, 1, 7, 1)
        values = mx.arange(7, dtype=mx.float32).reshape(1, 1, 7, 1)
        k, v = cache.update_and_fetch(keys, values)

        expected = mx.array([3, 4, 5, 6], dtype=mx.float32).reshape(1, 1, 4, 1)
        np.testing.assert_allclose(np.array(k), np.array(expected), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.array(v), np.array(expected), atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
