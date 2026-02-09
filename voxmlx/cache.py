import mlx.core as mx


class RotatingKVCache:
    step = 256

    def __init__(self, max_size):
        if int(max_size) <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size!r}")
        self.keys = None
        self.values = None
        self._offset = 0
        self.max_size = int(max_size)
        self._idx = 0

    @property
    def offset(self):
        return self._offset

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., trim_size:, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v):
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self._offset:
            return mx.concatenate(
                [v[..., self._idx:, :], v[..., : self._idx, :]],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(self, keys, values):
        if self.keys is None:
            add = keys.shape[2]
            if add > self.max_size:
                trim_size = add - self.max_size
                self.keys = keys[..., trim_size:, :]
                self.values = values[..., trim_size:, :]
            else:
                self.keys = keys
                self.values = values
        else:
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self.keys.shape[2]

            cur = self.keys.shape[2]
            add = keys.shape[2]
            trim_size = max(0, cur + add - self.max_size)
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self._offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        if __debug__:
            assert self.keys.shape[2] <= self.max_size
            assert self.values.shape[2] <= self.max_size
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        if self._idx == self.max_size:
            self._idx = 0

        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self._offset += S
        self._idx += S

        if self._offset < self.max_size:
            return self.keys[..., : self._offset, :], self.values[..., : self._offset, :]
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)
