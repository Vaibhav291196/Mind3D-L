
import numpy as np
from pathlib import Path


class FMRINormalizer:
    def __init__(self, clip_value=10.0, epsilon=1e-6):
        self.clip_value = clip_value
        self.epsilon = epsilon
        self.mean = None
        self.std = None

    def fit(self, train_fmri):
        """
        Fit using training scans and frames only.

        train_fmri: (N, T, V)
        """
        train_fmri = np.asarray(train_fmri, dtype=np.float32)

        self.mean = train_fmri.mean(axis=(0, 1))
        self.std = train_fmri.std(axis=(0, 1))

        self.std = np.maximum(self.std, self.epsilon)

        return self

    def transform(self, fmri):
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fitted first.")

        fmri = np.asarray(fmri, dtype=np.float32)

        normalized = (fmri - self.mean) / self.std

        if self.clip_value is not None:
            normalized = np.clip(
                normalized,
                -self.clip_value,
                self.clip_value
            )

        return normalized.astype(np.float32)

    def fit_transform(self, train_fmri):
        self.fit(train_fmri)
        return self.transform(train_fmri)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            mean=self.mean,
            std=self.std,
            clip_value=np.array(self.clip_value),
            epsilon=np.array(self.epsilon),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)

        normalizer = cls(
            clip_value=float(data["clip_value"]),
            epsilon=float(data["epsilon"]),
        )

        normalizer.mean = data["mean"].astype(np.float32)
        normalizer.std = data["std"].astype(np.float32)

        return normalizer
