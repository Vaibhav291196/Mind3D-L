
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from utils.fmri_normalization import FMRINormalizer


class FMRIDataset(Dataset):

    def __init__(
        self,
        npz_path,
        split="train",
        clip_feature_path=None,
        use_clip_features=False,
        use_normalization=True,
        normalization_path=None,
        use_spatial_mapping=False,
        voxel_path=None,
    ):
        super().__init__()

        self.split = split.lower()

        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split: {self.split}")

        with np.load(npz_path, allow_pickle=True) as data:
            self.fmri = data["X"].astype(np.float32)
            self.groups = data["group"].astype(np.int64)

            self.has_points = "obj_points" in data.files
            self.points = (
                data["obj_points"].astype(np.float32)
                if self.has_points else None
            )

        self.clip_features = None

        if use_clip_features:
            if clip_feature_path is None:
                raise ValueError("CLIP feature path required.")

            if self.split == "test":
                raise ValueError("Test split has no precomputed CLIP targets.")

            key = f"{self.split}_object_features"

            with np.load(clip_feature_path) as clip_data:
                self.clip_features = clip_data[key].astype(np.float32)

        # Training-derived normalization
        if use_normalization:
            if normalization_path is None:
                raise ValueError("Normalization path required.")

            normalizer = FMRINormalizer.load(normalization_path)
            self.fmri = normalizer.transform(self.fmri)

        # Spatial mapping
        self.use_spatial_mapping = use_spatial_mapping

        if self.use_spatial_mapping:
            if voxel_path is None:
                voxel_path = Path(npz_path).parent / "voxels.npz"

            with np.load(voxel_path) as voxel_data:
                self.rows = voxel_data["row"].astype(np.int64)
                self.cols = voxel_data["col"].astype(np.int64)
                self.spatial_mask = voxel_data["mask"].astype(bool)

            if self.fmri.shape[-1] != len(self.rows):
                raise ValueError(
                    f"Feature count {self.fmri.shape[-1]} "
                    f"does not match coordinate count {len(self.rows)}"
                )

            if len(self.rows) != len(self.cols):
                raise ValueError("Row and column coordinate counts differ.")

            if not np.all(self.spatial_mask[self.rows, self.cols]):
                raise ValueError("Some coordinates fall outside spatial mask.")

            if len(np.unique(np.stack([self.rows, self.cols], axis=1), axis=0)) != len(self.rows):
                raise ValueError("Duplicate spatial coordinates detected.")

            if np.count_nonzero(self.spatial_mask) != len(self.rows):
                raise ValueError("Spatial mask and coordinate counts differ.")

            self.spatial_shape = self.spatial_mask.shape

        # Validate scan-to-object mapping
        if self.has_points:
            if self.groups.min() < 0 or self.groups.max() >= len(self.points):
                raise ValueError("Invalid scan-to-point-cloud group mapping.")

        if self.clip_features is not None:
            if self.groups.min() < 0 or self.groups.max() >= len(self.clip_features):
                raise ValueError("Invalid scan-to-CLIP-feature group mapping.")

        print(f"\n{self.split.upper()} dataset:")
        print("Scans:", len(self.fmri))
        print("Original fMRI:", self.fmri.shape)
        print("Spatial mapping:", self.use_spatial_mapping)

        if self.use_spatial_mapping:
            print("Mapped fMRI:", (len(self.fmri), self.fmri.shape[1], *self.spatial_shape))

        print("Groups:", self.groups.shape)
        print("Points:", self.points.shape if self.has_points else None)
        print("CLIP features:", self.clip_features.shape if self.clip_features is not None else None)

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, index):

        group = int(self.groups[index])
        fmri_sample = self.fmri[index]

        if self.use_spatial_mapping:
            spatial_fmri = np.zeros(
                (fmri_sample.shape[0], *self.spatial_shape),
                dtype=np.float32,
            )

            spatial_fmri[:, self.rows, self.cols] = fmri_sample

            fmri_tensor = torch.from_numpy(spatial_fmri)
        else:
            fmri_tensor = torch.from_numpy(fmri_sample.copy())

        sample = {
            "fmri": fmri_tensor,
            "group": torch.tensor(group, dtype=torch.long),
            "scan_index": torch.tensor(index, dtype=torch.long),
        }

        if self.has_points:
            sample["points"] = torch.from_numpy(
                self.points[group].copy()
            )

        if self.clip_features is not None:
            sample["clip_features"] = torch.from_numpy(
                self.clip_features[group].copy()
            )

        return sample
