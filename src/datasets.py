import os
from typing import Optional, Sequence, Dict

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    """
    Generic deepfake dataset that reads from a metadata CSV.

    Expected CSV columns:
        - path: full path to the image file
        - label: 'real', 'gan_fake', or 'diff_fake'
        - split: 'train', 'val', or 'test'
        - (optional) source, generator

    Arguments
    ---------
    csv_path : str
        Path to the metadata CSV (e.g., metadata_core_10k_per_class.csv).
    split : str
        One of {'train', 'val', 'test'}.
    include_labels : Sequence[str] or None
        If given, only keep rows whose label is in this list
        (e.g., ['real', 'gan_fake'] for the GAN vs real binary task).
        If None, use all labels present.
    transform : callable or None
        Transform to apply to the PIL image (e.g., torchvision transforms).
    label_mapping : Optional[Dict[str, int]]
        Optional explicit mapping from label string to integer index.
        If None, a mapping is created from the sorted unique labels in the
        (possibly filtered) dataframe.
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        include_labels: Optional[Sequence[str]] = None,
        transform=None,
        label_mapping: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        df = pd.read_csv(csv_path)

        # Filter by split
        df = df[df["split"] == split].copy()

        # Optional: filter by subset of labels (for binary tasks)
        if include_labels is not None:
            df = df[df["label"].isin(include_labels)].copy()

        if df.empty:
            raise ValueError(
                f"No data left after filtering for split={split!r}, "
                f"include_labels={include_labels}. Check your CSV and filters."
            )

        self.df = df.reset_index(drop=True)
        self.transform = transform

        # Build or store label mapping
        if label_mapping is None:
            labels = sorted(self.df["label"].unique())
            self.label_to_idx = {lab: i for i, lab in enumerate(labels)}
        else:
            self.label_to_idx = dict(label_mapping)

        # Precompute numeric labels
        self.targets = self.df["label"].map(self.label_to_idx).astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["path"]

        # Load image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label_idx = self.targets[idx]
        return img, torch.tensor(label_idx, dtype=torch.long)

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)

    @property
    def classes(self):
        # Sorted by index
        inv = {v: k for k, v in self.label_to_idx.items()}
        return [inv[i] for i in range(len(inv))]
