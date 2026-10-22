"""Patient-level bags for multiple-instance learning.

Motivation: the slice-level pipeline predicts from one slice and then averages
probabilities per patient. That throws away the joint structure. 135 of the 287
patients contribute more than one slice, one contributes 22, and 228 slices carry
more than one lesion - so the genotype is a property of the *patient*, and the
evidence is spread across their slices. Treating a patient as a bag of instances
and learning how to weight them is the one architectural change that alters the
information available per prediction rather than just the capacity used to fit it,
which is why it has a chance where routing did not.

Each item is one patient: a stack of ``K`` slice tensors plus a validity mask.
``K`` varies from 1 to 22, so :func:`collate_bags` pads to the batch maximum and
the mask tells the attention layer which instances are real.

Memory: a 22-slice bag at 224x224x3 float32 is ~13 MB of input alone, and the
backbone sees ``batch_size * K`` instances at once. ``bag_max_slices`` caps the
training bag size by random subsampling - which doubles as augmentation, since a
patient is seen with different slice subsets each epoch - while evaluation uses
every slice so the reported number reflects all available evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ppgl.utils.logging import get_logger

log = get_logger(__name__)


class PPGLBagDataset:
    """One item per patient: ``(K, 3, S, S)`` images, a ``(K,)`` mask, one label.

    Shares the slice reader, crop logic and transforms with
    :class:`ppgl.data.dataset.PPGLSliceDataset`, so the only difference between
    the MIL model and the slice baselines is how evidence is combined.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        data_cfg,
        train: bool,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        clinical: Optional[np.ndarray] = None,
        max_slices: int = 0,
        seed: int = 0,
    ) -> None:
        from ppgl.data.dataset import PPGLSliceDataset

        # The per-slice dataset does the reading; this class only groups.
        self.slices = PPGLSliceDataset(
            frame, data_cfg, train=train, mean=mean, std=std, clinical=clinical
        )
        self.frame = self.slices.frame
        self.train = train
        self.max_slices = int(max_slices or 0)
        self._rng = np.random.default_rng(seed)

        # Stable patient order, and the row indices belonging to each patient.
        self.patient_uids: List[str] = list(dict.fromkeys(self.frame["patient_uid"].tolist()))
        self._rows: Dict[str, List[int]] = {
            uid: group.index.tolist()
            for uid, group in self.frame.groupby("patient_uid", sort=False)
        }

        labels = []
        for uid in self.patient_uids:
            unique = self.frame.loc[self._rows[uid], "cluster_idx"].unique()
            if len(unique) != 1:
                raise ValueError(f"patient {uid!r} has multiple labels {unique.tolist()}")
            labels.append(int(unique[0]))
        self._labels = np.asarray(labels, dtype=np.int64)

        sizes = np.array([len(self._rows[uid]) for uid in self.patient_uids])
        log.info(
            "bag dataset: %d patients, %d slices, bag size min=%d max=%d mean=%.2f%s",
            len(self.patient_uids),
            len(self.frame),
            sizes.min(),
            sizes.max(),
            sizes.mean(),
            f", capped at {self.max_slices} for training" if self.train and self.max_slices else "",
        )

    def __len__(self) -> int:
        return len(self.patient_uids)

    @property
    def cluster_labels(self) -> np.ndarray:
        """Bag-level labels, used by the balanced sampler."""
        return self._labels

    @property
    def region_labels(self) -> np.ndarray:
        """Modal region per patient, kept so the loss's auxiliary head still works."""
        out = np.full(len(self.patient_uids), -1, dtype=np.int64)
        if "region_idx" not in self.frame.columns:
            return out
        for position, uid in enumerate(self.patient_uids):
            values = self.frame.loc[self._rows[uid], "region_idx"].to_numpy()
            values = values[values >= 0]
            if values.size:
                out[position] = int(np.bincount(values).argmax())
        return out

    def _select(self, rows: List[int]) -> List[int]:
        if not self.max_slices or len(rows) <= self.max_slices or not self.train:
            return rows
        # Random subset during training: caps memory and acts as bag-level
        # augmentation, since the patient is seen with a different subset each epoch.
        chosen = self._rng.choice(len(rows), size=self.max_slices, replace=False)
        return [rows[i] for i in sorted(chosen)]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        uid = self.patient_uids[index]
        rows = self._select(self._rows[uid])

        items = [self.slices[row] for row in rows]
        images = torch.stack([item["image"] for item in items])  # (K, 3, S, S)

        out: Dict[str, Any] = {
            "image": images,
            "mask": torch.ones(images.shape[0], dtype=torch.bool),
            "cluster": torch.tensor(int(self._labels[index]), dtype=torch.long),
            "region": items[0]["region"],
            "region_confidence": items[0]["region_confidence"],
            "index": torch.tensor(index, dtype=torch.long),
            # One "slice_uid" per bag so the stored predictions stay joinable.
            "slice_uid": f"{uid}#bag",
            "patient_uid": uid,
            "n_slices": torch.tensor(images.shape[0], dtype=torch.long),
        }
        if "clinical" in items[0]:
            # Clinical covariates are patient-level, so any instance carries them.
            out["clinical"] = items[0]["clinical"]
        return out


def collate_bags(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length bags to the batch maximum and build the validity mask."""
    import torch

    max_k = max(int(item["image"].shape[0]) for item in items)
    batch = len(items)
    template = items[0]["image"]

    images = template.new_zeros((batch, max_k, *template.shape[1:]))
    mask = torch.zeros((batch, max_k), dtype=torch.bool)
    for position, item in enumerate(items):
        k = int(item["image"].shape[0])
        images[position, :k] = item["image"]
        mask[position, :k] = True

    out: Dict[str, Any] = {"image": images, "mask": mask}
    for key in items[0]:
        if key in ("image", "mask"):
            continue
        values = [item[key] for item in items]
        out[key] = values if isinstance(values[0], str) else torch.stack(values)
    return out


__all__ = ["PPGLBagDataset", "collate_bags"]
