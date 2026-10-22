"""Non-deep controls: the location-only baseline and the radiomics comparator.

These are first-class components, not ablations.

**Location-only** is the decision-gate experiment for the whole paper. Because
the proposed architecture is built on anatomical region, a model that beats the
deep pipeline using *nothing but the region label* would show that the reported
performance reflects the known location-genotype association rather than an
imaging phenotype. That association is well documented - a genotype x location
interaction is visible in SSTR-PET phenotype (Imperiale et al., EJNMMI 2026),
germline variant presence tracks topography (Lima et al., Endocrine Connections
2023), and SDHx carriers show a predominance of extra-adrenal tumours - yet no
published study quantifies how much of imaging-based genotype prediction it
explains. In this cohort abdomen accounts for the large majority of slices and
head & neck contains no kinase-signalling cases, so the baseline is stronger than
it looks. Run it first.

**Radiomics + gradient boosting** is the comparator the PPGL imaging community
actually uses (Zhou et al. 2024/2025; Yi et al.). See
:mod:`ppgl.data.radiomics` for the caveat that these features are computed on
pre-windowed 8-bit data and are therefore not HU-calibrated.

All classes expose ``fit(X, y)`` / ``predict_proba(X)`` and always return a
``(N, n_classes)`` matrix with columns ordered as :data:`ppgl.CLUSTERS`, even when
a class is absent from the training fold.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ppgl import REGIONS
from ppgl.labels import n_active_classes
from ppgl.utils.logging import get_logger

log = get_logger(__name__)


class _FixedColumnMixin:
    """Expands a sklearn ``predict_proba`` to the full class space."""

    n_classes: int
    classes_: np.ndarray

    def _expand(self, probabilities: np.ndarray) -> np.ndarray:
        out = np.zeros((probabilities.shape[0], self.n_classes), dtype=np.float64)
        for column, label in enumerate(self.classes_):
            out[:, int(label)] = probabilities[:, column]
        return out


class RegionMajorityClassifier(_FixedColumnMixin):
    """Per-region majority-class predictor. The floor for the location-only control.

    Predicts the empirical cluster distribution of the training slices within
    each region, so it is a calibrated prior rather than a hard majority vote.
    """

    def __init__(self, n_classes: Optional[int] = None, n_regions: int = len(REGIONS)) -> None:
        self.n_classes = int(n_classes or n_active_classes())
        self.n_regions = int(n_regions)
        self.classes_ = np.arange(self.n_classes)
        self.region_priors_: Optional[np.ndarray] = None
        self.global_prior_: Optional[np.ndarray] = None

    def fit(self, regions: Sequence[int], y: Sequence[int]) -> "RegionMajorityClassifier":
        regions = np.asarray(regions, dtype=int).reshape(-1)
        y = np.asarray(y, dtype=int).reshape(-1)

        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        self.global_prior_ = counts / max(counts.sum(), 1.0)

        priors = np.zeros((self.n_regions, self.n_classes), dtype=np.float64)
        for region in range(self.n_regions):
            mask = regions == region
            if not mask.any():
                priors[region] = self.global_prior_
                continue
            region_counts = np.bincount(y[mask], minlength=self.n_classes).astype(np.float64)
            priors[region] = region_counts / max(region_counts.sum(), 1.0)
        self.region_priors_ = priors
        return self

    def predict_proba(self, regions: Sequence[int]) -> np.ndarray:
        if self.region_priors_ is None:
            raise RuntimeError("call fit() first")
        regions = np.asarray(regions, dtype=int).reshape(-1)
        out = np.empty((regions.size, self.n_classes), dtype=np.float64)
        for row, region in enumerate(regions):
            if 0 <= region < self.n_regions:
                out[row] = self.region_priors_[region]
            else:
                out[row] = self.global_prior_
        return out


class LocationOnlyClassifier(_FixedColumnMixin):
    """Multinomial logistic regression on the one-hot anatomical region.

    Deliberately has access to *no image content whatsoever* - only which of the
    three anatomical regions the lesion sits in.
    """

    def __init__(
        self,
        n_classes: Optional[int] = None,
        n_regions: int = len(REGIONS),
        class_weight: Optional[str] = "balanced",
        C: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.n_classes = int(n_classes or n_active_classes())
        self.n_regions = int(n_regions)
        self.class_weight = class_weight
        self.C = float(C)
        self.random_state = int(random_state)
        self.model = None
        self.classes_ = np.arange(self.n_classes)
        self._fallback: Optional[np.ndarray] = None

    def _design(self, regions: Sequence[int]) -> np.ndarray:
        regions = np.asarray(regions, dtype=int).reshape(-1)
        design = np.zeros((regions.size, self.n_regions), dtype=np.float64)
        valid = (regions >= 0) & (regions < self.n_regions)
        design[np.arange(regions.size)[valid], regions[valid]] = 1.0
        return design

    def fit(self, regions: Sequence[int], y: Sequence[int]) -> "LocationOnlyClassifier":
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y, dtype=int).reshape(-1)
        design = self._design(regions)

        unique = np.unique(y)
        if unique.size < 2:
            counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
            self._fallback = counts / max(counts.sum(), 1.0)
            self.model = None
            return self

        self.model = LogisticRegression(
            max_iter=2000,
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.model.fit(design, y)
        self.classes_ = self.model.classes_
        self._fallback = None
        return self

    def predict_proba(self, regions: Sequence[int]) -> np.ndarray:
        design = self._design(regions)
        if self.model is None:
            if self._fallback is None:
                raise RuntimeError("call fit() first")
            return np.tile(self._fallback, (design.shape[0], 1))
        return self._expand(self.model.predict_proba(design))


class TabularLogisticClassifier(_FixedColumnMixin):
    """Multinomial logistic regression on a dense feature matrix.

    Used for the clinical-only control. Logistic regression rather than boosted
    trees because the clinical block is ~30 columns over ~180 patients per class,
    where a linear model is better matched and far less prone to fitting noise.
    The matrix arrives already imputed and standardised by
    :class:`ppgl.data.clinical.ClinicalPreprocessor`, fit on training rows only.
    """

    def __init__(
        self,
        n_classes: Optional[int] = None,
        C: float = 0.5,
        class_weight: Optional[str] = "balanced",
        random_state: int = 0,
    ) -> None:
        self.n_classes = int(n_classes or n_active_classes())
        self.C = float(C)
        self.class_weight = class_weight
        self.random_state = int(random_state)
        self.model = None
        self.classes_ = np.arange(self.n_classes)
        self._fallback: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, y: Sequence[int], **_: object) -> "TabularLogisticClassifier":
        from sklearn.linear_model import LogisticRegression

        X = np.nan_to_num(np.asarray(features, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray(y, dtype=int).reshape(-1)

        if X.shape[1] == 0 or np.unique(y).size < 2:
            counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
            self._fallback = counts / max(counts.sum(), 1.0)
            self.model = None
            return self

        self.model = LogisticRegression(
            max_iter=5000,
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self._fallback = None
        return self

    def predict_proba(self, features: np.ndarray, **_: object) -> np.ndarray:
        X = np.nan_to_num(np.asarray(features, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if self.model is None:
            if self._fallback is None:
                raise RuntimeError("call fit() first")
            return np.tile(self._fallback, (X.shape[0], 1))
        return self._expand(self.model.predict_proba(X))


class RadiomicsGBMClassifier(_FixedColumnMixin):
    """Gradient-boosted trees on hand-crafted 2D features.

    Uses LightGBM when installed, otherwise sklearn's
    ``HistGradientBoostingClassifier``. Features are standardised and
    zero-variance columns are dropped, both fitted on the training fold only.
    """

    def __init__(
        self,
        n_classes: Optional[int] = None,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 0,
        include_region: bool = False,
        n_regions: int = len(REGIONS),
    ) -> None:
        self.n_classes = int(n_classes or n_active_classes())
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.random_state = int(random_state)
        self.include_region = bool(include_region)
        self.n_regions = int(n_regions)

        self.model = None
        self.classes_ = np.arange(self.n_classes)
        self._keep: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._fallback: Optional[np.ndarray] = None

    def _augment(self, features: np.ndarray, regions: Optional[Sequence[int]]) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if not self.include_region or regions is None:
            return features
        regions = np.asarray(regions, dtype=int).reshape(-1)
        onehot = np.zeros((features.shape[0], self.n_regions), dtype=np.float64)
        valid = (regions >= 0) & (regions < self.n_regions)
        onehot[np.arange(features.shape[0])[valid], regions[valid]] = 1.0
        return np.hstack([features, onehot])

    def fit(
        self,
        features: np.ndarray,
        y: Sequence[int],
        regions: Optional[Sequence[int]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "RadiomicsGBMClassifier":
        X = self._augment(features, regions)
        y = np.asarray(y, dtype=int).reshape(-1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        variance = X.var(axis=0)
        self._keep = variance > 1e-12
        if not self._keep.any():
            self._keep = np.ones(X.shape[1], dtype=bool)
        X = X[:, self._keep]

        self._mean = X.mean(axis=0)
        self._scale = np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
        X = (X - self._mean) / self._scale

        if np.unique(y).size < 2:
            counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
            self._fallback = counts / max(counts.sum(), 1.0)
            self.model = None
            return self

        weights = sample_weight
        if weights is None:
            counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
            inverse = np.zeros_like(counts)
            positive = counts > 0
            inverse[positive] = counts[positive].sum() / counts[positive]
            weights = inverse[y]

        self.model = self._build_estimator()
        try:
            self.model.fit(X, y, sample_weight=weights)
        except TypeError:
            self.model.fit(X, y)
        self.classes_ = np.asarray(self.model.classes_)
        self._fallback = None
        return self

    def _build_estimator(self):
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=15,
                min_child_samples=10,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbose=-1,
                # Single-threaded on purpose. LightGBM defaults to one thread per
                # core and reads its own n_jobs rather than only OMP_NUM_THREADS,
                # so without this an 8-worker grid on a 96-core host asks for ~768
                # threads. At 630 rows x 57 features there is no work to spread:
                # measured 0.75 s single-threaded versus 3.94 s on 14 threads, and
                # a median of 27 MINUTES per fold on the oversubscribed server.
                # Parallelism belongs at the fold level, not inside the fit.
                n_jobs=1,
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier

            log.info("lightgbm unavailable; using sklearn HistGradientBoostingClassifier")
            # Thread count for this one comes from OMP_NUM_THREADS, which the
            # dispatcher pins per worker; it has no n_jobs parameter.
            return HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=self.random_state,
            )

    def predict_proba(
        self,
        features: np.ndarray,
        regions: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        X = self._augment(features, regions)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self._keep is None:
            raise RuntimeError("call fit() first")
        X = X[:, self._keep]
        X = (X - self._mean) / self._scale
        if self.model is None:
            return np.tile(self._fallback, (X.shape[0], 1))
        return self._expand(self.model.predict_proba(X))


__all__ = [
    "LocationOnlyClassifier",
    "RadiomicsGBMClassifier",
    "RegionMajorityClassifier",
    "TabularLogisticClassifier",
]
