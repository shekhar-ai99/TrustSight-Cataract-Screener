"""Cross-dataset evaluation utilities.

Implements two strict modes:
 - train_on_oia_evaluate_kaggle
 - train_on_kaggle_evaluate_oia

These functions do not perform training; they assume a trained model is available
and focus on evaluation calls using Phase 2 inference pipeline and OOD augmentations.
"""
from typing import Iterable, Tuple, Dict
import os
from ..ood.augmentations import apply_augmentations
from ..ood.ood_eval import evaluate_dataset


def train_on_oia_evaluate_kaggle(kaggle_image_paths: Iterable[str], kaggle_labels: Iterable[int], weights_path: str | None = None, n_mc: int = 15, seed: int = 42) -> Dict:
    # Placeholder wrapper that evaluates on Kaggle using existing weights
    return evaluate_dataset(kaggle_image_paths, kaggle_labels, weights_path=weights_path, n_mc=n_mc, seed=seed)


def train_on_kaggle_evaluate_oia(oia_image_paths: Iterable[str], oia_labels: Iterable[int], weights_path: str | None = None, n_mc: int = 15, seed: int = 42) -> Dict:
    # Placeholder wrapper that evaluates on OIA using existing weights
    return evaluate_dataset(oia_image_paths, oia_labels, weights_path=weights_path, n_mc=n_mc, seed=seed)
