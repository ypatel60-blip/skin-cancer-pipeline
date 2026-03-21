"""
utils.py
Small helpers shared by predict/evaluate/pipeline_infer.

Keeps project consistent by re-exporting label mapping from data_loader.py
and providing lightweight formatting utilities.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

import data_loader as dl


# Re-export mappings (single source of truth)
LABEL2ID: Dict[str, int] = dl.LABEL2ID
ID2LABEL: Dict[int, str] = dl.ID2LABEL
NUM_CLASSES: int = dl.NUM_CLASSES
CLASS_NAMES: List[str] = [ID2LABEL[i] for i in range(NUM_CLASSES)]


def probs_to_dict(probs: np.ndarray) -> Dict[str, float]:
    """
    Convert a probability vector into a readable {class_name: prob} dict.
    """
    probs = np.asarray(probs).reshape(-1)
    return {ID2LABEL[i]: float(probs[i]) for i in range(min(len(probs), NUM_CLASSES))}


def pred_from_probs(probs: np.ndarray) -> tuple[str, float, int]:
    """
    Returns: (label_name, confidence, class_index)
    """
    probs = np.asarray(probs).reshape(-1)
    pred_id = int(np.argmax(probs))
    return ID2LABEL[pred_id], float(probs[pred_id]), pred_id

