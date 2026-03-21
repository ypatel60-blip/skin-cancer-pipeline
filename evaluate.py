"""
evaluate.py
Simple evaluation for the HAM10000 classifier.

Compatible with this pipeline:
- Validation dataset comes from data_loader.build_val_dataset() with one-hot labels
- Model outputs raw logits
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

import data_loader as dl

DEFAULT_MODEL_PATH = "ham10000_resnet50v2.keras"
DEFAULT_OUT_DIR = "eval_out"


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)


def evaluate_model(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      accuracy
      y_true_idx (N,)
      y_pred_idx (N,)
      confusion_matrix (C,C)
    """
    y_true_all = []
    y_pred_all = []

    correct = 0
    total = 0

    for x_batch, y_batch_onehot in val_ds:
        logits = model(x_batch, training=False)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        labels = tf.argmax(y_batch_onehot, axis=-1, output_type=tf.int32)

        correct += int(tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32)).numpy())
        total += int(tf.shape(preds)[0])

        y_true_all.append(labels.numpy())
        y_pred_all.append(preds.numpy())

    if total == 0:
        raise ValueError("No validation samples evaluated. Check streaming dataset connectivity.")

    y_true_idx = np.concatenate(y_true_all, axis=0).astype(np.int32)
    y_pred_idx = np.concatenate(y_pred_all, axis=0).astype(np.int32)

    acc = correct / total
    cm = tf.math.confusion_matrix(
        y_true_idx,
        y_pred_idx,
        num_classes=dl.NUM_CLASSES,
    ).numpy().astype(np.int32)

    return float(acc), y_true_idx, y_pred_idx, cm


def print_confusion_matrix(cm: np.ndarray) -> None:
    labels = [dl.ID2LABEL[i] for i in range(dl.NUM_CLASSES)]
    header = "true\\pred," + ",".join(labels)
    print(header)
    for i, row in enumerate(cm):
        print(labels[i] + "," + ",".join(str(int(v)) for v in row))


def save_outputs(out_dir: Path, cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "confusion_matrix.npy", cm)
    np.save(out_dir / "y_true_idx.npy", y_true)
    np.save(out_dir / "y_pred_idx.npy", y_pred)

    labels = [dl.ID2LABEL[i] for i in range(dl.NUM_CLASSES)]

    (out_dir / "confusion_matrix.csv").write_text(
        "true\\pred," + ",".join(labels) + "\n"
        + "\n".join(labels[i] + "," + ",".join(str(int(v)) for v in cm[i]) for i in range(cm.shape[0]))
    )

    (out_dir / "class_names.json").write_text(json.dumps(labels, indent=2))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to saved Keras model (.keras)")
    parser.add_argument("--val_samples", type=int, default=dl.VAL_SAMPLES_PER_EPOCH, help="Validation samples to stream")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Directory to save confusion matrix + predictions")
    args = parser.parse_args()

    model = load_model(args.model)
    val_ds = dl.build_val_dataset(max_samples=args.val_samples)

    acc, y_true, y_pred, cm = evaluate_model(model, val_ds)

    print(f"Evaluated {len(y_true)} samples")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print_confusion_matrix(cm)

    save_outputs(Path(args.out_dir), cm, y_true, y_pred)
    print(f"\nSaved outputs to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()