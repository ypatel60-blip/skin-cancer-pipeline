# Standalone HAM10000 TensorFlow pipeline (train / eval / predict + optional marker removal).
#
# Required packages (install if missing):
#   pip install tensorflow datasets pillow requests numpy opencv-python
# Optional (training curves figure):
#   pip install matplotlib
#
# Examples (run from project root; use your own image path for predict, not a generated figure):
#   python "mainpipeline(1).py" train --epochs 5
#   python "mainpipeline(1).py" eval --model ham10000_resnet50v2.keras
#   python "mainpipeline(1).py" predict "C:\Users\yp120\OneDrive\Desktop\project\image1.png" --model ham10000_resnet50v2.keras --use_inpaint --save_processed

from __future__ import annotations

import os

# Must run before importing TensorFlow so C++ logging initializes with this level (less duplicate stderr noise).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import io
import json
import random
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple, Union

import cv2
import numpy as np
import requests
import tensorflow as tf
from datasets import load_dataset
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------------------------
# A) Config
# ------------------------------------------------------------------------------

DATASET_ID = "kuchikihater/HAM10000"
SPLIT = "train"
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
TRAIN_SAMPLES_PER_EPOCH = 6000
VAL_SAMPLES_PER_EPOCH = 1200
DEFAULT_MODEL_PATH = "ham10000_resnet50v2.keras"
DEFAULT_WEIGHTS_PATH = "ham10000_resnet50v2.weights.h5"
DEFAULT_OUT_DIR = "eval_out"
FIGURES_DIR = "figures"
PROCESSED_IMAGE_PATH = os.path.join(FIGURES_DIR, "processed_image.png")

LR = 2e-4

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

LABEL2ID: Dict[str, int] = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "bcc": 3,
    "akiec": 4,
    "vasc": 5,
    "df": 6,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

_session = requests.Session()


# ------------------------------------------------------------------------------
# B) Dataset loading
# ------------------------------------------------------------------------------


def _get_label(ex: dict) -> Union[int, None]:
    """Resolve integer class id from a Hugging Face row (same logic as original data_loader)."""
    for k in ["dx", "label", "class", "target"]:
        if k in ex:
            v = ex[k]
            if isinstance(v, int):
                return v
            if isinstance(v, str):
                return LABEL2ID.get(v, None)
            if isinstance(v, list) and v and isinstance(v[0], str):
                return LABEL2ID.get(v[0], None)

    md = ex.get("meta", {}) or ex.get("metadata", {})
    if isinstance(md, dict):
        dx = md.get("dx") or md.get("diagnosis")
        if isinstance(dx, str):
            return LABEL2ID.get(dx, None)
    return None


def _load_image_from_url(url: str) -> Image.Image:
    r = _session.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def _get_pil_image(ex: dict) -> Union[Image.Image, None]:
    img_field = ex.get("image") or ex.get("img")

    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")

    if isinstance(img_field, dict) and "url" in img_field and isinstance(img_field["url"], str):
        return _load_image_from_url(img_field["url"])

    if isinstance(img_field, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(img_field)).convert("RGB")
        except Exception:
            return None

    url = ex.get("image_url") or ex.get("url")
    if isinstance(url, str):
        return _load_image_from_url(url)

    return None


def ham_generator(max_samples: int, split: str = SPLIT) -> Iterator[Tuple[np.ndarray, int]]:
    """Stream (image_uint8, label_id) pairs from Hugging Face."""
    ds = load_dataset(DATASET_ID, split=split, streaming=True)
    count = 0

    for ex in ds:
        if count >= max_samples:
            break
        try:
            label = _get_label(ex)
            if label is None:
                continue

            pil_img = _get_pil_image(ex)
            if pil_img is None:
                continue

            img_np = np.array(pil_img, dtype=np.uint8)
            yield img_np, label
            count += 1
        except Exception:
            continue


def _preprocess_train(image: tf.Tensor) -> tf.Tensor:
    """Resize, light augmentation, ImageNet normalize (training)."""
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)

    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


def _preprocess_val(image: tf.Tensor) -> tf.Tensor:
    """Resize, scale, ImageNet normalize (validation — deterministic)."""
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


def build_dataset(
    max_samples: int,
    split: str = SPLIT,
    augment: bool = True,
    shuffle: bool = True,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """tf.data pipeline: uint8 images + int labels → normalized batches + one-hot labels."""
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor] = _preprocess_train if augment else _preprocess_val

    def gen():
        yield from ham_generator(max_samples, split)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=min(max_samples, 1000), seed=SEED)

    ds = ds.map(
        lambda img, lbl: (
            preprocess_fn(img),
            tf.one_hot(lbl, NUM_CLASSES),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_train_dataset(
    max_samples: int = TRAIN_SAMPLES_PER_EPOCH,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    return build_dataset(max_samples, augment=True, shuffle=True, batch_size=batch_size)


def build_val_dataset(
    max_samples: int = VAL_SAMPLES_PER_EPOCH,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    return build_dataset(max_samples, augment=False, shuffle=False, batch_size=batch_size)


# ------------------------------------------------------------------------------
# C) Marker-removal preprocessing (OpenCV, optional)
# ------------------------------------------------------------------------------


def _ensure_parent_dir(file_path: str) -> None:
    """Create the parent directory of a file path if it is non-empty (no-op for bare filenames)."""
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_processed_image(x01: tf.Tensor, path: str = PROCESSED_IMAGE_PATH) -> None:
    """Save float32 RGB [0,1] tensor as PNG (creates parent directory)."""
    _ensure_parent_dir(path)
    img = tf.clip_by_value(x01, 0.0, 1.0).numpy()
    img_u8 = (img * 255).astype(np.uint8)
    Image.fromarray(img_u8).save(path)


def inpaint_fn(x01: tf.Tensor) -> tf.Tensor:
    """
    Remove blue/purple skin-marker artifacts using HSV thresholds + cv2.inpaint (TELEA).

    Input/output: float32 RGB, spatial shape [H, W, 3], values in [0, 1].
    """
    img = tf.clip_by_value(x01, 0.0, 1.0).numpy()
    img_u8 = (img * 255).astype(np.uint8)

    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([80, 50, 50], dtype=np.uint8)
    upper_blue = np.array([120, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_purple = np.array([120, 50, 50], dtype=np.uint8)
    upper_purple = np.array([170, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_purple, upper_purple)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    result_bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    return tf.convert_to_tensor(rgb, dtype=tf.float32)


def preprocess_with_optional_inpaint(
    img_uint8: np.ndarray,
    use_inpaint: bool = False,
    save_output: bool = False,
) -> tf.Tensor:
    """
    Resize → [0,1] float → optional marker removal → optional save → ImageNet normalization.

    Returns a tensor ready for the model: shape [IMG_SIZE, IMG_SIZE, 3], normalized.
    """
    x = tf.convert_to_tensor(img_uint8, dtype=tf.uint8)
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    x = tf.cast(x, tf.float32) / 255.0

    if use_inpaint:
        x = inpaint_fn(x)
        x = tf.clip_by_value(x, 0.0, 1.0)
        if save_output:
            save_processed_image(x)

    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


# ------------------------------------------------------------------------------
# D) Model
# ------------------------------------------------------------------------------


def build_model(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE) -> keras.Model:
    """ResNet50V2 backbone + GAP + Dropout + linear logits (no softmax)."""
    base = keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling=None,
    )
    base.trainable = True

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image_input")
    x = base(inputs)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation=None, name="logits")(x)

    return keras.Model(inputs, outputs, name="HAM10000_ResNet50V2")


# ------------------------------------------------------------------------------
# E) Training
# ------------------------------------------------------------------------------


def _maybe_plot_training_history(history: keras.callbacks.History, out_path: str) -> None:
    """Save loss / accuracy curves if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    h = history.history
    if not h:
        return

    _ensure_parent_dir(out_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if "loss" in h:
        axes[0].plot(h["loss"], label="train_loss")
    if "val_loss" in h:
        axes[0].plot(h["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if "categorical_accuracy" in h:
        axes[1].plot(h["categorical_accuracy"], label="train_acc")
    if "val_categorical_accuracy" in h:
        axes[1].plot(h["val_categorical_accuracy"], label="val_acc")
    axes[1].set_title("Categorical accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves: {out_path}")


def train_model(
    epochs: int = 5,
    batch_size: int = BATCH_SIZE,
    train_samples: int = TRAIN_SAMPLES_PER_EPOCH,
    val_samples: int = VAL_SAMPLES_PER_EPOCH,
    model_out: str = DEFAULT_MODEL_PATH,
    weights_out: str = DEFAULT_WEIGHTS_PATH,
    learning_rate: float = LR,
) -> keras.callbacks.History:
    """
    Build datasets, compile model, fit, save .keras + .weights.h5, return Keras History.
    """
    print("TensorFlow version:", tf.__version__)
    print("Building streaming datasets …")
    train_ds = build_train_dataset(max_samples=train_samples, batch_size=batch_size)
    val_ds = build_val_dataset(max_samples=val_samples, batch_size=batch_size)

    model = build_model()
    model.summary(line_length=90)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
    )

    _ensure_parent_dir(model_out)
    _ensure_parent_dir(weights_out)
    model.save(model_out)
    print(f"Saved model: {model_out}")

    model.save_weights(weights_out)
    print(f"Saved weights: {weights_out}")

    curves_path = os.path.join(FIGURES_DIR, "training_curves.png")
    _maybe_plot_training_history(history, curves_path)

    return history


# ------------------------------------------------------------------------------
# F) Evaluation
# ------------------------------------------------------------------------------


def load_model_file(model_path: str = DEFAULT_MODEL_PATH) -> keras.Model:
    """Load a saved Keras model (compile=False for inference)."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return keras.models.load_model(model_path, compile=False)


def evaluate_model(
    model: keras.Model,
    val_ds: tf.data.Dataset,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run validation batches, compute accuracy from logits, return confusion matrix data.

    Returns:
        accuracy, y_true_idx, y_pred_idx, confusion_matrix (num_classes × num_classes)
    """
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
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
        raise ValueError("No validation samples evaluated. Check dataset connectivity and --val_samples.")

    y_true_idx = np.concatenate(y_true_all, axis=0).astype(np.int32)
    y_pred_idx = np.concatenate(y_pred_all, axis=0).astype(np.int32)
    acc = correct / total

    cm = tf.math.confusion_matrix(
        y_true_idx,
        y_pred_idx,
        num_classes=NUM_CLASSES,
    ).numpy().astype(np.int32)

    return float(acc), y_true_idx, y_pred_idx, cm


def print_confusion_matrix(cm: np.ndarray) -> None:
    """Print CSV-style confusion matrix to stdout."""
    labels = [ID2LABEL[i] for i in range(NUM_CLASSES)]
    print("true\\pred," + ",".join(labels))
    for i, row in enumerate(cm):
        print(labels[i] + "," + ",".join(str(int(v)) for v in row))


def save_outputs(out_dir: Union[str, Path], cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Write numpy arrays, CSV confusion matrix, and class name JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "confusion_matrix.npy", cm)
    np.save(out / "y_true_idx.npy", y_true)
    np.save(out / "y_pred_idx.npy", y_pred)

    labels = [ID2LABEL[i] for i in range(NUM_CLASSES)]
    csv_lines = ["true\\pred," + ",".join(labels)]
    for i in range(cm.shape[0]):
        csv_lines.append(labels[i] + "," + ",".join(str(int(v)) for v in cm[i]))
    (out / "confusion_matrix.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (out / "class_names.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")


# ------------------------------------------------------------------------------
# G) Single-image prediction
# ------------------------------------------------------------------------------


def load_image_uint8(image_path: str) -> np.ndarray:
    """Load an RGB image from disk as uint8 H×W×3."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def format_probabilities(probs: np.ndarray) -> Dict[str, float]:
    """Map class names to softmax probabilities."""
    return {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}


def predict_image(
    model: keras.Model,
    image_path: str,
    *,
    use_inpaint: bool = False,
    save_processed: bool = False,
) -> Tuple[str, float, np.ndarray]:
    """
    Preprocess one image (optional marker removal), run classifier, softmax → label + probs.

    Processed [0,1] image is saved only when ``use_inpaint`` and ``save_processed`` are True.
    """
    img_uint8 = load_image_uint8(image_path)
    x = preprocess_with_optional_inpaint(
        img_uint8,
        use_inpaint=use_inpaint,
        save_output=bool(use_inpaint and save_processed),
    )
    x = tf.expand_dims(x, axis=0)

    logits = model(x, training=False)
    probs = tf.nn.softmax(logits, axis=-1)[0].numpy()

    pred_id = int(np.argmax(probs))
    pred_label = ID2LABEL[pred_id]
    confidence = float(probs[pred_id])
    return pred_label, confidence, probs


# ------------------------------------------------------------------------------
# H) CLI
# ------------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> None:
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        model_out=args.model_out,
        weights_out=args.weights_out,
    )


def _cmd_eval(args: argparse.Namespace) -> None:
    model = load_model_file(args.model)
    val_ds = build_val_dataset(max_samples=args.val_samples, batch_size=BATCH_SIZE)
    acc, y_true, y_pred, cm = evaluate_model(model, val_ds)

    print(f"Evaluated {len(y_true)} samples")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print_confusion_matrix(cm)

    save_outputs(args.out_dir, cm, y_true, y_pred)
    print(f"\nSaved outputs to: {Path(args.out_dir).resolve()}")


def _cmd_predict(args: argparse.Namespace) -> None:
    if args.save_processed and not args.use_inpaint:
        print("Note: --save_processed only applies when --use_inpaint is set (nothing saved).")

    model = load_model_file(args.model)
    label, conf, probs = predict_image(
        model,
        args.image,
        use_inpaint=args.use_inpaint,
        save_processed=args.save_processed,
    )
    prob_lines = [f"  {k}: {v:.4f}" for k, v in format_probabilities(probs).items()]
    block = "\n".join(
        [
            f"Predicted: {label}",
            f"Confidence: {conf:.4f}",
            "Probabilities:",
            *prob_lines,
        ]
    )
    print(block, flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HAM10000 ResNet50V2 pipeline: train | eval | predict (standalone).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train classifier and save .keras + .weights.h5")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    t.add_argument("--train_samples", type=int, default=TRAIN_SAMPLES_PER_EPOCH)
    t.add_argument("--val_samples", type=int, default=VAL_SAMPLES_PER_EPOCH)
    t.add_argument("--model_out", default=DEFAULT_MODEL_PATH)
    t.add_argument("--weights_out", default=DEFAULT_WEIGHTS_PATH)
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("eval", help="Evaluate on streaming validation subset")
    e.add_argument("--model", default=DEFAULT_MODEL_PATH)
    e.add_argument("--val_samples", type=int, default=VAL_SAMPLES_PER_EPOCH)
    e.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    e.set_defaults(func=_cmd_eval)

    pr = sub.add_parser("predict", help="Run single-image prediction")
    pr.add_argument(
        "image",
        help="Path to your RGB image (e.g. a clinical photo); avoid figures/processed_image.png unless you intend to classify that saved output",
    )
    pr.add_argument("--model", default=DEFAULT_MODEL_PATH)
    pr.add_argument(
        "--use_inpaint",
        action="store_true",
        help="Remove blue/purple markers (OpenCV) before normalization",
    )
    pr.add_argument(
        "--save_processed",
        action="store_true",
        help="Save marker-cleaned [0,1] image to figures/processed_image.png (requires --use_inpaint)",
    )
    pr.set_defaults(func=_cmd_predict)

    return p


# ------------------------------------------------------------------------------
# I) Main
# ------------------------------------------------------------------------------


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
