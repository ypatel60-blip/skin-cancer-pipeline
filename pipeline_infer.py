"""
pipeline_infer.py
Future end-to-end flow demo:

image -> preprocessing -> OPTIONAL inpainting -> model (logits) -> softmax -> prediction

This file is intentionally simple for a student project, and consistent with:
- data_loader.py preprocessing (resize + ImageNet normalization)
- model.py outputs (raw logits)
- train.py loss (CategoricalCrossentropy(from_logits=True), one-hot labels)
"""

from __future__ import annotations

import os
from typing import Callable, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import data_loader as dl
from utils import probs_to_dict, pred_from_probs


DEFAULT_MODEL_PATH = "ham10000_resnet50v2.keras"
PROCESSED_IMAGE_PATH = "figures/processed_image.png"


def save_processed_image(x01: tf.Tensor, path: str = PROCESSED_IMAGE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = tf.clip_by_value(x01, 0.0, 1.0).numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def inpaint_fn(x01: tf.Tensor) -> tf.Tensor:
    img = tf.clip_by_value(x01, 0.0, 1.0).numpy()
    img = (img * 255).astype(np.uint8)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([90, 40, 40], dtype=np.uint8)
    upper = np.array([160, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    result = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    return tf.convert_to_tensor(rgb, dtype=tf.float32)


def preprocess_with_optional_inpaint(
    img_uint8: np.ndarray,
    *,
    use_inpaint: bool,
    inpaint: Callable[[tf.Tensor], tf.Tensor],
) -> tf.Tensor:
    """
    Matches data_loader.preprocess_val, but exposes the inpainting insertion point.
    """
    x = tf.convert_to_tensor(img_uint8, dtype=tf.uint8)
    x = tf.image.resize(x, [dl.IMG_SIZE, dl.IMG_SIZE])
    x = tf.cast(x, tf.float32) / 255.0  # 0..1

    # ---- Marker removal (HSV mask + OpenCV inpaint), then save before ImageNet norm ----
    if use_inpaint:
        x = inpaint(x)
        x = tf.clip_by_value(x, 0.0, 1.0)
        save_processed_image(x)
    # ------------------------------------------------------------------------------------

    # Same normalization as preprocess_val
    x = (x - dl.IMAGENET_MEAN) / dl.IMAGENET_STD
    return x


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)


def load_image_uint8(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def pipeline_predict(
    model: tf.keras.Model,
    image_path: str,
    *,
    use_inpaint: bool = False,
    inpaint: Callable[[tf.Tensor], tf.Tensor] = inpaint_fn,
) -> Tuple[str, float, np.ndarray]:
    img_uint8 = load_image_uint8(image_path)
    x = preprocess_with_optional_inpaint(img_uint8, use_inpaint=use_inpaint, inpaint=inpaint)
    x = tf.expand_dims(x, axis=0)  # [1,224,224,3]

    logits = model(x, training=False)
    probs = tf.nn.softmax(logits, axis=-1)[0].numpy()

    label, conf, _ = pred_from_probs(probs)
    return label, conf, probs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image (jpg/png)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to saved Keras model (.keras)")
    parser.add_argument(
        "--use_inpaint",
        action="store_true",
        help="Remove blue/purple skin markers (HSV + OpenCV inpaint) and save figures/processed_image.png",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    label, conf, probs = pipeline_predict(model, args.image, use_inpaint=args.use_inpaint)

    print(f"Predicted: {label}")
    print(f"Confidence: {conf:.4f}")
    print("Probabilities:")
    print(probs_to_dict(probs))


if __name__ == "__main__":
    main()
