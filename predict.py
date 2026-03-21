"""
predict.py
Single-image inference for HAM10000 classifier.
"""

from __future__ import annotations

from typing import Tuple
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from PIL import Image

import data_loader as dl

DEFAULT_MODEL_PATH = "ham10000_resnet50v2.keras"


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)


def load_image_uint8(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def preprocess_for_inference(img_uint8: np.ndarray) -> tf.Tensor:
    x = tf.convert_to_tensor(img_uint8, dtype=tf.uint8)
    x = dl.preprocess_val(x)
    return x


def predict_image(
    model: tf.keras.Model,
    image_path: str,
) -> Tuple[str, float, np.ndarray]:
    img_uint8 = load_image_uint8(image_path)
    x = preprocess_for_inference(img_uint8)
    x = tf.expand_dims(x, axis=0)  # [1, 224, 224, 3]

    logits = model(x, training=False)
    probs = tf.nn.softmax(logits, axis=-1)[0].numpy()

    pred_id = int(np.argmax(probs))
    pred_label = dl.ID2LABEL[pred_id]
    confidence = float(probs[pred_id])

    return pred_label, confidence, probs


def format_probabilities(probs: np.ndarray) -> dict[str, float]:
    return {dl.ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image (jpg/png)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to saved Keras model (.keras)")
    args = parser.parse_args()

    model = load_model(args.model)
    label, conf, probs = predict_image(model, args.image)

    print(f"Predicted: {label}")
    print(f"Confidence: {conf:.4f}")
    print("Probabilities:")
    print(format_probabilities(probs))


if __name__ == "__main__":
    main()