import io
import random
import requests
import numpy as np
import tensorflow as tf
from PIL import Image
from datasets import load_dataset

# ─────────────────────────── CONFIG ────────────────────────────
DATASET_ID = "kuchikihater/HAM10000"
SPLIT = "train"
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
TRAIN_SAMPLES_PER_EPOCH = 6000
VAL_SAMPLES_PER_EPOCH = 1200

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

LABEL2ID = {"nv": 0, "mel": 1, "bkl": 2, "bcc": 3, "akiec": 4, "vasc": 5, "df": 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)

_session = requests.Session()


def preprocess_train(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.random_flip_left_right(image)

    # Simple safe augmentation using tf.image only
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)

    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


def preprocess_val(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


def _get_label(ex: dict):
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


def _load_image_from_url(url: str):
    r = _session.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def _get_pil_image(ex: dict):
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


def ham_generator(max_samples: int, split: str = SPLIT):
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


def build_dataset(
    max_samples: int,
    split: str = SPLIT,
    augment: bool = True,
    shuffle: bool = True,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    preprocess_fn = preprocess_train if augment else preprocess_val

    ds = tf.data.Dataset.from_generator(
        lambda: ham_generator(max_samples, split),
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