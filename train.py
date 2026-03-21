# ==============================================================
# train.py
# TF/Keras equivalent of your PyTorch training pipeline:
#   train_and_save()    → same name, same structure, same print format
#   train_one_epoch()   → replaced by tf.GradientTape loop (same steps)
#   evaluate()          → kept as separate function (imported in evaluate.py)
#
# Your PyTorch training loop steps:
#   opt.zero_grad()  →  (automatic in TF inside GradientTape)
#   logits = model(x)
#   loss = criterion(logits, y)
#   loss.backward()  →  tape.gradient(loss, model.trainable_variables)
#   opt.step()       →  optimizer.apply_gradients(...)
#
# Loss function:
#   PyTorch: nn.CrossEntropyLoss()  — expects raw logits + integer labels
#   TF:      CategoricalCrossentropy(from_logits=True)
#            — expects raw logits + one-hot labels (one-hot done in data_loader)
#   Both are numerically equivalent (log-softmax + NLL internally).
#
# Optimizer:
#   PyTorch: torch.optim.AdamW(model.parameters(), lr=LR)
#   TF:      keras.optimizers.AdamW(learning_rate=LR)
#   Identical algorithm, same default weight_decay=0.01.
#
# Model saving:
#   PyTorch: torch.save(model.state_dict(), "file.pth")
#   TF:      model.save("file.keras")   — saves architecture + weights together
#            model.save_weights("file.weights.h5")  — weights only (like .pth)
# ==============================================================

import tensorflow as tf
from tensorflow import keras

from data_loader import (
    BATCH_SIZE,
    SEED,
    TRAIN_SAMPLES_PER_EPOCH,
    VAL_SAMPLES_PER_EPOCH,
    NUM_CLASSES,
    build_train_dataset,
    build_val_dataset,
)
from model import build_model

# ─────────────────────────── CONFIG ────────────────────────────
# Same values as your PyTorch CONFIG block
EPOCHS    = 5
LR        = 2e-4
SAVE_PATH = "ham10000_resnet50v2.keras"   # full SavedModel (arch + weights)
# ───────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)


# ───────────────────── TRAINING STEP ───────────────────────────

def train_one_epoch(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    optimizer: keras.optimizers.Optimizer,
    loss_fn: keras.losses.Loss,
) -> tuple[float, float]:
    """
    One full training epoch using GradientTape.

    Mirrors your PyTorch train_one_epoch() step by step:
        zero_grad  →  (implicit inside tape context)
        forward    →  model(x, training=True)
        loss       →  loss_fn(y, logits)
        backward   →  tape.gradient(loss, model.trainable_variables)
        step       →  optimizer.apply_gradients(...)

    Returns: (avg_loss, accuracy)  — same as your PyTorch version
    """
    total_loss = 0.0
    correct    = 0
    total      = 0

    for x_batch, y_batch in train_ds:
        # GradientTape records operations for automatic differentiation
        # This replaces: opt.zero_grad() ... loss.backward() ... opt.step()
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)   # training=True → Dropout ON

            # NOTE: y_batch is one-hot [B, 7]; loss_fn has from_logits=True
            loss = loss_fn(y_batch, logits)

        # Compute gradients — equivalent to loss.backward()
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply gradients — equivalent to opt.step()
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Accumulate metrics — same as your total_loss / correct / total counters
        batch_size  = tf.shape(x_batch)[0]
        total_loss += loss.numpy() * int(batch_size)

        preds    = tf.argmax(logits, axis=1)          # same as logits.argmax(1)
        labels   = tf.argmax(y_batch, axis=1)         # one-hot → integer index
        correct += int(tf.reduce_sum(tf.cast(preds == labels, tf.int32)).numpy())
        total   += int(batch_size)

    avg_loss = total_loss / max(total, 1)
    accuracy  = correct   / max(total, 1)
    return avg_loss, accuracy


# ─────────────────────── EVALUATION ────────────────────────────

def evaluate(model: keras.Model, val_ds: tf.data.Dataset) -> float:
    """
    Compute accuracy on a validation dataset.

    Mirrors your PyTorch evaluate() decorated with @torch.no_grad():
        model.eval() → model(x, training=False)  [Dropout OFF, BN uses running stats]
        logits.argmax(1) == y  →  preds == labels
        correct / total        →  same return value

    Returns: accuracy float in [0, 1]
    """
    correct = 0
    total   = 0

    for x_batch, y_batch in val_ds:
        # training=False → Dropout disabled, BatchNorm uses stored running stats
        logits = model(x_batch, training=False)

        preds  = tf.argmax(logits, axis=1)
        labels = tf.argmax(y_batch, axis=1)

        correct += int(tf.reduce_sum(tf.cast(preds == labels, tf.int32)).numpy())
        total   += int(tf.shape(x_batch)[0])

    return correct / max(total, 1)


# ─────────────────────── MAIN ENTRY ────────────────────────────

def train_and_save():
    """
    Full training pipeline — direct TF equivalent of your PyTorch train_and_save().

    Preserves:
      - same CONFIG values (EPOCHS=5, LR=2e-4, BATCH_SIZE=32)
      - same dataset sample limits (6000 train / 1200 val)
      - same loss: CrossEntropyLoss (from_logits=True) → AdamW
      - same print format:
          Epoch {e}/{E} | loss={:.4f} | train_acc={:.4f} | val_acc={:.4f}
      - saves model at the end (equivalent to torch.save(state_dict, path))
    """
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpus)}")

    # ── Datasets ─────────────────────────────────────────────
    print("Building streaming datasets …")
    train_ds = build_train_dataset(max_samples=TRAIN_SAMPLES_PER_EPOCH)
    val_ds   = build_val_dataset(max_samples=VAL_SAMPLES_PER_EPOCH)

    # ── Model ─────────────────────────────────────────────────
    model = build_model()
    model.summary(line_length=90)

    # ── Loss ──────────────────────────────────────────────────
    # Your PyTorch: nn.CrossEntropyLoss()
    # TF equivalent: CategoricalCrossentropy(from_logits=True)
    # Labels are one-hot (done in data_loader); logits are raw (no softmax in model)
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

    # ── Optimizer ─────────────────────────────────────────────
    # Your PyTorch: torch.optim.AdamW(model.parameters(), lr=LR)
    optimizer = keras.optimizers.AdamW(learning_rate=LR)

    # ── Epoch loop ────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_ds, optimizer, loss_fn)
        val_acc = evaluate(model, val_ds)

        # Same print format as your PyTorch code
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    # ── Save ──────────────────────────────────────────────────
    # Your PyTorch: torch.save(model.state_dict(), "ham10000_resnet18_stream.pth")
    # TF full model (architecture + weights, recommended):
    model.save(SAVE_PATH)
    print(f"Saved: {SAVE_PATH}")

    # Optionally save weights-only file (closer to PyTorch .pth behaviour):
    weights_path = SAVE_PATH.replace(".keras", ".weights.h5")
    model.save_weights(weights_path)
    print(f"Weights also saved: {weights_path}")


if __name__ == "__main__":
    train_and_save()
