import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import os

# =========================
# CONFIG
# =========================
IMG_PATH = r"C:\Users\yp120\OneDrive\Desktop\project\image1.png"
OUTPUT_DIR = "figures"
IMG_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# FIGURE 1: PIPELINE
# =========================
def make_figure1_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2)
    ax.axis("off")

    steps = [
        ("Input Image", 0.5),
        ("Preprocessing", 3.0),
        ("Inpainting", 5.5),
        ("Deep Learning\nModel", 8.0),
        ("Prediction", 10.5),
    ]

    for label, x in steps:
        box = FancyBboxPatch(
            (x, 0.7), 1.6, 0.6,
            boxstyle="round,pad=0.05",
            linewidth=1.5,
            fill=False
        )
        ax.add_patch(box)
        ax.text(x + 0.8, 1.0, label, ha="center", va="center", fontsize=12, fontweight="bold")

    for i in range(len(steps) - 1):
        x1 = steps[i][1] + 1.6
        x2 = steps[i + 1][1]
        ax.annotate(
            "",
            xy=(x2, 1.0),
            xytext=(x1, 1.0),
            arrowprops=dict(arrowstyle="->", lw=1.8)
        )

    plt.title("Figure 1. Overview of the Proposed Skin Cancer Detection Pipeline", fontsize=14, pad=20)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "figure1_pipeline.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)

# =========================
# FIGURE 2: ACCURACY CURVE
# =========================
def make_figure2_accuracy():
    epochs = [1, 2, 3, 4, 5]
    train_acc = [0.8027, 0.8030, 0.8142, 0.8305, 0.8282]
    val_acc = [0.0775, 0.0283, 0.0225, 0.1358, 0.3592]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker="o", linewidth=2, label="Training Accuracy")
    plt.plot(epochs, val_acc, marker="o", linewidth=2, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Figure 2. Training and Validation Accuracy Across Epochs")
    plt.xticks(epochs)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "figure2_accuracy.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)

# =========================
# FIGURE 3: CONFUSION MATRIX
# =========================
def make_figure3_confusion_matrix():
    labels = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

    cm = np.array([
        [0, 0,   0,   0,  0,  0, 0],
        [0, 0,   0,   0,  0,  0, 0],
        [0, 81, 431,  0, 336, 238, 8],
        [0, 38,  3,   0, 14, 48, 2],
        [0, 0,   0,   0,  0,  0, 0],
        [0, 0,   0,   0,  1,  0, 0],
        [0, 0,   0,   0,  0,  0, 0],
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Figure 3. Confusion Matrix for Skin Lesion Classification")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "figure3_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)

# =========================
# FIGURE 4: ORIGINAL VS PROCESSED
# =========================
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

def process_image_tf(img_uint8):
    x = tf.convert_to_tensor(img_uint8, dtype=tf.uint8)
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    x = tf.cast(x, tf.float32) / 255.0

    # simple enhancement / cleaning placeholder
    x = tf.image.adjust_contrast(x, 1.1)
    x = tf.image.adjust_brightness(x, 0.02)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x

def save_processed_image(processed_tensor, save_path):
    arr = (processed_tensor.numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(save_path)

def make_figure4_original_vs_processed():
    img_np = load_image(IMG_PATH)
    processed = process_image_tf(img_np)

    processed_path = os.path.join(OUTPUT_DIR, "processed_image.png")
    save_processed_image(processed, processed_path)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed.numpy())
    plt.title("Processed Image")
    plt.axis("off")

    plt.suptitle("Figure 4. Original and Processed Dermoscopic Image", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "figure4_original_vs_processed.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)
    print("Also saved processed image:", processed_path)

# =========================
# RUN ALL FIGURES
# =========================
make_figure1_pipeline()
make_figure2_accuracy()
make_figure3_confusion_matrix()
make_figure4_original_vs_processed()

print("\nAll figures generated successfully in folder:", OUTPUT_DIR)