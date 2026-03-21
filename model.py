# ==============================================================
# model.py
# TF/Keras equivalent of your PyTorch build_model()
#
# Your PyTorch original:
#   model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
#   model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
#
# TF equivalent:
#   base = ResNet50V2(weights="imagenet", include_top=False)
#   x = GlobalAveragePooling2D()(base.output)   ← matches PyTorch avgpool
#   output = Dense(NUM_CLASSES)(x)              ← matches model.fc
#
# WHY ResNet50V2 instead of ResNet18?
#   Keras does not ship a pretrained ResNet18. The closest officially
#   supported pretrained ResNet in keras.applications is ResNet50V2.
#   ResNet50V2 is a stronger model (50 layers vs 18), so your accuracy
#   should be equal or better. The training interface is identical.
#   If you specifically need 18 layers, you could use:
#       pip install keras-cv   → keras_cv.models.ResNet18Backbone
#   but that adds complexity not needed for a student project.
#
# WHY from_logits=True in train.py?
#   Your PyTorch CrossEntropyLoss works on raw logits (no softmax needed).
#   We match this by using Dense with no activation (linear head) and
#   setting CategoricalCrossentropy(from_logits=True) in train.py.
#   Softmax is applied only at inference time in predict.py.
# ==============================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_loader import NUM_CLASSES, IMG_SIZE


def build_model(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE) -> keras.Model:
    """
    Builds a pretrained ResNet50V2 with a 7-class linear head.

    Architecture mirrors your PyTorch ResNet18:
        Input [224, 224, 3]
        → ResNet50V2 backbone (pretrained ImageNet, no top)
        → GlobalAveragePooling2D   ← equivalent to PyTorch AdaptiveAvgPool2d(1,1)
        → Dropout(0.3)             ← light regularisation (not in original,
                                      added because ResNet50V2 is larger than 18)
        → Dense(7, activation=None)  ← raw logits, same as PyTorch model.fc

    Entire backbone is trainable from epoch 1, matching your PyTorch
    approach where all parameters are updated by AdamW.
    """
    # Pretrained backbone — no classification top, no final pooling
    base = keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling=None,
    )
    base.trainable = True   # fine-tune everything, same as PyTorch

    # Functional API model
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image_input")

    # training=True keeps BatchNorm in train mode during model.fit
    x = base(inputs)

    # Global average pool — exact functional equivalent of PyTorch's avgpool
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

    # Dropout — only because ResNet50V2 is bigger than ResNet18
    # Remove this line if you want a strict 1:1 match with your original
    x = layers.Dropout(0.3, name="dropout")(x)

    # Linear head — raw logits, no softmax (matches PyTorch CrossEntropyLoss)
    outputs = layers.Dense(num_classes, activation=None, name="logits")(x)

    model = keras.Model(inputs, outputs, name="HAM10000_ResNet50V2")
    return model


if __name__ == "__main__":
    # Quick sanity check: print model summary
    model = build_model()
    model.summary(line_length=90)
    print(f"\nOutput shape (batch=2): {model(tf.zeros([2, 224, 224, 3])).shape}")
