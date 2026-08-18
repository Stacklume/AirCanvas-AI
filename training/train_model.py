import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


# =========================================
# SETTINGS
# =========================================

IMAGE_SIZE = 128

DATASET_DIR = "dataset"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sketch_classifier.keras"
)

CLASSES = [
    "circle",
    "square",
    "triangle",
    "star",
    "heart"
]

NUM_CLASSES = len(CLASSES)

EPOCHS = 15

BATCH_SIZE = 32


# =========================================
# Load Dataset
# =========================================

def load_dataset():

    images = []
    labels = []

    print("\nLoading dataset...\n")

    for label, class_name in enumerate(CLASSES):

        folder = os.path.join(
            DATASET_DIR,
            class_name
        )

        print(f"Loading {class_name}...")

        for filename in os.listdir(folder):

            filepath = os.path.join(
                folder,
                filename
            )

            image = cv2.imread(
                filepath,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            image = image.astype(
                np.float32
            ) / 255.0

            images.append(image)

            labels.append(label)

    images = np.array(images)

    labels = np.array(labels)

    # Add channel dimension
    images = images.reshape(
        -1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        1
    )

    print("\nDataset loaded!")

    print("Images:", images.shape)
    print("Labels:", labels.shape)

    return images, labels


# =========================================
# Build CNN
# =========================================

def build_model():

    model = tf.keras.Sequential([

        # First convolution
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
                1
            )
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),


        # Second convolution
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),


        # Third convolution
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),


        # Convert feature maps to vector
        tf.keras.layers.Flatten(),


        # Dense layer
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        # Prevent overfitting
        tf.keras.layers.Dropout(
            0.3
        ),


        # Output layer
        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])


    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


# =========================================
# Main Training
# =========================================

def main():

    # Load data
    X, y = load_dataset()


    # =====================================
    # Train/Test Split
    # =====================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    # =====================================
    # Validation Split
    # =====================================

    X_train, X_val, y_train, y_val = train_test_split(

        X_train,
        y_train,

        test_size=0.20,

        random_state=42,

        stratify=y_train
    )


    print("\nDataset split:")

    print(
        "Training:",
        X_train.shape[0]
    )

    print(
        "Validation:",
        X_val.shape[0]
    )

    print(
        "Testing:",
        X_test.shape[0]
    )


    # =====================================
    # Build Model
    # =====================================

    model = build_model()


    print("\nModel architecture:\n")

    model.summary()


    # =====================================
    # Train Model
    # =====================================

    print("\nStarting training...\n")


    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        verbose=1
    )


    # =====================================
    # Evaluate Model
    # =====================================

    print("\nEvaluating model...\n")


    test_loss, test_accuracy = model.evaluate(

        X_test,
        y_test,

        verbose=1
    )


    print(
        f"\nTest Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )


    # =====================================
    # Save Model
    # =====================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model.save(
        MODEL_PATH
    )


    print(
        f"\nModel saved to:"
        f"\n{MODEL_PATH}"
    )


# =========================================
# Run
# =========================================

if __name__ == "__main__":

    main()