import cv2
import numpy as np
import tensorflow as tf


# =========================================
# SETTINGS
# =========================================

IMAGE_SIZE = 128

MODEL_PATH = "models/sketch_classifier.keras"

CLASSES = [
    "circle",
    "square",
    "triangle",
    "star",
    "heart"
]


# =========================================
# Load Model
# =========================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# =========================================
# Predict Image
# =========================================

def predict_image(image_path):

    # Read image
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print(
            f"Could not read image: {image_path}"
        )

        return


    # =====================================
    # Find the actual drawing
    # =====================================

    # Anything brighter than black
    # is considered part of the drawing

    _, threshold = cv2.threshold(
        image,
        20,
        255,
        cv2.THRESH_BINARY
    )


    # Find drawing coordinates

    coordinates = cv2.findNonZero(
        threshold
    )


    if coordinates is None:

        print("No drawing detected.")

        return


    # =====================================
    # Crop drawing
    # =====================================

    x, y, w, h = cv2.boundingRect(
        coordinates
    )

    cropped = image[
        y:y + h,
        x:x + w
    ]


    # =====================================
    # Make square canvas
    # =====================================

    size = max(w, h)

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )


    # Center the drawing

    offset_x = (
        size - w
    ) // 2

    offset_y = (
        size - h
    ) // 2


    square[
        offset_y:offset_y + h,
        offset_x:offset_x + w
    ] = cropped


    # =====================================
    # Resize
    # =====================================

    image = cv2.resize(
        square,
        (IMAGE_SIZE, IMAGE_SIZE)
    )


    # =====================================
    # Normalize
    # =====================================

    image = image.astype(
        np.float32
    ) / 255.0


    # =====================================
    # Add CNN dimensions
    # =====================================

    image = image.reshape(
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        1
    )


    # =====================================
    # Prediction
    # =====================================

    predictions = model.predict(
        image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = CLASSES[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ]


    # =====================================
    # Display Results
    # =====================================

    print("\n==============================")

    print(
        f"Prediction: {predicted_class.upper()}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("==============================\n")


    print("All predictions:")

    for i, class_name in enumerate(CLASSES):

        print(
            f"{class_name:10s}: "
            f"{probabilities[i] * 100:.2f}%"
        )
# =========================================
# Test
# =========================================

if __name__ == "__main__":

    predict_image(
        "aircanvas_1787064370535.png"
    )