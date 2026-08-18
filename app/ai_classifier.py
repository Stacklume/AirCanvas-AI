import cv2
import numpy as np
import tensorflow as tf


class AIClassifier:

    def __init__(self):

        # =====================================
        # MODEL
        # =====================================

        self.model = tf.keras.models.load_model(
            "models/sketch_classifier.keras"
        )


        # =====================================
        # CLASSES
        # =====================================

        self.classes = [
            "circle",
            "square",
            "triangle",
            "star",
            "heart"
        ]

        self.image_size = 128

        print("AI Classifier loaded successfully!")


    # =========================================
    # PREPROCESS ONE SHAPE
    # =========================================

    def preprocess(self, canvas):

        # Convert to grayscale
        gray = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2GRAY
        )


        # Find white drawing
        _, threshold = cv2.threshold(
            gray,
            20,
            255,
            cv2.THRESH_BINARY
        )


        # Find drawing
        coordinates = cv2.findNonZero(
            threshold
        )


        if coordinates is None:

            return None


        # Bounding box
        x, y, w, h = cv2.boundingRect(
            coordinates
        )


        # Crop
        cropped = gray[
            y:y + h,
            x:x + w
        ]


        # =====================================
        # Make square
        # =====================================

        size = max(w, h)

        square = np.zeros(
            (size, size),
            dtype=np.uint8
        )


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

        resized = cv2.resize(
            square,
            (
                self.image_size,
                self.image_size
            )
        )


        # Normalize
        resized = (
            resized.astype(np.float32) / 255.0
        )


        # Add dimensions
        resized = resized.reshape(
            1,
            self.image_size,
            self.image_size,
            1
        )


        return resized


    # =========================================
    # PREDICT ONE IMAGE
    # =========================================

    def predict(self, canvas):

        processed = self.preprocess(
            canvas
        )


        if processed is None:

            return None, 0.0


        predictions = self.model.predict(
            processed,
            verbose=0
        )


        probabilities = predictions[0]


        index = np.argmax(
            probabilities
        )


        label = self.classes[index]

        confidence = (
            probabilities[index] * 100
        )


        return label, confidence


    # =========================================
    # DETECT MULTIPLE SHAPES
    # =========================================

    def predict_multiple(self, canvas):

        # Convert to grayscale
        gray = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2GRAY
        )


        # Threshold
        _, binary = cv2.threshold(
            gray,
            20,
            255,
            cv2.THRESH_BINARY
        )


        # =====================================
        # Find connected components
        # =====================================

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )


        results = []


        # =====================================
        # Process each component
        # =====================================

        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            area = stats[i, cv2.CC_STAT_AREA]


            # Ignore tiny noise
            if area < 100:

                continue


            # =================================
            # Create isolated component
            # =================================

            component = np.zeros_like(
                canvas
            )


            component[
                labels == i
            ] = canvas[
                labels == i
            ]


            # =================================
            # Predict component
            # =================================

            label, confidence = self.predict(
                component
            )


            if label is not None:

                results.append({
                    "label": label.upper(),
                    "confidence": confidence,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                })


        # Sort left-to-right
        results.sort(
            key=lambda item: item["x"]
        )


        return results