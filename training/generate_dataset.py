import cv2
import numpy as np
import os
import random


# =========================================
# SETTINGS
# =========================================

IMAGE_SIZE = 128
IMAGES_PER_CLASS = 1000

DATASET_DIR = "dataset"


# =========================================
# Create a blank image
# =========================================

def create_canvas():

    return np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )


# =========================================
# Add random transformation
# =========================================

def transform_image(image):

    # Random rotation
    angle = random.randint(-20, 20)

    center = (
        IMAGE_SIZE // 2,
        IMAGE_SIZE // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        random.uniform(0.85, 1.15)
    )

    image = cv2.warpAffine(
        image,
        matrix,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Random translation
    tx = random.randint(-10, 10)
    ty = random.randint(-10, 10)

    translation_matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    image = cv2.warpAffine(
        image,
        translation_matrix,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Slight blur sometimes
    if random.random() < 0.2:

        image = cv2.GaussianBlur(
            image,
            (3, 3),
            0
        )

    return image


# =========================================
# Draw Circle
# =========================================

def draw_circle():

    image = create_canvas()

    center = (
        IMAGE_SIZE // 2 + random.randint(-5, 5),
        IMAGE_SIZE // 2 + random.randint(-5, 5)
    )

    radius = random.randint(25, 40)

    thickness = random.randint(3, 7)

    cv2.circle(
        image,
        center,
        radius,
        255,
        thickness
    )

    return image


# =========================================
# Draw Square
# =========================================

def draw_square():

    image = create_canvas()

    center_x = IMAGE_SIZE // 2
    center_y = IMAGE_SIZE // 2

    size = random.randint(45, 70)

    x1 = center_x - size // 2
    y1 = center_y - size // 2

    x2 = center_x + size // 2
    y2 = center_y + size // 2

    thickness = random.randint(3, 7)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        255,
        thickness
    )

    return image


# =========================================
# Draw Triangle
# =========================================

def draw_triangle():

    image = create_canvas()

    center_x = IMAGE_SIZE // 2
    center_y = IMAGE_SIZE // 2

    size = random.randint(45, 65)

    points = np.array([
        [
            center_x,
            center_y - size // 2
        ],
        [
            center_x - size // 2,
            center_y + size // 2
        ],
        [
            center_x + size // 2,
            center_y + size // 2
        ]
    ])

    thickness = random.randint(3, 7)

    cv2.polylines(
        image,
        [points],
        True,
        255,
        thickness
    )

    return image


# =========================================
# Draw Star
# =========================================

def draw_star():

    image = create_canvas()

    center_x = IMAGE_SIZE // 2
    center_y = IMAGE_SIZE // 2

    outer_radius = random.randint(30, 40)
    inner_radius = random.randint(12, 20)

    points = []

    for i in range(10):

        angle = -np.pi / 2 + i * np.pi / 5

        if i % 2 == 0:
            radius = outer_radius
        else:
            radius = inner_radius

        x = int(
            center_x +
            radius * np.cos(angle)
        )

        y = int(
            center_y +
            radius * np.sin(angle)
        )

        points.append([x, y])

    points = np.array(points)

    thickness = random.randint(3, 6)

    cv2.polylines(
        image,
        [points],
        True,
        255,
        thickness
    )

    return image


# =========================================
# Draw Heart
# =========================================

def draw_heart():

    image = create_canvas()

    center_x = IMAGE_SIZE // 2
    center_y = IMAGE_SIZE // 2

    points = []

    for t in np.linspace(
        0,
        2 * np.pi,
        100
    ):

        x = 16 * np.sin(t) ** 3

        y = (
            13 * np.cos(t)
            - 5 * np.cos(2 * t)
            - 2 * np.cos(3 * t)
            - np.cos(4 * t)
        )

        scale = random.uniform(2.0, 2.5)

        px = int(
            center_x + x * scale
        )

        py = int(
            center_y - y * scale
        )

        points.append([px, py])

    points = np.array(points)

    thickness = random.randint(3, 6)

    cv2.polylines(
        image,
        [points],
        True,
        255,
        thickness
    )

    return image


# =========================================
# Shape generators
# =========================================

SHAPES = {

    "circle": draw_circle,

    "square": draw_square,

    "triangle": draw_triangle,

    "star": draw_star,

    "heart": draw_heart
}


# =========================================
# Generate dataset
# =========================================

def generate_dataset():

    print("\nGenerating AirCanvas dataset...\n")

    for shape_name, draw_function in SHAPES.items():

        folder = os.path.join(
            DATASET_DIR,
            shape_name
        )

        os.makedirs(
            folder,
            exist_ok=True
        )

        print(
            f"Generating {shape_name}..."
        )

        for i in range(IMAGES_PER_CLASS):

            # Create shape
            image = draw_function()

            # Apply random transformations
            image = transform_image(image)

            # File name
            filename = os.path.join(
                folder,
                f"{shape_name}_{i + 1:04d}.png"
            )

            # Save image
            cv2.imwrite(
                filename,
                image
            )

        print(
            f"✓ {shape_name}: "
            f"{IMAGES_PER_CLASS} images"
        )

    print("\n=================================")
    print("Dataset generation complete!")
    print("=================================")


# =========================================
# Run
# =========================================

if __name__ == "__main__":

    generate_dataset()