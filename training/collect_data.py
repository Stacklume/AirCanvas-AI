import cv2
import os
import numpy as np


# =========================================
# SETTINGS
# =========================================

IMAGE_SIZE = 128

DATASET_DIR = "dataset"

CLASSES = [
    "circle",
    "square",
    "triangle",
    "star",
    "heart"
]


# =========================================
# Create dataset folders
# =========================================

for class_name in CLASSES:

    folder = os.path.join(
        DATASET_DIR,
        class_name
    )

    os.makedirs(
        folder,
        exist_ok=True
    )


# =========================================
# Get next image number
# =========================================

def get_next_number(folder):

    files = os.listdir(folder)

    numbers = []

    for file in files:

        if file.endswith(".png"):

            try:

                number = int(
                    file.split("_")[-1]
                    .split(".")[0]
                )

                numbers.append(number)

            except ValueError:

                pass

    if not numbers:

        return 1

    return max(numbers) + 1


# =========================================
# Preprocess drawing
# =========================================

def preprocess_drawing(canvas):

    # Convert to grayscale
    gray = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    # Find non-black pixels
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

    # Make square
    size = max(w, h)

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    # Center drawing
    offset_x = (size - w) // 2
    offset_y = (size - h) // 2

    square[
        offset_y:offset_y + h,
        offset_x:offset_x + w
    ] = cropped

    # Resize
    resized = cv2.resize(
        square,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    return resized


# =========================================
# Select class
# =========================================

print("\n===================================")
print("      AIR CANVAS DATA COLLECTOR")
print("===================================\n")

print("Choose a shape:\n")

for i, class_name in enumerate(CLASSES):

    print(
        f"{i + 1}. {class_name.upper()}"
    )

print("\n0. EXIT")

choice = input(
    "\nEnter your choice: "
)


try:

    choice = int(choice)

except ValueError:

    print("Invalid choice.")
    exit()


if choice == 0:

    exit()


if choice < 1 or choice > len(CLASSES):

    print("Invalid choice.")
    exit()


selected_class = CLASSES[
    choice - 1
]

folder = os.path.join(
    DATASET_DIR,
    selected_class
)


# =========================================
# Camera
# =========================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# =========================================
# Drawing canvas
# =========================================

canvas = np.zeros(
    (720, 1280, 3),
    dtype=np.uint8
)


drawing = False

last_x = None
last_y = None


# =========================================
# Mouse drawing
# =========================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global drawing
    global last_x
    global last_y

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        last_x = x
        last_y = y


    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            cv2.line(
                canvas,
                (last_x, last_y),
                (x, y),
                (255, 255, 255),
                6
            )

            last_x = x
            last_y = y


    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        last_x = None
        last_y = None


# =========================================
# Window
# =========================================

window_name = "Dataset Collector"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)


# =========================================
# Starting number
# =========================================

image_number = get_next_number(
    folder
)


print(
    f"\nCollecting: "
    f"{selected_class.upper()}"
)

print(
    "Draw using the mouse."
)

print(
    "Press S to SAVE."
)

print(
    "Press C to CLEAR."
)

print(
    "Press Q to EXIT."
)

print(
    f"Starting from image "
    f"{image_number:04d}"
)


# =========================================
# Main loop
# =========================================

while True:

    success, frame = cap.read()

    if not success:

        print(
            "Could not access camera."
        )

        break


    frame = cv2.flip(
        frame,
        1
    )


    # Combine camera + drawing
    display = cv2.add(
        frame,
        canvas
    )


    # =====================================
    # Instructions
    # =====================================

    cv2.putText(
        display,
        f"COLLECTING: {selected_class.upper()}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "S = SAVE    C = CLEAR    Q = EXIT",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # =====================================
    # Display
    # =====================================

    cv2.imshow(
        window_name,
        display
    )


    key = cv2.waitKey(1) & 0xFF


    # =====================================
    # SAVE
    # =====================================

    if key == ord("s"):

        processed = preprocess_drawing(
            canvas
        )

        if processed is None:

            print(
                "Nothing to save."
            )

            continue


        filename = os.path.join(
            folder,
            f"{selected_class}_"
            f"{image_number:04d}.png"
        )


        cv2.imwrite(
            filename,
            processed
        )


        print(
            f"Saved: {filename}"
        )


        image_number += 1


        # Clear for next drawing
        canvas[:] = 0


    # =====================================
    # CLEAR
    # =====================================

    elif key == ord("c"):

        canvas[:] = 0

        print(
            "Canvas cleared."
        )


    # =====================================
    # EXIT
    # =====================================

    elif key == ord("q"):

        break


# =========================================
# Cleanup
# =========================================

cap.release()

cv2.destroyAllWindows()

print(
    "\nData collection finished."
)