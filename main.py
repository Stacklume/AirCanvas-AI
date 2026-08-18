import cv2
import time

from app.camera import HandTracker
from app.canvas import AirCanvas
from app.gestures import GestureDetector
from app.toolbar import Toolbar
from app.ai_classifier import AIClassifier


# =========================================
# CAMERA
# =========================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# =========================================
# COMPONENTS
# =========================================

tracker = HandTracker()
gesture_detector = GestureDetector()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

canvas = AirCanvas(width, height)

toolbar = Toolbar()
ai_classifier = AIClassifier()

ai_results = []


# =========================================
# TOOLBAR ACTION LOCK
# =========================================

action_locked = False


# =========================================
# FULLSCREEN
# =========================================

cv2.namedWindow(
    "AirCanvas AI",
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    "AirCanvas AI",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# =========================================
# MAIN LOOP
# =========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not access camera")
        break


    # =====================================
    # MIRROR CAMERA
    # =====================================

    frame = cv2.flip(frame, 1)


    # =====================================
    # HAND DETECTION
    # =====================================

    frame, result = tracker.find_hands(frame)

    gesture = "NO HAND"


    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        h, w, _ = frame.shape


        # =================================
        # DRAW HAND LANDMARKS
        # =================================

        for landmark in hand:

            landmark_x = int(
                landmark.x * w
            )

            landmark_y = int(
                landmark.y * h
            )

            cv2.circle(
                frame,
                (landmark_x, landmark_y),
                5,
                (0, 255, 0),
                -1
            )


        # =================================
        # GESTURE
        # =================================

        gesture = gesture_detector.get_gesture(hand)


        # =================================
        # INDEX FINGERTIP
        # =================================

        index_tip = hand[8]

        x = int(index_tip.x * w)
        y = int(index_tip.y * h)


        # =================================
        # RED FINGERTIP
        # =================================

        cv2.circle(
            frame,
            (x, y),
            12,
            (0, 0, 255),
            -1
        )


        # =================================
        # DRAW
        # =================================

        if gesture == "DRAW":

            canvas.draw((x, y))


        # =================================
        # ERASE
        # =================================

        elif gesture == "ERASE":

            canvas.erase((x, y))


        # =================================
        # SELECT
        # =================================

        elif gesture == "SELECT":

            # Stop drawing
            canvas.draw(None)


            # Check toolbar
            button_type, name, value = (
                toolbar.check_selection(x, y)
            )


            # -----------------------------
            # NOTHING SELECTED
            # -----------------------------

            if button_type is None:

                action_locked = False


            # -----------------------------
            # COLOR
            # -----------------------------

            elif button_type == "color":

                canvas.set_color(value)

                action_locked = False


            # -----------------------------
            # BRUSH SIZE
            # -----------------------------

            elif button_type == "size":

                canvas.set_brush_size(value)

                action_locked = False


            # -----------------------------
            # ACTION
            # -----------------------------

            elif button_type == "action":

                # Execute only once
                if not action_locked:


                    # =====================
                    # CLEAR
                    # =====================

                    if name == "CLEAR":

                        canvas.clear()

                        ai_results = []

                        print("Canvas cleared")


                    # =====================
                    # UNDO
                    # =====================

                    elif name == "UNDO":

                        canvas.undo()

                        print("Undo")


                    # =====================
                    # REDO
                    # =====================

                    elif name == "REDO":

                        canvas.redo()

                        print("Redo")


                    # =====================
                    # SAVE
                    # =====================

                    elif name == "SAVE":

                        filename = (
                            f"aircanvas_"
                            f"{int(time.time() * 1000)}"
                            f".png"
                        )

                        canvas.save(filename)

                        print(
                            f"Artwork saved as: "
                            f"{filename}"
                        )


                    # =====================
                    # AI
                    # =====================

                    elif name == "AI":

                        # Analyze separate shapes
                        ai_results = (
                            ai_classifier.predict_multiple(
                                canvas.get_canvas()
                            )
                        )


                        if ai_results:

                            print(
                                "\n=============================="
                            )

                            print(
                                "AI ANALYSIS"
                            )

                            print(
                                "=============================="
                            )


                            for result_item in ai_results:

                                print(
                                    f"{result_item['label']} "
                                    f"({result_item['confidence']:.2f}%)"
                                )


                            print(
                                "==============================\n"
                            )


                        else:

                            print(
                                "AI: No shapes detected"
                            )


                    # Lock after action
                    action_locked = True


        # =================================
        # OTHER GESTURE
        # =================================

        else:

            canvas.draw(None)

            action_locked = False


    # =====================================
    # NO HAND
    # =====================================

    else:

        canvas.draw(None)

        action_locked = False


    # =====================================
    # GET DRAWING
    # =====================================

    drawing = canvas.get_canvas()


    # =====================================
    # COMBINE CAMERA + DRAWING
    # =====================================

    combined = cv2.add(
        frame,
        drawing
    )


    # =====================================
    # DRAW TOOLBAR
    # =====================================

    toolbar.draw(combined)


    # =====================================
    # GESTURE TEXT
    # =====================================

    cv2.putText(
        combined,
        f"Gesture: {gesture}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # =====================================
    # AI RESULT PANEL
    # =====================================

    if ai_results:

        panel_x = 20
        panel_y = 160

        panel_width = 360

        panel_height = (
            70 + len(ai_results) * 55
        )


        # ---------------------------------
        # Panel background
        # ---------------------------------

        overlay = combined.copy()

        cv2.rectangle(
            overlay,
            (
                panel_x,
                panel_y
            ),
            (
                panel_x + panel_width,
                panel_y + panel_height
            ),
            (30, 30, 30),
            -1
        )


        # ---------------------------------
        # Transparency
        # ---------------------------------

        combined = cv2.addWeighted(
            overlay,
            0.85,
            combined,
            0.15,
            0
        )


        # ---------------------------------
        # Panel border
        # ---------------------------------

        cv2.rectangle(
            combined,
            (
                panel_x,
                panel_y
            ),
            (
                panel_x + panel_width,
                panel_y + panel_height
            ),
            (255, 255, 255),
            2
        )


        # ---------------------------------
        # Title
        # ---------------------------------

        cv2.putText(
            combined,
            "AI ANALYSIS",
            (
                panel_x + 20,
                panel_y + 35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ---------------------------------
        # Results
        # ---------------------------------

        for i, result_item in enumerate(
            ai_results
        ):

            text_y = (
                panel_y
                + 75
                + i * 55
            )


            label = result_item["label"]

            confidence = result_item["confidence"]


            cv2.putText(
                combined,
                label,
                (
                    panel_x + 20,
                    text_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.putText(
                combined,
                f"{confidence:.1f}%",
                (
                    panel_x + 220,
                    text_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


    # =====================================
    # DISPLAY
    # =====================================

    cv2.imshow(
        "AirCanvas AI",
        combined
    )


    # =====================================
    # QUIT
    # =====================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================================
# CLEANUP
# =========================================

cap.release()

cv2.destroyAllWindows()