import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:

    def __init__(self):

        base_options = python.BaseOptions(
            model_asset_path="models/hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def find_hands(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int(
            cv2.getTickCount() /
            cv2.getTickFrequency() *
            1000
        )

        result = self.detector.detect_for_video(
            mp_image,
            timestamp
        )

        return frame, result