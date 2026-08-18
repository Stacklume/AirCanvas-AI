class GestureDetector:

    def get_finger_states(self, hand):
        """
        Returns the state of the four fingers:
        index, middle, ring, pinky

        True  = finger is raised
        False = finger is folded
        """

        fingers = []

        # Index finger
        if hand[8].y < hand[6].y:
            fingers.append(True)
        else:
            fingers.append(False)

        # Middle finger
        if hand[12].y < hand[10].y:
            fingers.append(True)
        else:
            fingers.append(False)

        # Ring finger
        if hand[16].y < hand[14].y:
            fingers.append(True)
        else:
            fingers.append(False)

        # Pinky finger
        if hand[20].y < hand[18].y:
            fingers.append(True)
        else:
            fingers.append(False)

        return fingers


    def get_gesture(self, hand):

        fingers = self.get_finger_states(hand)

        index, middle, ring, pinky = fingers

        # ☝️ Only index finger is raised
        if index and not middle and not ring and not pinky:
            return "DRAW"

        # ✌️ Index + middle fingers are raised
        elif index and middle and not ring and not pinky:
            return "SELECT"

        # ✊ All fingers folded
        elif not index and not middle and not ring and not pinky:
            return "ERASE"

        # 🖐️ All four fingers raised
        elif index and middle and ring and pinky:
            return "PAUSE"

        return "UNKNOWN"