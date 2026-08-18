import cv2


class Toolbar:

    def __init__(self):

        # =========================================
        # COLORS
        # =========================================

        self.colors = {
            "RED": (0, 0, 255),
            "BLUE": (255, 0, 0),
            "GREEN": (0, 255, 0),
            "YELLOW": (0, 255, 255),
            "WHITE": (255, 255, 255)
        }


        # =========================================
        # BRUSH SIZES
        # =========================================

        self.sizes = {
            "SMALL": 5,
            "MEDIUM": 10,
            "LARGE": 20
        }


        # =========================================
        # BUTTON LIST
        # =========================================

        self.buttons = []

        x = 20


        # =========================================
        # COLOR BUTTONS
        # =========================================

        for name, color in self.colors.items():

            self.buttons.append({
                "type": "color",
                "name": name,
                "value": color,
                "x1": x,
                "y1": 10,
                "x2": x + 60,
                "y2": 70
            })

            x += 75


        # =========================================
        # BRUSH SIZE BUTTONS
        # =========================================

        for name, size in self.sizes.items():

            self.buttons.append({
                "type": "size",
                "name": name,
                "value": size,
                "x1": x,
                "y1": 10,
                "x2": x + 70,
                "y2": 70
            })

            x += 85


        # =========================================
        # CLEAR
        # =========================================

        self.buttons.append({
            "type": "action",
            "name": "CLEAR",
            "value": None,
            "x1": x,
            "y1": 10,
            "x2": x + 80,
            "y2": 70
        })

        x += 90


        # =========================================
        # UNDO
        # =========================================

        self.buttons.append({
            "type": "action",
            "name": "UNDO",
            "value": None,
            "x1": x,
            "y1": 10,
            "x2": x + 80,
            "y2": 70
        })

        x += 90


        # =========================================
        # REDO
        # =========================================

        self.buttons.append({
            "type": "action",
            "name": "REDO",
            "value": None,
            "x1": x,
            "y1": 10,
            "x2": x + 80,
            "y2": 70
        })

        x += 90


        # =========================================
        # AI RECOGNIZE
        # =========================================

        self.buttons.append({
            "type": "action",
            "name": "AI",
            "value": None,
            "x1": x,
            "y1": 10,
            "x2": x + 80,
            "y2": 70
        })


    # =========================================
    # DRAW TOOLBAR
    # =========================================

    def draw(self, frame):

        for button in self.buttons:

            x1 = button["x1"]
            y1 = button["y1"]
            x2 = button["x2"]
            y2 = button["y2"]


            # ---------------------------------
            # COLOR
            # ---------------------------------

            if button["type"] == "color":

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    button["value"],
                    -1
                )


            # ---------------------------------
            # SIZE
            # ---------------------------------

            elif button["type"] == "size":

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (50, 50, 50),
                    -1
                )

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    button["value"] // 2,
                    (255, 255, 255),
                    -1
                )


            # ---------------------------------
            # ACTION
            # ---------------------------------

            elif button["type"] == "action":

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (50, 50, 50),
                    -1
                )

                cv2.putText(
                    frame,
                    button["name"],
                    (x1 + 10, y1 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


            # ---------------------------------
            # BORDER
            # ---------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

        return frame


    # =========================================
    # CHECK SELECTION
    # =========================================

    def check_selection(self, x, y):

        for button in self.buttons:

            if (
                button["x1"] <= x <= button["x2"]
                and
                button["y1"] <= y <= button["y2"]
            ):

                return (
                    button["type"],
                    button["name"],
                    button["value"]
                )

        return None, None, None