from fileinput import filename

import cv2
import numpy as np


class AirCanvas:

    def __init__(self, width, height):

        self.width = width
        self.height = height

        # All completed strokes
        self.strokes = []

        # Stroke currently being drawn
        self.current_stroke = None

        # Strokes removed by undo
        self.redo_stack = []

        # Brush settings
        self.brush_color = (255, 255, 255)
        self.brush_thickness = 8

        # Previous point
        self.previous_point = None


    def draw(self, point):

        # Finger released / drawing stopped
        if point is None:

            if self.current_stroke:

                self.strokes.append(self.current_stroke)

                self.current_stroke = None

                # New drawing invalidates redo history
                self.redo_stack.clear()

            self.previous_point = None

            return


        x, y = point

        # Start a new stroke
        if self.current_stroke is None:

            self.current_stroke = {
                "type": "draw",
                "color": self.brush_color,
                "thickness": self.brush_thickness,
                "points": []
            }

        self.current_stroke["points"].append((x, y))

        self.previous_point = (x, y)


    def erase(self, point, radius=30):

        if point is None:

            if self.current_stroke:

                self.strokes.append(self.current_stroke)

                self.current_stroke = None

                self.redo_stack.clear()

            self.previous_point = None

            return


        x, y = point

        # Start eraser stroke
        if self.current_stroke is None:

            self.current_stroke = {
                "type": "erase",
                "color": (0, 0, 0),
                "thickness": radius * 2,
                "points": []
            }

        self.current_stroke["points"].append((x, y))

        self.previous_point = (x, y)


    def set_color(self, color):

        self.brush_color = color


    def set_brush_size(self, size):

        self.brush_thickness = size


    def undo(self):

        # Finish current stroke first
        if self.current_stroke:

            self.strokes.append(self.current_stroke)

            self.current_stroke = None

        if self.strokes:

            stroke = self.strokes.pop()

            self.redo_stack.append(stroke)


    def redo(self):

        if self.redo_stack:

            stroke = self.redo_stack.pop()

            self.strokes.append(stroke)


    def clear(self):

        # Save entire drawing for undo
        if self.strokes or self.current_stroke:

            if self.current_stroke:

                self.strokes.append(self.current_stroke)

                self.current_stroke = None

            self.redo_stack.append({
                "type": "clear",
                "strokes": self.strokes.copy()
            })

        self.strokes = []

        self.current_stroke = None

        self.previous_point = None
    def save(self, filename):
    
            drawing = self.get_canvas()
    
            cv2.imwrite(filename, drawing)
    
            return filename

    def get_canvas(self):

        # Create blank canvas
        canvas = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )

        # Draw all completed strokes
        for stroke in self.strokes:

            if stroke["type"] == "clear":
                continue

            points = stroke["points"]

            if len(points) == 1:

                cv2.circle(
                    canvas,
                    points[0],
                    stroke["thickness"] // 2,
                    stroke["color"],
                    -1
                )

            else:

                for i in range(1, len(points)):

                    cv2.line(
                        canvas,
                        points[i - 1],
                        points[i],
                        stroke["color"],
                        stroke["thickness"]
                    )


        # Draw current stroke
        if self.current_stroke:

            points = self.current_stroke["points"]

            for i in range(1, len(points)):

                cv2.line(
                    canvas,
                    points[i - 1],
                    points[i],
                    self.current_stroke["color"],
                    self.current_stroke["thickness"]
                )

        return canvas
    