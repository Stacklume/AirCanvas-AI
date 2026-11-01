# 🎨 AirCanvas AI — Gesture-Based Drawing & CNN Shape Recognition

**Draw in the air. Let AI recognize your creativity.**

AirCanvas AI is an AI-powered virtual drawing system that combines computer vision, real-time hand tracking, gesture recognition, and Convolutional Neural Network (CNN) classification. Users can draw shapes in the air using their index finger, while the system processes the drawing and predicts the shape with a confidence score.

🌐 **Live Browser Demo:** [Try AirCanvas AI](https://stacklume.github.io/AirCanvas-AI/)

## 🚀 Project Overview

Traditional drawing applications require physical input devices such as a mouse, stylus, or touchscreen. AirCanvas AI introduces a more interactive approach by allowing users to draw through hand gestures captured by a webcam.

The main desktop application follows an end-to-end pipeline, from capturing live video to classifying the final drawing using a trained CNN model.

## 🧠 System Architecture

```text
             Webcam
                ↓
          Hand Detection
                ↓
      Hand Landmark Tracking
                ↓
        Gesture Recognition
                ↓
         AirCanvas Drawing
                ↓
          Shape Extraction
                ↓
         CNN Classification
                ↓
    Shape + Confidence Score
```

### How the System Works

**1. Webcam Input**

The webcam captures real-time video frames of the user's hand. These frames provide the input for hand detection and gesture tracking.

**2. Hand Detection**

The computer vision pipeline identifies the hand in each frame so that its position and movements can be analyzed.

**3. Hand Landmark Tracking**

The system tracks key points, or landmarks, on the detected hand. These landmarks help determine the position of the index finger and other finger movements.

**4. Gesture Recognition**

The tracked landmarks are interpreted to determine the user's gesture. For example, an index-finger drawing gesture can control the drawing action, while other supported gestures can be used to interact with the canvas.

**5. AirCanvas Drawing**

The application maps the index finger's movement to canvas coordinates and draws strokes in real time. Users can create drawings without physically touching the screen.

**6. Shape Extraction**

Once the user finishes drawing, the drawing is prepared for classification. The relevant canvas content is extracted and converted into a representation suitable for the CNN model.

**7. CNN Classification**

The extracted shape is passed to a trained Convolutional Neural Network. The model analyzes visual features learned during training to predict the most likely shape class.

**8. Shape and Confidence Score**

The system outputs the predicted shape along with a confidence score indicating how strongly the model favors that prediction.

*Note: A model's confidence score is not necessarily the same as its actual probability of being correct.*

## ✨ Key Features

- **Real-Time Hand Detection:** Detects a user's hand through webcam input.
- **Hand Landmark Tracking:** Tracks hand keypoints to support gesture-based interaction.
- **Gesture-Based Drawing:** Enables hands-free drawing using index-finger movements.
- **Interactive Digital Canvas:** Draw directly on a virtual canvas.
- **Drawing Tools:** Browser version includes color selection, brush-size adjustment, eraser, undo, clear, and save controls.
- **Shape Extraction:** Prepares the drawn content for classification in the main recognition pipeline.
- **CNN-Based Shape Recognition:** Uses a trained deep-learning model to classify extracted drawings.
- **Prediction Confidence:** Provides a confidence score alongside the predicted class.
- **Desktop and Browser Versions:** Includes the original Python desktop application and a browser-based drawing experience.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core desktop application logic |
| OpenCV | Webcam input and image processing |
| MediaPipe | Hand detection and landmark tracking |
| TensorFlow / Keras | CNN model training and inference |
| JavaScript | Browser-based drawing interactions |
| HTML5 Canvas | Rendering digital strokes |
| HTML and CSS | Browser interface |
| GitHub Pages | Hosting the browser version |
| GitHub Actions | Automated deployment |

## 🌐 Try the Browser Demo

**[Launch AirCanvas AI](https://stacklume.github.io/AirCanvas-AI/)**

1. Open the live demo in a supported browser.
2. Allow webcam access.
3. Position your hand within the camera view.
4. Use your index finger to draw on the canvas.
5. Experiment with the available drawing controls.

The current browser demo focuses on gesture-based drawing. The complete CNN shape-classification pipeline described above belongs to the main desktop application and should not be assumed to run in the browser version yet.

## 💻 Run the Browser Version Locally

### Prerequisites

- Python installed
- A webcam
- A modern browser
- Internet access for the MediaPipe JavaScript library

### Installation

Clone the repository:

```bash
git clone https://github.com/Stacklume/AirCanvas-AI.git
cd AirCanvas-AI
```

Start a local HTTP server:

```bash
python -m http.server 8000 --directory web
```

Open:

```text
http://localhost:8000
```

Allow camera access to start drawing.

## 🔮 Future Enhancements

- Integrate CNN-based shape classification into the browser version.
- Display predicted shape labels and confidence scores in the interface.
- Improve recognition accuracy for imperfect or incomplete drawings.
- Expand the supported set of shapes.
- Add more gesture-based canvas controls.
- Optimize end-to-end recognition latency.

## 👩‍💻 Author

**Safa Syed**

- GitHub: [@Stacklume](https://github.com/Stacklume)
- LinkedIn: [Safa Syed](https://www.linkedin.com/in/safa-syed-a21a17294)

---

⭐ If you find AirCanvas AI interesting, consider starring the repository!

*Combining computer vision, deep learning, and creativity — one gesture at a time.* 💙
