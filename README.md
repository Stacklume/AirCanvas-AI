# 🎨 AirCanvas AI

An AI-powered virtual drawing canvas that allows users to draw in the air using hand gestures and automatically recognize hand-drawn shapes using a Convolutional Neural Network.

## ✨ Features

- 🖐️ Real-time hand tracking
- ✏️ Gesture-controlled air drawing
- 🎨 Multiple brush colors
- 🖌️ Adjustable brush sizes
- 🧽 Eraser
- ↩️ Undo
- ↪️ Redo
- 🗑️ Clear canvas
- 💾 Save artwork
- 🧠 AI-powered shape recognition
- 📊 Prediction confidence scores
- 🔍 Recognition of five shapes:
  - Circle
  - Square
  - Triangle
  - Star
  - Heart
- 🤖 AI recognition directly inside the drawing application

---

## 🧠 How It Works

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
