
import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
const video = document.getElementById("webcam");

const colorPicker = document.getElementById("colorPicker");
const brushSize = document.getElementById("brushSize");
const brushValue = document.getElementById("brushValue");
const emptyState = document.getElementById("emptyState");
const status = document.getElementById("status");
const cameraBtn = document.getElementById("cameraBtn");
const cameraMessage = document.getElementById("cameraMessage");
const errorMessage = document.getElementById("errorMessage");

let handLandmarker = null;
let cameraStream = null;
let cameraRunning = false;
let animationId = null;
let lastVideoTime = -1;

let drawing = false;
let erasing = false;
let lastPoint = null;
let strokes = [];
let currentStroke = null;

ctx.lineCap = "round";
ctx.lineJoin = "round";

function canvasPoint(x, y) {
  const rect = canvas.getBoundingClientRect();

  return {
    x: x * canvas.width,
    y: y * canvas.height
  };
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const stroke of strokes) {
    if (!stroke.points.length) continue;

    ctx.beginPath();
    ctx.strokeStyle = stroke.color;
    ctx.lineWidth = stroke.size;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.moveTo(stroke.points[0].x, stroke.points[0].y);

    for (const point of stroke.points.slice(1)) {
      ctx.lineTo(point.x, point.y);
    }

    if (stroke.points.length === 1) {
      ctx.lineTo(
        stroke.points[0].x + 0.01,
        stroke.points[0].y
      );
    }

    ctx.stroke();
  }

  emptyState.classList.toggle("hidden", strokes.length > 0);
}

function startStroke(point) {
  currentStroke = {
    color: erasing ? "#080b14" : colorPicker.value,
    size: erasing
      ? Number(brushSize.value) * 3
      : Number(brushSize.value),
    points: [point]
  };

  strokes.push(currentStroke);
  lastPoint = point;
}

function addPoint(point) {
  if (!currentStroke) {
    startStroke(point);
    return;
  }

  currentStroke.points.push(point);
  lastPoint = point;
}

function stopDrawing() {
  drawing = false;
  currentStroke = null;
  lastPoint = null;
}

function fingerExtended(landmarks, tipIndex, pipIndex) {
  return landmarks[tipIndex].y < landmarks[pipIndex].y;
}

function processHand(landmarks) {
  // MediaPipe landmark indexes:
  // Index finger tip = 8; index finger PIP joint = 6.
  // Middle finger tip = 12; middle finger PIP joint = 10.
  const indexUp = fingerExtended(landmarks, 8, 6);
  const middleUp = fingerExtended(landmarks, 12, 10);

  // Draw only when the index finger is raised and
  // the middle finger is lowered.
  const shouldDraw = indexUp && !middleUp;

  if (!shouldDraw) {
    stopDrawing();
    cameraMessage.textContent =
      "Hand detected. Raise only your index finger to draw.";
    return;
  }

  // Mirror the horizontal coordinate for a natural selfie
  // experience. MediaPipe coordinates range from 0 to 1.
  const tip = landmarks[8];
  const point = canvasPoint(1 - tip.x, tip.y);

  if (!drawing) {
    drawing = true;
    startStroke(point);
  } else {
    addPoint(point);
  }

  cameraMessage.textContent = "Drawing — move your index finger!";
  redraw();
}

function renderLoop() {
  if (!cameraRunning || !handLandmarker) return;

  if (
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    video.currentTime !== lastVideoTime
  ) {
    lastVideoTime = video.currentTime;

    try {
      const result = handLandmarker.detectForVideo(
        video,
        performance.now()
      );

      if (result.landmarks && result.landmarks.length > 0) {
        processHand(result.landmarks[0]);
        status.textContent = drawing ? "Drawing" : "Hand detected";
      } else {
        stopDrawing();
        status.textContent = "Looking for hand";
        cameraMessage.textContent =
          "Show your hand clearly in front of the camera.";
      }
    } catch (error) {
      console.error("Hand detection error:", error);
      stopDrawing();
      status.textContent = "Tracking error";
    }
  }

  animationId = requestAnimationFrame(renderLoop);
}

async function startCamera() {
  errorMessage.textContent = "";
  cameraBtn.disabled = true;
  cameraBtn.textContent = "Loading AI…";

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "Camera access is unavailable. Open the website using localhost or HTTPS."
      );
    }

    cameraMessage.textContent = "Loading hand-tracking model…";

    if (!handLandmarker) {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
      );

      handLandmarker = await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.6,
          minHandPresenceConfidence: 0.6,
          minTrackingConfidence: 0.6
        }
      );
    }

    cameraMessage.textContent = "Requesting webcam permission…";

    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 }
      }
    });

    video.srcObject = cameraStream;
    await video.play();

    cameraRunning = true;
    lastVideoTime = -1;

    cameraBtn.textContent = "Stop camera";
    cameraBtn.disabled = false;
    status.textContent = "Camera on";
    cameraMessage.textContent =
      "Camera ready. Raise only your index finger to draw.";

    animationId = requestAnimationFrame(renderLoop);
  } catch (error) {
    console.error("Camera startup failed:", error);

    stopCamera();

    let message = error.message || "Unable to start the camera.";

    if (error.name === "NotAllowedError") {
      message =
        "Camera permission was denied. Allow camera access for localhost in Brave settings, then try again.";
    } else if (error.name === "NotFoundError") {
      message = "No webcam was found on this device.";
    } else if (error.name === "NotReadableError") {
      message =
        "The webcam may be in use by another application. Close other camera apps and retry.";
    } else if (
      error.name === "NotSupportedError" ||
      error.name === "SecurityError"
    ) {
      message = "Camera access requires localhost or a secure HTTPS website.";
    }

    errorMessage.textContent = message;
    cameraMessage.textContent = "Camera could not start.";
    status.textContent = "Camera off";
    cameraBtn.textContent = "Try camera again";
    cameraBtn.disabled = false;
  }
}

function stopCamera() {
  cameraRunning = false;
  stopDrawing();

  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  video.srcObject = null;
  cameraBtn.textContent = "Start camera ↗";
  cameraBtn.disabled = false;
  status.textContent = "Camera off";
  cameraMessage.textContent = "Allow camera access to begin.";
}

cameraBtn.addEventListener("click", () => {
  if (cameraRunning) {
    stopCamera();
  } else {
    startCamera();
  }
});

// Brush controls
brushSize.addEventListener("input", () => {
  brushValue.textContent = `${brushSize.value} px`;
});

// Eraser
document.getElementById("eraseBtn").addEventListener("click", (event) => {
  erasing = !erasing;
  event.currentTarget.classList.toggle("active", erasing);
  event.currentTarget.textContent = erasing ? "✎ Draw mode" : "⌫ Eraser";
});

// Undo
document.getElementById("undoBtn").addEventListener("click", () => {
  strokes.pop();
  stopDrawing();
  redraw();
});

// Clear
document.getElementById("clearBtn").addEventListener("click", () => {
  strokes = [];
  stopDrawing();
  redraw();
});

// Save artwork
document.getElementById("saveBtn").addEventListener("click", () => {
  const link = document.createElement("a");
  link.download = "aircanvas-art.png";
  link.href = canvas.toDataURL("image/png");
  link.click();
});

// Keep mouse drawing available as a fallback.
canvas.addEventListener("pointerdown", (event) => {
  if (cameraRunning) return;

  canvas.setPointerCapture(event.pointerId);
  drawing = true;
  startStroke(canvasPoint(
    (event.clientX - canvas.getBoundingClientRect().left) /
      canvas.getBoundingClientRect().width,
    (event.clientY - canvas.getBoundingClientRect().top) /
      canvas.getBoundingClientRect().height
  ));
  redraw();
});

canvas.addEventListener("pointermove", (event) => {
  if (!drawing || cameraRunning) return;

  const rect = canvas.getBoundingClientRect();
  addPoint(canvasPoint(
    (event.clientX - rect.left) / rect.width,
    (event.clientY - rect.top) / rect.height
  ));
  redraw();
});

canvas.addEventListener("pointerup", () => {
  if (!cameraRunning) stopDrawing();
});

canvas.addEventListener("pointercancel", () => {
  if (!cameraRunning) stopDrawing();
});

window.addEventListener("beforeunload", () => {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
  }
});

redraw();
