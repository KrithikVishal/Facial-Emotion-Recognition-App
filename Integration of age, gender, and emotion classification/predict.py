# predict.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import os
from io import BytesIO
import time
import subprocess # Required for executing wget from Python script

# Import constants and preprocessing from utils.py
from utils import preprocess_image_for_vgg16, EMOTION_LABELS, GENDER_LABELS, AGE_LABELS, IMG_WIDTH, IMG_HEIGHT

# Define your project's main path where model and cascade are located
MAIN_PROJECT_PATH = '/content/drive/MyDrive/Project/Face Emotion Recoginization /facial_analysis_app'

# --- 1. Load the Trained Model ---
MODEL_PATH = os.path.join(MAIN_PROJECT_PATH, 'final_multi_task_vgg16_model.h5')

model = None
if not os.path.exists(MODEL_PATH):
    print("Error: Model not found at {}. Please run model.py to train and save it first.".format(MODEL_PATH))
else:
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully from {}".format(MODEL_PATH))
    except Exception as e:
        print("Error loading model: {}".format(e))

# --- 2. Initialize Face Detector (Haar Cascade for simplicity) ---
face_cascade_path = os.path.join(MAIN_PROJECT_PATH, 'haarcascade_frontalface_default.xml')

if not os.path.exists(face_cascade_path):
    print("Warning: Haar Cascade XML not found at {}. Attempting to download it...".format(face_cascade_path))
    try:
        # Use subprocess.run to execute the wget command
        subprocess.run([
            "wget",
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascades/haarcascade_frontalface_default.xml",
            "-O",
            face_cascade_path
        ], check=True, capture_output=True)
        print("Haar Cascade downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading Haar Cascade: {e.stderr.decode()}")
        print("Failed to download Haar Cascade. Real-time prediction may not work.")
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        print("Failed to download Haar Cascade. Real-time prediction may not work.")

face_cascade = cv2.CascadeClassifier(face_cascade_path)

# --- Colab-specific Webcam Access Functions ---
def video_stream():
  js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;

    var pendingResolve = null;
    var shutdown = false;

    async function start() {
      div = document.createElement('div');
      document.body.appendChild(div);

      labelElement = document.createElement('div');
      div.appendChild(labelElement);

      video = document.createElement('video');
      video.style.display = 'block';
      div.appendChild(video);

      stream = await navigator.mediaDevices.getUserMedia({video: true});
      video.srcObject = stream;
      await video.play();

      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      imgElement = document.createElement('img');
      div.appendChild(imgElement);

      google.colab.output.registerOutputForDisposal(async () => {
        shutdown = true;
        stream.getVideoTracks()[0].stop();
        video.remove();
        div.remove();
        video = null;
        div = null;
        stream = null;
        captureCanvas = null;
        imgElement = null;
        labelElement = null;
      });
    }

    function stop() {
      if (stream) {
        stream.getVideoTracks()[0].stop();
        video.remove();
        if (div) {
          div.remove();
        }
        video = null;
        div = null;
        stream = null;
        captureCanvas = null;
        imgElement = null;
        labelElement = null;
      }
      shutdown = true;
    }

    start();
  ''')
  display(js)

def video_frame(quality=0.8):
  js_code = 'google.colab.video.capture(0)'
  data = eval_js(js_code)
  return data.split(',')[1] if data else None

# --- Prediction Loop (Adapted for Colab Webcam) ---
def run_prediction_loop():
    if model is None:
        print("Model not available. Skipping prediction loop.")
        return

    print("Starting real-time analysis. Grant webcam permissions when prompted.")
    video_stream()
    time.sleep(2) # Give webcam time to initialize

    try:
        while True:
            js_reply = video_frame()
            if js_reply is None:
                print("No frame received from webcam. Stopping stream.")
                break

            image_bytes = b64decode(js_reply)
            image_stream = BytesIO(image_bytes)
            jpg_as_np = np.frombuffer(image_stream.read(), dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, flags=1)

            if frame is None:
                print("Failed to decode frame, skipping...")
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                processed_face = preprocess_image_for_vgg16(face_roi)
                processed_face = np.expand_dims(processed_face, axis=0)

                try:
                    emotion_pred, gender_pred, age_pred = model.predict(processed_face, verbose=0)
                    emotion_label = EMOTION_LABELS[np.argmax(emotion_pred[0])]
                    gender_label = GENDER_LABELS[np.argmax(gender_pred[0])]
                    age_label = AGE_LABELS[np.argmax(age_pred[0])]

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    text_color_emotion = (0, 255, 0)
                    text_color_gender = (0, 255, 255)
                    text_color_age = (0, 165, 255)
                    text_y_start = y - 10
                    if text_y_start < 0:
                        text_y_start = y + h + 20

                    cv2.putText(frame, "Emotion: {}".format(emotion_label), (x, text_y_start),
                                font, font_scale, text_color_emotion, font_thickness, cv2.LINE_AA)
                    cv2.putText(frame, "Gender: {}".format(gender_label), (x, text_y_start + 25),
                                font, font_scale, text_color_gender, font_thickness, cv2.LINE_AA)
                    cv2.putText(frame, "Age: {}".format(age_label), (x, text_y_start + 50),
                                font, font_scale, text_color_age, font_thickness, cv2.LINE_AA)

                except Exception as e:
                    print("Error during prediction or drawing for a face: {}".format(e))
                    cv2.putText(frame, "Prediction Error", (x, y - 10),
                                font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

            _, jpeg_frame = cv2.imencode('.jpg', frame)
            jpeg_bytes = jpeg_frame.tobytes()
            b64_frame = b64encode(jpeg_bytes).decode('utf-8')
            js_code = "imgElement.src = 'data:image/jpeg;base64,{0}';".format(b64_frame)
            eval_js(js_code)

    except KeyboardInterrupt:
        print("Stream stopped by user (Ctrl+C detected in Python).")
    except Exception as e:
        print("An unexpected error occurred in prediction loop: {}".format(e))
    finally:
        eval_js('stop()')
        print("Application closed.")

if __name__ == '__main__':
    run_prediction_loop()