from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

@app.route('/analyze-emotion', methods=['POST'])
def analyze_emotion():
    data = request.json
    img_data = data.get('image')
    if not img_data:
        return jsonify({'error': 'No image provided'}), 400
    try:
        img_bytes = base64.b64decode(img_data.split(',')[1])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Optionally, detect face using haarcascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = img[y:y+h, x:x+w]
        else:
            face_img = img
        result = DeepFace.analyze(face_img, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        # Convert all values to standard Python types
        emotion = result.get('emotion', {})
        emotion = {k: float(v) for k, v in emotion.items()}
        total = sum(emotion.values())
        if total > 0:
            emotion = {k: v / total for k, v in emotion.items()}
        gender = result.get('gender', '')
        if not isinstance(gender, str) and isinstance(gender, dict):
            gender = max(gender, key=gender.get)
        response = {
            'age': int(result.get('age', 0)),
            'gender': str(gender),
            'emotion': emotion
        }
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the full traceback to the terminal
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 