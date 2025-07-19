# Facial Emotion Recognition App

This project combines a Python-based backend for facial recognition and emotion analysis with a modern React-based frontend for user interaction and visualization.

## Project Structure

```
facial-emotion-recognition-app/
  backend/    # Python backend (DeepFace, OpenCV)
  frontend/   # React frontend (Vite, TypeScript, Tailwind)
```

## Backend (Python)
- Located in the `backend/` folder
- Provides face recognition, age, gender, and emotion analysis using DeepFace and OpenCV
- To run (from the `backend` folder):
  ```bash
  python "Face Recog.py"
  ```
- (Recommended) Convert to a REST API for frontend integration (Flask/FastAPI)

## Frontend (React)
- Located in the `frontend/` folder
- Modern UI for capturing/uploading images and displaying emotion analysis
- To run (from the `frontend` folder):
  ```bash
  npm install
  npm run dev
  ```

## Integration
- The backend and frontend are currently separate.
- To connect them, expose backend functionality as an API and update the frontend to call it.

## Setup
1. Install Python dependencies in `backend/` (see its README for details)
2. Install Node.js dependencies in `frontend/`
3. Run both backend and frontend as described above

## License
See individual `LICENSE` files in each folder for details. 