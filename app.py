from fastapi import FastAPI, UploadFile
import cv2
from deepface import DeepFace
import numpy as np
import io

app = FastAPI(title="Face Analyze API")

# Load Haar Cascade Classifier for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.post("/analyze/")
async def analyze_image(image: UploadFile):
    try:
        # Read the uploaded image
        image_bytes = await image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Check if the image contains a face
        faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            # Return an error if no face is detected
            return {"error": "No face detected in the image"}

        # Detect age, gender, race, and emotion
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)

        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
