from flask import Flask, render_template, Response
import cv2
import numpy as np
import albumentations as A
from mtcnn import MTCNN
import tensorflow as tf

app = Flask(__name__)

# Load your pre-trained Keras model that expects grayscale 48x48 images
model = tf.keras.models.load_model('emotion_recognition_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

detector = MTCNN()

# Albumentations transform for preprocessing (can be used on grayscale too)
transform = A.Compose([
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
])

def gen_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb_frame)

        for face in detections:
            x, y, w, h = face['box']
            x, y = max(0,x), max(0,y)
            face_img = rgb_frame[y:y+h, x:x+w]

            if face_img.size == 0:
                continue

            # Resize to 48x48
            face_img = cv2.resize(face_img, (48, 48))

            # Convert to grayscale as model expects single channel
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)

            # Apply CLAHE preprocessing on grayscale image
            face_gray = transform(image=face_gray)['image']

            # Normalize pixel values
            face_gray = face_gray.astype('float32') / 255.0

            # Add channel dimension for grayscale
            face_gray = np.expand_dims(face_gray, axis=-1)

            # Add batch dimension
            face_input = np.expand_dims(face_gray, axis=0)

            # Predict emotion
            preds = model.predict(face_input)[0]
            label = emotion_labels[np.argmax(preds)]

            # Draw rectangle and label on original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
