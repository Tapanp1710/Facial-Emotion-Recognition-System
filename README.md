# Facial Emotion Recognition System

A real-time facial emotion recognition web application built with Python, Flask, OpenCV, TensorFlow, and MTCNN. The system detects faces in webcam video streams, classifies facial emotions, and displays live emotion labels on detected faces through a user-friendly web interface.

---

## Features

- **Real-time face detection** using MTCNN for high accuracy and robustness.
- **Emotion classification** into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
- **Web interface** built with Flask serving live video with emotion labels.
- **Start/Stop capture control** for better user interaction.
- **Preprocessing** on grayscale face images with CLAHE for improved model accuracy.

---

## Demo

![Demo Screenshot](screenshot.png)  
*Replace with your application screenshot*

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam connected to your PC

### Installation

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create and activate a virtual environment (recommended):
python -m venv venv

Windows
venv\Scripts\activate


3. Install dependencies:

pip install -r requirements.txt


4. Ensure your pretrained model file `emotion_recognition_model.h5` is in the project root folder.

### Running the Application

Start the Flask app by running:
http://127.0.0.1:5000/

Use the **Start Capture** button to begin the live emotion recognition.

---

## Project Structure

Facial-Emotion-Recognition-System/
│
├── app.py # Main Flask application
├── emotion_recognition_model.h5 # Pretrained Keras model for emotion classification
├── templates/
│ └── index.html # HTML web interface template
├── README.md # Project documentation
└── requirements.txt # Python dependencies list


---

## Dependencies

Key libraries used in this project:

- [Flask](https://flask.palletsprojects.com/) - Web framework
- [OpenCV](https://opencv.org/) - Real-time computer vision tasks
- [TensorFlow](https://tensorflow.org/) - Deep learning framework
- [MTCNN](https://github.com/ipazc/mtcnn) - Multitask Cascaded CNN for face detection
- [Albumentations](https://albumentations.ai/) - Image preprocessing and augmentation
- [NumPy](https://numpy.org/) - Numerical operations

---

## Future Improvements

- Integrate more advanced emotion recognition models.
- Add support for multiple faces and aggregate emotion analysis.
- Provide emotion history and statistics visualization.
- Deploy as a cloud-based web app with user authentication.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspiration and techniques adapted from open-source emotion recognition projects.
- Thanks to contributors of TensorFlow, OpenCV, MTCNN, and Albumentations libraries.

---

## Contact

For questions or suggestions, please contact:  
**Tapan Pendyala** – [pendyala.tapan@gmail.com](mailto:pendyala.tapan@gmail.com)  
1. Clone the repository:

