# Face Recognition System

This project implements a real-time face recognition system using OpenCV and Python. It uses the Local Binary Patterns Histograms (LBPH) algorithm for face recognition.

## Project Structure

```
FaceRecognition/
├── dataset/                # Directory where face samples are stored
├── src/                    # Source code directory
│   ├── Dataset_generator.py # Script to capture face images
│   ├── trainer.py          # Script to train the recognition model
│   └── facedetector.py     # Script to recognize faces in real-time
├── haarcascade_frontalface_default.xml # Haar cascade file (required)
├── test_trainingdata.yml   # Trained model file (generated after training)
└── README.md               # Project documentation
```

## Prerequisites

Ensure you have Python installed along with the following libraries:

*   OpenCV (`opencv-python` and `opencv-contrib-python`)
*   NumPy (`numpy`)
*   Pillow (`Pillow`)

You can install the dependencies using pip:

```bash
pip install opencv-python opencv-contrib-python numpy Pillow
```

**Note:** This project requires the `haarcascade_frontalface_default.xml` file to be present in the project root directory. You can download it from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

## Usage

The system works in three steps: Data Collection, Training, and Recognition.

### 1. Data Collection

Run the `Dataset_generator.py` script to capture face images for a new user.

```bash
python src/Dataset_generator.py
```

*   You will be prompted to enter a **User ID** (integer).
*   The script will open your webcam and capture 500 sample images of your face.
*   The images will be saved in the `dataset/` folder.

### 2. Training

After collecting data, run the `trainer.py` script to train the face recognition model.

```bash
python src/trainer.py
```

*   This script reads the images from the `dataset/` folder.
*   It trains the LBPH recognizer.
*   The trained model is saved as `test_trainingdata.yml` in the project root.

### 3. Face Recognition

Finally, run the `facedetector.py` script to recognize faces in real-time.

```bash
python src/facedetector.py
```

*   The script opens the webcam and detects faces.
*   It attempts to recognize the face based on the trained model.
*   Press `q` to quit the application.

## Configuration

*   **User Names:** You can map User IDs to names in `src/facedetector.py` by modifying the `if-elif` block inside the loop.

```python
if id == 1:
    id = "Your Name"
```

## Troubleshooting

*   **AttributeError: module 'cv2' has no attribute 'createLBPHFaceRecognizer'**:
    *   This error occurs if you have a newer version of OpenCV. Try installing `opencv-contrib-python` and use `cv2.face.LBPHFaceRecognizer_create()` instead of `cv2.createLBPHFaceRecognizer()`.
    *   *Note: The current code uses the older syntax `cv2.createLBPHFaceRecognizer()`. If you face issues, you might need to update the code to `cv2.face.LBPHFaceRecognizer_create()`.
