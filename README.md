# Hand-gesture-recoginition-project

A real-time hand gesture recognition system using OpenCV and MediaPipe. This project processes webcam feed to detect hand landmarks and uses a machine learning model to classify static gestures, enabling touchless control.

Topics: computer-vision, opencv, mediapipe, machine-learning, python, gesture-control

# Real-Time Hand Gesture Recognition 👋

A Python application that recognizes static hand gestures in real-time using a webcam. It leverages Google's MediaPipe for robust hand landmark detection and a scikit-learn model for classification.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.3%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## ✨ Features

- **Real-time Processing:** Processes webcam feed at high FPS for smooth interaction.
- **Accurate Landmark Detection:** Uses MediaPipe Hands to find 21 precise hand keypoints.
- **Machine Learning Pipeline:** Classifies gestures using a pre-trained model (included).
- **Customizable:** Easy to collect new data and train the model on new gestures.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- A webcam

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/hand-gesture-recognition.git
    cd hand-gesture-recognition
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv gesture-env
    # On Windows
    gesture-env\Scripts\activate
    # On macOS/Linux
    source gesture-env/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Run the recognition script:**
    ```bash
    python recognize_gesture.py
    ```

2.  **Perform gestures in view of your webcam.** The recognized gesture name will be displayed in the top-left corner.

3.  **Press `q`** to quit the application.

## 🧠 How It Works

1.  **Frame Capture:** OpenCV captures frames from the webcam.
2.  **Hand Landmark Detection:** Each frame is processed by MediaPipe Hands to find and extract 21 (x, y) coordinates for each hand joint.
3.  **Data Preparation:** The coordinates are normalized and flattened into a feature vector.
4.  **Classification:** The pre-trained machine learning model (e.g., Random Forest) takes the feature vector and predicts the gesture class.
5.  **Visualization:** The hand landmarks, bounding box, and predicted gesture label are drawn on the frame and displayed.

## 📁 Project Structure
