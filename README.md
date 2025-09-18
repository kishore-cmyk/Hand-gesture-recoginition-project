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
hand-gesture-recognition/
├── data/ # Directory for training data (CSV files)
├── models/ # Directory for saved models
│ └── gesture_model.pkl # Pre-trained model
├── recognize_gesture.py # Main script for real-time recognition
├── collect_data.py # Script to collect training data for new gestures
├── train_model.py # Script to train a new model on collected data
├── requirements.txt # List of dependencies
└── README.md


## 🛠️ Training on Custom Gestures

Want to teach the system new gestures? It's easy!

1.  **Collect Data:**
    ```bash
    python collect_data.py --label thumbs_up --samples 300
    ```
    This will save 300 data samples for the "thumbs_up" gesture. Repeat for other gestures.

2.  **Train the Model:**
    ```bash
    python train_model.py
    ```
    This will train a new model on all the data in the `data/` directory and save it as `models/your_new_model.pkl`.

3.  **Update the Recognition Script:** Point the `recognize_gesture.py` script to load your new model.

## 🔮 Future Enhancements

- **Dynamic Gestures:** Recognize sequences of movements (e.g., swipes, circles).
- **Volume Control:** Implement a volume control gesture using the distance between thumb and index finger.
- **Background Suppression:** Improve detection in cluttered backgrounds.
- **Deep Learning Model:** Replace the classic ML model with a simple neural network for potentially higher accuracy.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for any improvements.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgments

- [Google MediaPipe](https://mediapipe.dev/) for the excellent hand landmark model.
- The OpenCV community for continuous development and support.

