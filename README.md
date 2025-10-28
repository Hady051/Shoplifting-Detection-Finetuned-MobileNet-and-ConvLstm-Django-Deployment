# Shoplifting-Detection-Finetuned-MobileNet-and-ConvLstm-Django-Deployment

Welcome to the **Shoplifting Detection System**, a web application designed to detect shoplifting activities from uploaded video footage using a deep learning model. This project leverages a fine-tuned MobileNet architecture with ConvLSTM layers, deployed using Django, HTML, and CSS to provide an interactive interface. 

[I downloaded `ffmpeg` for playing the uploaded video in the browsers as the codec of the videos wasn't supported]  

## Overview

This application allows users to upload video files (e.g., MP4, AVI). The system processes the video, extracts frames, and uses a pre-trained AI model to predict whether shoplifting has occurred, displaying the uploaded video alongside the prediction result (e.g., "Shoplifting Detected" or "No Shoplifting Detected") with a confidence score.

### Key Features
- **Video Upload**: Supports MP4, AVI, MOV, and MKV formats.
- **Real-time Prediction**: Uses a fine-tuned deep learning model to analyze video content.
- **User Interface**: Displays the uploaded video and prediction side-by-side in a dark-themed, responsive design.

## Model Details

The detection model is based on a **pre-trained MobileNetV2** backbone, enhanced with **ConvLSTM layers** for temporal modeling of video frames. The model was fine-tuned on a custom dataset of shoplifting and non-shoplifting videos.

### Model Evaluation Metrics
- **Accuracy**: 0.9298
- **Precision**: 0.9297
- **Recall**: 0.9298
- **F1-Score**: 0.9296

These metrics reflect the model's performance on a test set, demonstrating high reliability in distinguishing shoplifting incidents.

## Technologies Used

- **Backend**: [Django](https://www.djangoproject.com/) - A high-level Python web framework for handling requests, file uploads, and server logic.
- **Machine Learning**: [TensorFlow/Keras](https://www.tensorflow.org/) - Used to load and run the `best_model.keras` file.
- **Video Processing**: [OpenCV](https://opencv.org/) - Extracts and preprocesses video frames (with FFmpeg support).
- **Frontend**:
  - [HTML5](https://developer.mozilla.org/en-US/docs/Web/HTML) - Structures the web page.
  - [CSS3](https://developer.mozilla.org/en-US/docs/Web/CSS) - Styles the interface with a dark theme.
- **Dependencies**: NumPy for numerical operations.

---

## Project Structure

shoplift_detect/
├── manage.py              # Django management script
├── best_model.keras       # Fine-tuned Keras model
├── shoplift_detect/       # Project configuration
│   ├── settings.py        # Configuration (e.g., media, static files)
│   ├── urls.py            # URL routing
│   └── ...
├── detector/              # Application module
│   ├── views.py           # Logic for video processing and prediction
│   ├── urls.py            # App-specific URL routing
│   └── ...
├── templates/
│   └── detector/
│       └── index.html     # HTML template for the user interface
├── static/
│   └── detector/
│       ├── css/
│       │   └── style.css  # CSS for dark theme and layout
│       └── images/
│           ├── Gemini_Generated_Image1.png  # Header background image
│           └── image2.jpg                   # "How to Use" illustration
└── media/                 # Directory for uploaded video files

## How it works

* Video Processing: The `views.py` file extracts 40 frames from the uploaded video, resizes them to 64x64 pixels, normalizes the pixel values, and prepares them as input for the model.

* Model Prediction: The `best_model.keras` model processes the frame sequence and outputs a probability score, which is converted into a class label (0 = No Shoplifting, 1 = Shoplifting) with a confidence percentage.
  
* Display: The `index.html` template renders the video using a `<video>` tag and shows the prediction text, styled with style.css.

---

## Project Sample

<img width="1366" height="768" alt="Screenshot (1671)" src="https://github.com/user-attachments/assets/9c7d610a-df47-4431-b90c-d60aade4a5d6" />

<img width="1366" height="768" alt="Screenshot (1673)" src="https://github.com/user-attachments/assets/c52d2835-8115-4ac6-89fe-382e99232f12" />

<img width="1366" height="697" alt="Screenshot (1675)" src="https://github.com/user-attachments/assets/0824c2d5-dd48-4269-be56-3e2cb05a3f50" />
