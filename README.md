# Shoplifting-Detection-Finetuned-MobileNet-and-ConvLstm-Django-Deployment

Welcome to the **Shoplifting Detection System**, a web application designed to detect shoplifting activities from uploaded video footage using a deep learning model. This project leverages a fine-tuned MobileNet architecture with ConvLSTM layers, deployed using Django, HTML, and CSS to provide an interactive interface. 

[I downloaded `ffmpeg` for playing the uploaded video in the browsers as the codec of the videos wasn't supported]  

## Overview

This application allows users to upload video files (e.g., MP4, AVI). The system processes the video, extracts frames, and uses a pre-trained AI model to predict whether shoplifting has occurred, displaying the uploaded video alongside the prediction result (e.g., "Shoplifting Detected" or "No Shoplifting Detected") with a confidence score.

### Key Features
- **Video Upload**: Supports MP4, AVI, MOV, and MKV formats.
- **Real-time Prediction**: Uses a fine-tuned deep learning model to analyze video content.
- **User Interface**: Displays the uploaded video and prediction side-by-side in a dark-themed, responsive design.

## Model Details- Version 3 (1-cnn-convlstmrnn-shopliftdet-pretrained-mobilenet.ipynb)

The detection model is based on a **pre-trained MobileNetV2** backbone, enhanced with **ConvLSTM layers** for temporal modeling of video frames. The model was fine-tuned on a custom dataset of shoplifting and non-shoplifting videos.

### Model Evaluation Metrics
- **Accuracy**: 0.9298
- **Precision**: 0.9297
- **Recall**: 0.9298
- **F1-Score**: 0.9296

---

## Model Details- Version 4 (1-shopliftdet-mobilenet-convlstm-load-memory.ipynb) (best version)

The detection model is based on a **pre-trained MobileNetV2** backbone, enhanced with **ConvLSTM layers** for temporal modeling of video frames. The model was fine-tuned on a custom dataset of shoplifting and non-shoplifting videos.

**I made the same architecture but changed the hyperparameters of the input to:**

* **Frame size: (72, 72)** instead of (64, 64)
* **NUM_FRAMES: 45** instead of 40

**This was possible because of the videos I skipped loading into memory in the next part.**

### Compating Data Leakage

I added a couple of functions to detect similar videos using `cosine similarity` and pair them with `similarity threshold= 1.0` , previewing a couple of the video paths to check them manually if they are identical or not. I then **skipped loading a pair of the similar videos**.

**This ensured no data leakage occurring when splittting the data**.

**I also changed `predict_video_from_dataset()` which is used to predict and show the video, as I forgot to convert to `RGB` in it. When trying it, it lead the model predicting all videos to be non-shoplifters**

<img width="1389" height="490" alt="__results___122_0" src="https://github.com/user-attachments/assets/aeacd016-e24c-4d19-9977-179daba766a1" />

- **Train Accuracy**: 0.8488
- **Validation Accuracy**: 0.9913

### Model Evaluation Metrics on test set
- **Accuracy**: 1.0000
- **Precision**: 1.0000
- **Recall**: 1.0000
- **F1-Score**: 1.0000

These metrics reflect the model's performance on a test set, demonstrating high reliability in distinguishing shoplifting incidents.

---

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

Below is the organized structure of the Shoplifting Detection System project:

| **Path**                          | **Description**                                      |
|-----------------------------------|-----------------------------------------------------|
| `shoplift_detect/`                | Project root directory                              |
| ├── `manage.py`                   | Django management script (run with `python manage.py runserver`) |
| ├── `best_model.keras`            | Fine-tuned Keras model file                         |
| ├── `shoplift_detect/`            | Inner project configuration directory               |
| │   ├── `settings.py`             | Configuration file (e.g., media, static settings)   |
| │   ├── `urls.py`                 | Main URL routing configuration                     |
| │   └── ...                       | Other auto-generated files (e.g., `wsgi.py`)        |
| ├── `detector/`                   | Application module directory                        |
| │   ├── `views.py`                | Logic for video processing and prediction           |
| │   ├── `urls.py`                 | App-specific URL routing                            |
| │   └── ...                       | Other auto-generated files (e.g., `apps.py`)        |
| ├── `templates/`                  | Directory for HTML templates                        |
| │   └── `detector/`               | App-specific template directory                     |
| │       └── `index.html`          | Main HTML template for the user interface           |
| ├── `static/`                     | Directory for static files (CSS, images)            |
| │   └── `detector/`               | App-specific static files directory                 |
| │       ├── `css/`                | CSS subdirectory                                    |
| │       │   └── `style.css`       | CSS file for dark theme and layout                  |
| │       └── `images/`             | Images subdirectory                                 |
| │           ├── `Gemini_Generated_Image1.png` | Header background image       |
| │           └── `image2.jpg`      | "How to Use" section illustration                   |
| └── `media/`                      | Directory for uploaded video files (auto-created)   |

### Notes
- Ensure `best_model.keras`, `Gemini_Generated_Image1.png`, and `image2.jpg` are placed in their respective locations before running the project.
- The `media/` folder is dynamically created when videos are uploaded.


## How it works

* Video Processing: The `views.py` file extracts 40 frames from the uploaded video, resizes them to 64x64 pixels, normalizes the pixel values, and prepares them as input for the model.

* Model Prediction: The `best_model.keras` model processes the frame sequence and outputs a probability score, which is converted into a class label (0 = No Shoplifting, 1 = Shoplifting) with a confidence percentage.
  
* Display: The `index.html` template renders the video using a `<video>` tag and shows the prediction text, styled with style.css.

---

## Project Sample

<img width="1366" height="665" alt="Screenshot (1671)" src="https://github.com/user-attachments/assets/30386054-c37d-4609-a9b1-349232c283cb" />

<img width="1366" height="669" alt="Screenshot (1673)" src="https://github.com/user-attachments/assets/05fc5d11-6d22-42a9-a41d-0091081b68d6" />

<img width="1366" height="697" alt="Screenshot (1675)" src="https://github.com/user-attachments/assets/0824c2d5-dd48-4269-be56-3e2cb05a3f50" />
