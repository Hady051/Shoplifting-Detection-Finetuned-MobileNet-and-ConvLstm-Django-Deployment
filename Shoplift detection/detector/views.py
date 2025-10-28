from django.http import JsonResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import regularizers

# Rebuild the model architecture
def build_cnn_convlstm_model(input_shape=(40, 64, 64, 3), num_classes=2, trainable_layers=30):
    cnn_base = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(input_shape[1], input_shape[2], input_shape[3])
    )
    for layer in cnn_base.layers[:-trainable_layers]:
        layer.trainable = False
    cnn_base.trainable = True

    cnn_out = models.Sequential([
        cnn_base,
        layers.GlobalAveragePooling2D()
    ], name="cnn_feature_extractor")

    inputs = layers.Input(shape=input_shape)
    x = layers.TimeDistributed(cnn_out)(inputs)
    x = layers.TimeDistributed(layers.Reshape((8, 8, -1)))(x)
    x = layers.ConvLSTM2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False,
        activation='tanh',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_ConvLSTM")
    return model

# Rebuild and load weights
MODEL_PATH = os.path.join(settings.BASE_DIR, 'best_model.keras')
model = build_cnn_convlstm_model()
model.load_weights(MODEL_PATH)

# Preprocessing parameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_FRAMES = 40
CHANNELS = 3

def extract_frames_from_video(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8))

    return np.array(frames[:num_frames])

def normalize_frames(frames):
    return frames.astype(np.float32) / 255.0

def preprocess_video(video_path):
    frames = extract_frames_from_video(video_path)
    if frames is None:
        return None
    frames = normalize_frames(frames)
    return frames

def index(request):
    video_url = None
    prediction = None
    error = None

    if request.method == 'POST' and 'file' in request.FILES:
        file = request.FILES['file']
        if file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            filename = fs.save(file.name, file)
            video_full_path = os.path.join(settings.MEDIA_ROOT, filename)
            video_url = f"{settings.MEDIA_URL}{filename}"

            print("MEDIA_ROOT:", settings.MEDIA_ROOT)
            print("File exists:", os.path.exists(video_full_path))

            ## ✅ Re-encode video using OpenCV + FFmpeg to re-encode it as H.264 .mp4 (avc1 codec).
            try:
                import cv2

                ## Debugging part to see if openCV can open the video or not
                # cap = cv2.VideoCapture(video_full_path)
                # if not cap.isOpened():
                #     print("⚠️ OpenCV could not open:", video_full_path)
                # else:
                #     print("✅ OpenCV successfully opened:", video_full_path)
                # cap.release()

                cap = cv2.VideoCapture(video_full_path)

                if not cap.isOpened():
                    print("⚠️ Could not open uploaded video for verification")
                else:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0 or np.isnan(fps):
                        fps = 25
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Re-encode using H.264 (browser safe)
                    safe_video_path = os.path.join(settings.MEDIA_ROOT, f"browser_{filename}")
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 now available via FFmpeg
                    out = cv2.VideoWriter(safe_video_path, fourcc, fps, (width, height))

                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                        frame_count += 1

                    cap.release()
                    out.release()

                    if frame_count > 0:
                        video_full_path = safe_video_path
                        video_url = settings.MEDIA_URL + os.path.basename(safe_video_path)
                        print(f"✅ Re-encoded {frame_count} frames using H.264.")
                    else:
                        print("⚠️ No frames were written during re-encode.")

            except Exception as e:
                print(f"⚠️ Video re-encode error: {e}")

            try:
                frames = preprocess_video(video_full_path)
                if frames is not None:
                    input_data = np.expand_dims(frames, axis=0)
                    pred_probs = model.predict(input_data)
                    pred_class = np.argmax(pred_probs, axis=1)[0]
                    confidence = pred_probs[0][pred_class] * 100
                    prediction = f"{'Shoplifting Detected' if pred_class == 1 else 'No Shoplifting Detected'} (Confidence: {confidence:.2f}%)"
                else:
                    error = "Invalid or unreadable video file."
            except Exception as e:
                error = f"Error processing video: {str(e)}"
        else:
            error = "Please upload a valid video file (.mp4, .avi, etc.)."

    return render(request, 'detector/index.html', {
        'video_url': video_url,
        'prediction': prediction,
        'error': error
    })
