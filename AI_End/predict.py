import torch
import numpy as np

from image_preprocess import preprocess_image
from model_loader import feature_extractor, scaler, svm_model
from config import CLASS_NAMES


def predict(image_bytes):

    # preprocess
    img = preprocess_image(image_bytes)

    # feature extraction
    with torch.no_grad():
        features = feature_extractor(img)

    features = features.numpy().reshape(1, -1)  # ✅ fixed

    # scale features
    features = scaler.transform(features)

    # SVM prediction
    prediction = svm_model.predict(features)[0]

    label = CLASS_NAMES[prediction]

    return label