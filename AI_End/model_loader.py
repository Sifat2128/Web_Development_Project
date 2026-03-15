import torch
import joblib
import os
from torchvision import models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# recreate same architecture used during training
model = models.resnet18(weights=None)

# remove final FC layer
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# load weights
feature_extractor.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "feature_extractor.pth"), map_location="cpu")
)

feature_extractor.eval()

# load scaler
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# load svm
svm_model = joblib.load(os.path.join(BASE_DIR, "svm_model.pkl"))

print("Models loaded successfully")

