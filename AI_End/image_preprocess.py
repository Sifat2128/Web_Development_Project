from PIL import Image
import torchvision.transforms as transforms
from config import IMAGE_SIZE

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def preprocess_image(image_bytes):

    img = Image.open(image_bytes).convert("RGB")
    img = transform(img).unsqueeze(0)

    return img