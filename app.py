import streamlit as st
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

# Define transformations
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Rotate(limit=20, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RandomBrightnessContrast(p=0.2),  # Thêm phép biến đổi độ sáng và độ tương phản
    A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.0),
    ToTensorV2(),
])

# Define model
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=12):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        nn.Dropout(0.2),  # Thêm Dropout
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Dropout(0.2)  # Thêm Dropout
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

# Load model
model = ConvMixer(dim=256, depth=8)
model.load_state_dict(torch.load('model_retrain.pth'), strict=False)
model.eval()

# Load fish class names
with open('fish_classes.txt', 'r') as f:
    fish_classes = [line.strip() for line in f]


# Streamlit app
st.title("ConvMixer Image Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess image
    img_tensor = train_transform(image=np.asarray(image))['image'].unsqueeze(0)

    # If GPU is available, move the tensor to GPU
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
        model = model.cuda()

    # Predict
    with torch.no_grad():
        prob = model(img_tensor)

    # Display prediction
    prob = torch.nn.functional.softmax(prob, dim=1)
    prob = prob.cpu().numpy().flatten()
    class_indices = prob.argsort()[::-1]  # Sort by probability
    for i in range(len(class_indices)):
        st.write(f'{fish_classes[class_indices[i]]}: {prob[class_indices[i]]:.4f} (Class: {class_indices[i]})')

    st.write(f'Predicted: {fish_classes[class_indices[0]]} (Class: {class_indices[0]})')