import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import streamlit as st


@st.cache_resource(show_spinner=True)
def get_resnet_feature_extractor():
    """
    Load pretrained ResNet50 (torchvision) as a frozen feature extractor.
    """
    device = torch.device("cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    modules = list(model.children())[:-1]
    feature_extractor = nn.Sequential(*modules).to(device)
    feature_extractor.eval()

    weights = models.ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    return feature_extractor, preprocess, device


def image_to_feature_vector_torch(img: Image.Image, feature_extractor, preprocess, device) -> np.ndarray:
    """Convert PIL image to ResNet50 feature vector using PyTorch."""
    img = img.convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(tensor)
    feat = feat.view(-1).cpu().numpy()
    return feat


def load_and_featurize_image(path, feature_extractor, preprocess, device) -> np.ndarray:
    """Load image from path and convert to feature vector."""
    img = Image.open(path)
    return image_to_feature_vector_torch(img, feature_extractor, preprocess, device)