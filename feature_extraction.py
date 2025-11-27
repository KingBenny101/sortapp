"""Image feature extraction using ResNet50.

This module provides functionality to extract deep learning features from images
using a pre-trained ResNet50 model from PyTorch/Torchvision.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import streamlit as st


@st.cache_resource(show_spinner=True)
def get_resnet_feature_extractor():
    """Load pre-trained ResNet50 model as a frozen feature extractor.

    Returns:
        tuple: A 3-tuple containing:
            - feature_extractor (nn.Sequential): Frozen ResNet50 model without final classification layer
            - preprocess (transforms.Compose): Image preprocessing transforms for ResNet50
            - device (torch.device): Device to run inference on (CPU)

    Note:
        This function is cached by Streamlit to avoid reloading the model on every rerun.
    """
    device = torch.device("cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    modules = list(model.children())[:-1]
    feature_extractor = nn.Sequential(*modules).to(device)
    feature_extractor.eval()

    weights = models.ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    return feature_extractor, preprocess, device


def image_to_feature_vector_torch(
    img: Image.Image, feature_extractor, preprocess, device
) -> np.ndarray:
    """Convert a PIL image to a ResNet50 feature vector.

    Args:
        img (Image.Image): Input PIL image to extract features from
        feature_extractor (nn.Sequential): Pre-trained ResNet50 feature extractor
        preprocess (transforms.Compose): Image preprocessing transforms
        device (torch.device): Device to run inference on

    Returns:
        np.ndarray: 2048-dimensional feature vector representing the image

    Note:
        The image is automatically converted to RGB format before processing.
    """
    img = img.convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(tensor)
    feat = feat.view(-1).cpu().numpy()
    return feat


def load_and_featurize_image(path, feature_extractor, preprocess, device) -> np.ndarray:
    """Load an image from disk and convert it to a feature vector.

    Args:
        path (str): Path to the image file
        feature_extractor (nn.Sequential): Pre-trained ResNet50 feature extractor
        preprocess (transforms.Compose): Image preprocessing transforms
        device (torch.device): Device to run inference on

    Returns:
        np.ndarray: 2048-dimensional feature vector representing the image
    """
    img = Image.open(path)
    return image_to_feature_vector_torch(img, feature_extractor, preprocess, device)
