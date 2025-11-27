# WhatsApp Incremental Learning Image Classifier

> **Note**: This entire project is vibecoded - built through iterative development and experimentation.

A useful/useless image classifier with incremental learning capabilities, designed for WhatsApp image datasets. The system uses active learning to continuously improve its performance by learning from user feedback without catastrophic forgetting.

## Features

- **Incremental Learning**: Updates model without losing previous knowledge
- **Active Learning**: Uses uncertainty sampling to select most informative samples
- **ResNet50 Feature Extraction**: Pre-trained CNN for robust image features
- **Interactive Web Interface**: Streamlit-based UI for easy labeling
- **Persistent Storage**: Saves model state and labeling history
- **Modular Architecture**: Clean separation of concerns across multiple files

## Architecture

- **Feature Extraction**: ResNet50 (ImageNet pretrained) → 2048-dim vectors
- **Classification**: SGDClassifier with incremental learning support
- **Active Learning**: modAL framework with uncertainty sampling
- **Interface**: Streamlit web application

## Libraries Used

### Core ML & Deep Learning

- `torch` - PyTorch deep learning framework
- `torchvision` - ResNet50 model and image transformations
- `scikit-learn` - SGDClassifier, StandardScaler, preprocessing
- `modAL-python` - Active learning framework

### Data Processing

- `numpy` - Numerical computations
- `pandas` - Data manipulation and CSV handling
- `Pillow (PIL)` - Image processing and loading

### Web Interface

- `streamlit` - Interactive web application framework

### Utilities

- `joblib` - Model serialization and persistence
- `pathlib` - Modern path handling

## Installation & Setup

### 1. Create Environment

```bash
conda create --name whatsapp-classifier python=3.13
conda activate whatsapp-classifier
```

### 2. Install Dependencies

```bash
pip install torch torchvision scikit-learn modAL-python streamlit pillow pandas joblib numpy
```

### 3. Prepare Dataset Structure

Create the following directory structure:

```
whatsapp-dataset/
├── images/          # Unlabeled images to classify
├── sorted/          # Labeled images
│   ├── useful/      # Images labeled as useful
│   └── useless/     # Images labeled as useless
└── models/          # Saved model artifacts
```

## How to Run

### 1. Start the Application

```bash
streamlit run app.py
```

### 2. Access the Interface

- Open your browser to `http://localhost:8501`
- The application will automatically load or create a new model

### 3. Label Images

- Images from the `images/` directory will be presented for labeling
- Model predictions are shown before labeling
- Click "Useful 👍" or "Useless 👎" to label images
- Use "Skip" to move to next image without labeling
- Labeled images are automatically copied to `labeled_output/`

### 4. Model Management

- Model automatically saves after each label (configurable)
- Model state persists between sessions
- Labeling history saved in `labels_log.csv`

## Configuration

Update paths in `config.py` to match your dataset location:

```python
BASE_DATASET_DIR = Path("/home/kingbenny101/code/whatsapp-dataset")
SORTED_DIR = BASE_DATASET_DIR / "sorted"
UNLABELED_DIR = BASE_DATASET_DIR / "images"
```

## Incremental Learning Details

- Uses SGDClassifier with `partial_fit()` for online learning
- Implements active learning via uncertainty sampling
- Maintains feature standardization across incremental updates
- Prevents catastrophic forgetting through careful model architecture

## File Structure

```
whatsapp-incremental-learning/
├── app.py                    # Main Streamlit application entry point
├── config.py                 # Configuration and path settings
├── feature_extraction.py     # ResNet50 feature extraction utilities
├── data_handler.py          # Data loading and persistence functions
├── active_learning.py       # Active learning model management
├── logging_utils.py         # Label logging and file operations
├── ui_components.py         # Streamlit UI page components
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore patterns
├── labeled_output/         # Copied labeled images
│   ├── useful/             # Images labeled as useful
│   └── useless/            # Images labeled as useless
├── models/                 # Saved model files
├── labels_log.csv         # Labeling history log
└── processed_images.csv   # Processed images tracking
```

## Module Descriptions

- **`app.py`**: Main entry point with Streamlit page navigation
- **`config.py`**: Centralized configuration for paths and settings
- **`feature_extraction.py`**: PyTorch ResNet50 feature extraction
- **`data_handler.py`**: Dataset loading, image listing, and CSV persistence
- **`active_learning.py`**: modAL learner initialization and model management
- **`logging_utils.py`**: Label logging and file copying utilities
- **`ui_components.py`**: Streamlit UI pages and interaction logic

## Contributing

This project is vibecoded - feel free to experiment, modify, and improve the implementation based on your specific use case and dataset characteristics.
