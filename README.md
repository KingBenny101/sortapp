# Sort App

> **Note**: This entire project is vibecoded - built through iterative development and experimentation.

A binary image classifier using incremental learning and active learning to sort images into **useful** and **useless** categories.

## Core Libraries

- **PyTorch/Torchvision** - ResNet50 feature extraction
- **scikit-learn** - SGDClassifier for incremental learning
- **modAL** - Active learning framework
- **Streamlit** - Web interface
- **Pillow** - Image processing

## Dataset Structure

The system uses two separate directories configured in `config.py`:

**Training Dataset** (`config.DATASET_DIR`):

```
DATASET_DIR/
├── useful/                   # Pre-labeled useful images
└── useless/                  # Pre-labeled useless images
```

**Working Directory** (`config.UNLABELED_DIR`):

```
UNLABELED_DIR/               # Unlabeled images to classify
├── image1.jpg
├── image2.png
└── ...
```

## How it works

1. **Initial Training**: Loads up to `MAX_PER_CLASS` images from `DATASET_DIR/useful` and `DATASET_DIR/useless` to train the model
2. **Classification**: Presents unlabeled images from `UNLABELED_DIR` for active learning
3. **Incremental Updates**: Model learns from each new label without forgetting previous knowledge
4. **Output Storage**: Copies useful images to `OUTPUT_USEFUL_DIR` (useless images copied only if `COPY_USELESS_IMAGES=True`)

## Configuration

Edit `config.py` to customize:

- **`DATASET_DIR`** - Path to pre-labeled training images (with `useful/` and `useless/` subdirectories)
- **`UNLABELED_DIR`** - Path to unlabeled images for classification
- **`DATA_DIR`** - Model and tracking data storage (default: `./data`)
- **`OUTPUT_DIR`** - Classified images output (default: `./output`)
- **`OUTPUT_USEFUL_DIR`** - Useful images destination (`OUTPUT_DIR/useful`)
- **`OUTPUT_USELESS_DIR`** - Useless images destination (`OUTPUT_DIR/useless`)
- **`TEMP_DIR`** - Cached resized images (default: `./temp`)
- **`MAX_PER_CLASS`** - Maximum training images per class (default: 200)
- **`AUTO_MODE_WAIT_TIME`** - Seconds between images in auto mode (default: 1.0)
- **`SKIP_AUTO_MODE_WAIT`** - Skip wait time in auto mode (default: False)
- **`COPY_USELESS_IMAGES`** - Copy useless images to output (default: False)
- **`DISPLAY_WIDTH`**, **`DISPLAY_HEIGHT`** - Image display dimensions

## How to Run

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure paths in `config.py`**:

   - Set `DATASET_DIR` to your pre-labeled training images
   - Set `UNLABELED_DIR` to your unlabeled images

3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

4. **Access the interface**: Open your browser to the URL shown (usually `http://localhost:8501`)

## How to Use

### 1. Label Images Page

- Images from `UNLABELED_DIR` are presented with model predictions
- Click **Useful 👍** or **Useless 👎** to label images
- Use **Skip** to move to next image without labeling
- **Auto Mode**: Automatically labels based on model predictions
  - Toggle on to enable automatic processing
  - Only **Stop Auto** button remains active during auto mode
- Labeled images are copied to `OUTPUT_USEFUL_DIR` (and `OUTPUT_USELESS_DIR` if configured)

### 2. Settings Page

- View and configure all paths
- Check model status and training progress
- Monitor processed images count

### 3. Info Page

- View system information
- Check configuration details
- Review active learning statistics

### Model Management

- Model automatically saves to `DATA_DIR/modal_resnet50_sgd.joblib`
- Processed images tracked in `DATA_DIR/processed_images.csv`
- Model state persists between sessions
