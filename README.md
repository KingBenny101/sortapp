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

The system supports two modes: **Pre-trained** (with initial dataset) and **Cold Start** (from scratch).

### Pre-trained Mode

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

### Cold Start Mode

If `ALLOW_COLD_START=True` and no pre-labeled dataset exists:

- Start labeling images immediately from `UNLABELED_DIR`
- Model initializes automatically after collecting `MIN_SAMPLES_PER_CLASS` samples per class
- No initial training data required

## How it works

### Pre-trained Mode

1. **Initial Training**: Loads up to `MAX_PER_CLASS` images from `DATASET_DIR/useful` and `DATASET_DIR/useless` to train the model
2. **Classification**: Presents unlabeled images from `UNLABELED_DIR` for active learning
3. **Incremental Updates**: Model learns from each new label without forgetting previous knowledge
4. **Output Storage**: Copies useful images to `OUTPUT_USEFUL_DIR` (useless images copied only if `COPY_USELESS_IMAGES=True`)

### Cold Start Mode

1. **Bootstrap Collection**: Manually label images until `MIN_SAMPLES_PER_CLASS` samples per class are collected
2. **Model Initialization**: System automatically creates and trains the model once minimum samples are reached
3. **Incremental Learning**: Continues with normal incremental learning after initialization
4. **Output Storage**: Same as pre-trained mode

## Configuration

Edit `config.py` to customize:

### Directory Paths

- **`DATASET_DIR`** - Path to pre-labeled training images (with `useful/` and `useless/` subdirectories)
- **`UNLABELED_DIR`** - Path to unlabeled images for classification
- **`DATA_DIR`** - Model and tracking data storage (default: `./data`)
- **`OUTPUT_DIR`** - Classified images output (default: `./output`)
- **`OUTPUT_USEFUL_DIR`** - Useful images destination (`OUTPUT_DIR/useful`)
- **`OUTPUT_USELESS_DIR`** - Useless images destination (`OUTPUT_DIR/useless`)
- **`TEMP_DIR`** - Cached resized images (default: `./temp`)

### Training Settings

- **`MAX_PER_CLASS`** - Maximum training images per class (default: 200)
- **`ALLOW_COLD_START`** - Enable starting without pre-labeled dataset (default: True)
- **`MIN_SAMPLES_PER_CLASS`** - Minimum samples per class for cold start bootstrap (default: 5)

### Performance Settings

- **`BATCH_SIZE`** - Images processed per batch in auto mode (default: 32)
- **`USE_GPU`** - Enable GPU acceleration if available (default: True)
- **`AUTO_MODE_WAIT_TIME`** - Seconds between images in auto mode (default: 1.0)
- **`SKIP_AUTO_MODE_WAIT`** - Skip wait time in auto mode (default: True)

### Output Settings

- **`COPY_USELESS_IMAGES`** - Copy useless images to output (default: False)
- **`DISPLAY_WIDTH`**, **`DISPLAY_HEIGHT`** - Image display dimensions

## How to Run

### Option 1: Cold Start (No Pre-labeled Data)

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure paths in `config.py`**:

   ```python
   UNLABELED_DIR = Path("/path/to/your/images")
   ALLOW_COLD_START = True
   MIN_SAMPLES_PER_CLASS = 5  # Start with at least 5 per class
   ```

3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

4. **Bootstrap the model**:
   - Label at least 5 images as "Useful"
   - Label at least 5 images as "Useless"
   - Model will initialize automatically

### Option 2: Pre-trained Mode (With Dataset)

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your training dataset**:

   ```
   DATASET_DIR/
   ├── useful/     # Add 50-200 useful images
   └── useless/    # Add 50-200 useless images
   ```

3. **Configure paths in `config.py`**:

   ```python
   DATASET_DIR = Path("/path/to/training/dataset")
   UNLABELED_DIR = Path("/path/to/unlabeled/images")
   ```

4. **Run the app**:

   ```bash
   streamlit run app.py
   ```

5. **Access the interface**: Open your browser to the URL shown (usually `http://localhost:8501`)

## How to Use

- Images from `UNLABELED_DIR` are presented with model predictions
- Click **Useful 👍** or **Useless 👎** to label images
- Use **Skip** to move to next image without labeling
- **Cold Start Mode**: If no model exists, shows bootstrap progress
  - Predictions are 50/50 until model is initialized
  - Auto mode is disabled until model is ready
- **Auto Mode**: Automatically labels based on model predictions (batch processing)
  - Toggle on to enable automatic processing
  - Only **Stop Auto** button remains active during auto mode
  - Processes images in batches for maximum speed
- Labeled images are copied to `OUTPUT_USEFUL_DIR` (and `OUTPUT_USELESS_DIR` if configured)


## Model Management

- Model automatically saves to `DATA_DIR/modal_resnet50_sgd.joblib`
- Processed images tracked in `DATA_DIR/processed_images.csv`
- Model state persists between sessions
- In cold start mode, model is created after minimum samples are collected
