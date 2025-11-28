# Sort App

> **Note**: This entire project is vibecoded.

A binary image classifier using incremental learning and active learning to sort images into **useful** and **useless** categories.

## Quick Start

All configuration is done through `config.yaml` in the project root. See [Configuration](#configuration) section for details.

## Core Libraries

- **PyTorch/Torchvision** - ResNet50 feature extraction
- **scikit-learn** - SGDClassifier for incremental learning
- **modAL** - Active learning framework
- **Streamlit** - Web interface
- **Pillow** - Image processing

## Dataset Structure

The system supports two modes: **Pre-trained** (with initial dataset) and **Cold Start** (from scratch).

### Pre-trained Mode

**Training Dataset** (`dataset_dir`):

```
dataset_dir/
├── useful/                   # Pre-labeled useful images
└── useless/                  # Pre-labeled useless images
```

**Working Directory** (`unlabeled_dir`):

```
unlabeled_dir/               # Unlabeled images to classify
├── image1.jpg
├── image2.png
└── ...
```

### Cold Start Mode

If `allow_cold_start=true` and no pre-labeled dataset is provided:

- Start labeling images immediately from `unlabeled_dir`
- Model initializes automatically after collecting `min_samples_per_class` samples per class
- No initial training data required

## How it works

### Pre-trained Mode

1. **Initial Training**: Loads up to `max_per_class` images from `dataset_dir/useful` and `dataset_dir/useless` to train the model
2. **Classification**: Presents unlabeled images from `unlabeled_dir` for active learning
3. **Incremental Updates**: Model learns from each new label without forgetting previous knowledge
4. **Output Storage**: Copies useful images to output directory (useless images copied only if `copy_useless_images=true`)

### Cold Start Mode

1. **Bootstrap Collection**: Manually label images until `min_samples_per_class` samples per class are collected
2. **Model Initialization**: System automatically creates and trains the model once minimum samples are reached
3. **Incremental Learning**: Continues with normal incremental learning after initialization
4. **Output Storage**: Same as pre-trained mode

## How to Run

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/KingBenny101/sortapp.git
   cd sortapp
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda create -n sortapp python=3.9
   conda activate sortapp
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the application**:

   Edit `config.yaml` (in the project root, next to `app.py`):

   ```yaml
   # Required: Path to unlabeled images
   unlabeled_dir: /path/to/your/images

   # Optional: Pre-labeled training dataset
   # If not provided, app runs in cold start mode
   dataset_dir: /path/to/training/dataset

   # Optional: Output directory for classified images
   output_dir: /path/to/output
   ```

5. **Prepare your dataset** (optional):

   For pre-trained mode, organize your training data as follows:

   ```
   DATASET_DIR/
   ├── useful/     # 50-200 useful images
   └── useless/    # 50-200 useless images
   ```

6. **Run the application**:

   ```bash
   streamlit run app.py
   ```

   The app will open in your browser (usually `http://localhost:8501`).

   > **Note**: In cold start mode, label at least 5 images per class to initialize the model.

## How to Use

### Main Interface

- **Image Presentation**: Images from `unlabeled_dir` are displayed with model predictions (in pre-trained mode)
- **Labeling**: Click **Useful 👍** or **Useless 👎** to classify images
- **Skip**: Skip to the next image without labeling
- **Labeled Output**: Classified images are automatically copied to `output_dir`

### Cold Start Mode

If no pre-labeled dataset is provided (`dataset_dir` is empty):

- Start with bootstrap labeling: label at least 5 images per class
- Predictions are random until the model initializes
- Auto mode is disabled until the model is ready
- Once initialized, the model works like pre-trained mode

### Auto Mode

Automatically classify images based on model predictions:

- Toggle **Auto Mode** to enable/disable
- Only **Stop Auto** button is active during processing
- Images are processed in batches for maximum speed
- Useful images are copied to output directory automatically

## Configuration

Edit `config.yaml` to customize settings. All values can be left blank to use defaults from `src/config.py`.

### Required Settings

- **`unlabeled_dir`** - Path to unlabeled images for classification

### Optional Settings

- **`dataset_dir`** - Path to pre-labeled training images (with `useful/` and `useless/` subdirectories)
- **`output_dir`** - Classified images output directory
- **`data_dir`** - Model and tracking data storage (default: `./data`)
- **`temp_dir`** - Cached resized images (default: `./temp`)

### Training Parameters

- **`max_per_class`** - Maximum training images per class (default: 2000)
- **`allow_cold_start`** - Enable starting without pre-labeled dataset (default: True)
- **`min_samples_per_class`** - Minimum samples per class for bootstrap (default: 5)

### Performance Settings

- **`batch_size`** - Images per batch for processing (default: 512)
- **`use_gpu`** - Enable GPU acceleration (default: True)
- **`auto_mode_wait_time`** - Seconds between images in auto mode (default: 1.0)
- **`skip_auto_mode_wait`** - Skip wait time in auto mode (default: True)

### Display & Output

- **`display_width`**, **`display_height`** - Image display dimensions (default: 480x360)
- **`copy_useless_images`** - Copy useless images to output (default: False)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to:

- Report issues or bugs
- Suggest improvements
- Submit pull requests
- Share your use cases

For major changes, please open an issue first to discuss your proposed changes.
