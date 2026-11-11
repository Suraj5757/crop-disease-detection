# Crop Disease Detection System using EfficientNetB0

An explainable deep learning system that detects crop leaf diseases from a single photograph. It classifies leaves into **15 classes** across **Pepper, Potato and Tomato** crops, explains every prediction with **Grad-CAM** heatmaps, and ships with an interactive **Streamlit** dashboard for real-time diagnosis.

**Test accuracy: 99.61%** | **Macro F1-score: 0.996** | **Dataset: PlantVillage (20,638 images)**

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model and Training](#model-and-training)
- [Results](#results)
- [Dashboard](#dashboard)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Author](#author)

---

## Features

- Transfer learning with a pretrained **EfficientNetB0** (about 4.03M parameters)
- Two-phase fine-tuning for stable, high-accuracy training
- Handles a 21:1 class imbalance with inverse-frequency class weighting
- Custom, hook-based **Grad-CAM** to show which leaf regions drive each prediction
- 5-page **Streamlit** dashboard: data exploration, model metrics, Grad-CAM gallery and live prediction
- Live prediction returns the disease class, a confidence score, a Grad-CAM overlay and a suggested treatment action
- Runs on **CUDA**, **Apple Silicon (MPS)** or **CPU**, with automatic device selection

## Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python 3.10+ |
| Deep Learning | PyTorch, torchvision (EfficientNetB0, ImageNet pretrained weights) |
| Explainability | Custom hook-based Grad-CAM |
| Image Processing | Pillow, OpenCV (opencv-python-headless) |
| Data Handling | NumPy, Pandas |
| Metrics | scikit-learn (train/test split, precision, recall, F1, confusion matrix) |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| Persistence | joblib (artifacts), PyTorch state dict (weights) |
| Utilities | tqdm |

## Dataset

The project uses the open-access **PlantVillage** dataset (the subset covering Pepper, Potato and Tomato):

- **20,638** leaf images, **15** classes, **3** crops (3 healthy and 12 disease classes)
- Stratified **70 / 15 / 15** split: 14,446 train, 3,095 validation, 3,097 test images (`random_state=42`)
- Class sizes range from 152 to 3,208 images (roughly 21:1 imbalance)

| Crop | Classes |
| --- | --- |
| Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

## Model and Training

**Preprocessing:** images are converted to RGB, resized to 224 x 224 and normalised with ImageNet mean and standard deviation.

**Augmentation (training only):** random horizontal flip, rotation, colour jitter and affine transforms.

**Architecture:** the pretrained EfficientNetB0 backbone is reused and the ImageNet head is replaced with `Dropout(0.3) + Linear(1280, 15)`.

**Two-phase training:**

| Phase | What trains | Learning rate | Epochs |
| --- | --- | --- | --- |
| 1 | Classifier head only (backbone frozen) | 1e-3 | 5 |
| 2 | Last 3 backbone blocks + head | 1e-4 | up to 10 |

**Other settings:** weighted CrossEntropyLoss (inverse-frequency weights), AdamW (weight decay 1e-4), ReduceLROnPlateau (patience 3, factor 0.5), early stopping (patience 5 on validation loss), gradient clipping (max norm 1.0), batch size 32.

```
PlantVillage -> preprocessing + augmentation -> stratified split
   -> Phase 1 (head only) -> Phase 2 (fine-tune last 3 blocks)
   -> early stopping on validation loss -> best checkpoint -> test evaluation
```

## Results

Evaluated on a held-out test set of **3,097 images**:

| Metric | Value |
| --- | --- |
| Accuracy | 99.61% |
| Macro Precision | 0.9958 |
| Macro Recall | 0.9967 |
| Macro F1-score | 0.9963 |
| Weighted F1-score | 0.9961 |

Every class reaches an F1-score of at least 0.99. The lowest-recall class is Tomato Septoria Leaf Spot (0.981) and the lowest-precision class is Tomato Leaf Mold (0.979). Grad-CAM heatmaps consistently highlight diseased lesion regions rather than the background, which supports that the model relies on real disease symptoms.

> These results are measured on lab-condition images and should be read as an upper bound. See [Limitations](#limitations).

## Dashboard

The Streamlit app (`app/streamlit_app.py`) has five pages:

1. **Overview:** key test-set metrics and a pipeline summary
2. **Data Explorer:** class distribution, crop-wise breakdowns and sample images
3. **Model Performance:** confusion matrix, training curves and per-class metrics
4. **Grad-CAM Explanations:** gallery of precomputed Grad-CAM overlays by class
5. **Live Prediction:** upload a leaf image to get the predicted class, confidence score, Grad-CAM overlay and treatment suggestion

<!-- Add screenshots here, for example: -->
<!-- ![Live Prediction](assets/screenshots/live_prediction.png) -->

## Project Structure

```
crop-disease-detection/
|-- app/
|   `-- streamlit_app.py       # 5-page interactive dashboard
|-- src/
|   |-- data_loader.py         # dataset scanning, splitting, transforms
|   |-- model_training.py      # model build, train, evaluate, save/load
|   |-- explainability.py      # Grad-CAM implementation
|   `-- run_pipeline.py        # end-to-end training pipeline
|-- models/
|   |-- best_model.pt          # trained model weights
|   |-- artifacts.pkl          # class names, transform config
|   |-- pipeline_results.json  # dataset stats, training config, test metrics
|   |-- confusion_matrix.json
|   |-- per_class_metrics.csv
|   `-- training_history.json
|-- assets/
|   |-- sample_images/         # sample leaf images per class
|   `-- gradcam_samples/       # precomputed Grad-CAM overlays
`-- requirements.txt
```

## Installation

**Requirements:** Python 3.10+, pip, and a web browser. A GPU is optional (CUDA and Apple MPS are supported; CPU-only also works). 8 GB RAM is the minimum and 16 GB is recommended.

```bash
# 1. Clone the repository
git clone https://github.com/Suraj5757/crop-disease-detection.git
cd crop-disease-detection

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

**Run the dashboard** (uses the trained model in `models/`):

```bash
streamlit run app/streamlit_app.py
```

**Retrain the model from scratch** (requires the PlantVillage dataset on disk; set its path in `src/data_loader.py`):

```bash
python src/run_pipeline.py
```

Training writes the best checkpoint, metrics, confusion matrix and training history to the `models/` folder.

## Limitations

- **Lab-condition data:** trained and evaluated only on PlantVillage (clean backgrounds, controlled lighting). Accuracy on real field photographs has not been validated and is expected to be lower.
- **Limited coverage:** only 15 classes across 3 crops.
- **Class imbalance:** small classes such as Potato Healthy (152 images) give less reliable metrics on tiny test subsets.
- **Single-disease assumption:** no handling of multiple diseases on one leaf, and no severity estimation.
- **Prototype only:** not tested with a smartphone camera pipeline or in low-connectivity rural settings.

## Future Scope

- Add more crops and disease categories
- Validate and fine-tune on field-collected, in-situ images
- Build a mobile app for offline, in-field diagnosis
- Real-time camera-based detection
- Cloud API deployment
- Disease severity estimation
- Multilingual support for smallholder farmers
- Detailed, regionally adapted treatment guide

## Author

**Suraj Singh**
B.Tech Computer Science (AI/ML), UPES Dehradun
[GitHub](https://github.com/Suraj5757) | [LinkedIn](https://www.linkedin.com/in/suraj-singh-86647b299) | bishtsuraj2005@gmail.com

Developed as part of a Remote Mentoring Internship (AI/ML) at Pheme Software Pvt. Ltd. (IBM Platinum Business Partner).

## Acknowledgements

- The PlantVillage dataset authors for the open-access benchmark
- The EfficientNet authors for the architecture
- Pheme Software Pvt. Ltd. and UPES for mentorship and resources
