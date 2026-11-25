# Computer Vision – Homework 2: Image Classification

## Project Overview

This project explores image classification using transfer learning and traditional machine learning on the **Oxford-IIIT Pet Dataset**.

The objective is to classify images into 37 different cat and dog breeds using three approaches based on the MobileNetV3-Small architecture.

The three approaches are:
1. Full Fine-Tuning
2. Classifier-Head Training
3. Feature Extraction with a Linear SVM

## Dataset

- **Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **Number of classes:** 37 (12 cat breeds and 25 dog breeds)
- **Official trainval partition:** 3,680 images
- **Training images:** 2,944
- **Validation images:** 736
- **Official test partition:** 3,669 images
- **Input image size:** 224 × 224 pixels

The official trainval and test partitions were used. The official trainval partition was divided into training and validation subsets using stratified sampling with `random_state=42` and a validation ratio of 20%.

The official test partition was kept separate from training and model selection and was used for final evaluation.

### Class Distribution

The number of images per breed in the training, validation, and test sets is provided in `class_counts.csv`.

### Dataset License and Citation

The Oxford-IIIT Pet Dataset is available under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license. Copyright of the individual images remains with their original owners.

The dataset is not redistributed in this repository. It is downloaded automatically through TorchVision when running the notebook.

**Original publication:**

O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, "Cats and Dogs," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

Dataset website: https://www.robots.ox.ac.uk/~vgg/data/pets/

License: https://creativecommons.org/licenses/by-sa/4.0/

## Architecture and Pre-trained Weights

All three approaches use:

- **Architecture:** MobileNetV3-Small
- **Pre-trained weights:** `MobileNet_V3_Small_Weights.DEFAULT`
- **Pre-training dataset:** ImageNet-1K
- **Number of output classes:** 37

The final classification layer was replaced with a fully connected layer producing 37 output scores, one for each breed.

## Methods

### 1. Full Fine-Tuning

All MobileNetV3-Small parameters were trainable, including the convolutional feature extractor and the classifier.

The final classification layer was replaced to predict 37 classes.

- Trainable parameters: 1,555,781
- Optimizer: Adam
- Learning rate: 0.0001
- Loss function: CrossEntropyLoss
- Batch size: 16
- Number of epochs: 10
- Model selection: Checkpoint with the highest validation accuracy

### 2. Classifier-Head Training

The convolutional feature extractor (`model.features`) was frozen, and only the fully connected classifier layers were trained.

The trainable classifier layers were `classifier[0]` and `classifier[3]`. The intermediate activation and dropout layers contain no trainable parameters.

- Trainable parameters: 628,773
- Frozen parameters: 927,008
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Batch size: 16
- Number of epochs: 10
- Model selection: Checkpoint with the highest validation accuracy

The frozen feature extractor was kept in evaluation mode to prevent Batch Normalization statistics from being updated during training.

### 3. Feature Extraction + Linear SVM

The pre-trained MobileNetV3-Small network was frozen and used as a feature extractor.

Feature vectors were obtained after the feature extractor and global average pooling operation, producing 576-dimensional vectors.

A Linear SVM (`LinearSVC`) was trained using scikit-learn.

- Trainable MobileNet parameters: 0
- Feature dimensionality: 576
- Classifier: LinearSVC
- Tested C values: 0.01, 0.1, 1, 10
- Model selection: Highest validation accuracy
- Best C value: 0.01

A `StandardScaler` was fitted using training features only and then applied to validation and test features.

The SVM itself learns classifier parameters; zero trainable parameters refers only to the frozen MobileNet network.

## Data Preprocessing

For the two neural-network approaches, training images were augmented using:

- `RandomResizedCrop(224)`
- `RandomHorizontalFlip(p=0.5)`
- ImageNet normalization

Validation and test images used the evaluation preprocessing associated with the selected pre-trained weights:

- Resize to 256 pixels
- Center crop to 224 × 224
- ImageNet normalization

The normalization parameters were:

- Mean: `[0.485, 0.456, 0.406]`
- Standard deviation: `[0.229, 0.224, 0.225]`

For the SVM approach, the frozen feature extractor used the evaluation preprocessing pipeline without training-time augmentation.

No additional class-balancing technique was applied.

## Experimental Configuration

| Parameter | Configuration |
|---|---|
| Architecture | MobileNetV3-Small |
| Pre-trained weights | ImageNet-1K |
| Input size | 224 × 224 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Full Fine-Tuning learning rate | 0.0001 |
| Classifier-Head learning rate | 0.001 |
| Batch size | 16 |
| Number of epochs | 10 |
| SVM classifier | LinearSVC |
| SVM C values | 0.01, 0.1, 1, 10 |
| Model selection | Best validation accuracy |

## Results

The following table reports the best validation accuracy and final test performance.

| Approach | Validation Accuracy | Test Accuracy | Macro F1-score |
|---|---:|---:|---:|
| Full Fine-Tuning | **88.18%** | **84.71%** | **0.8419** |
| Classifier-Head | 86.82% | 82.53% | 0.8240 |
| Feature Extraction + SVM | 87.50% | 83.35% | 0.8309 |

### Computational Cost

| Approach | Trainable Parameters | Training Time (min) |
|---|---:|---:|
| Full Fine-Tuning | 1,555,781 | 3.70 |
| Classifier-Head | 628,773 | 3.28 |
| Feature Extraction + SVM | 0* | 37.71 |

*For Feature Extraction + SVM, zero refers to trainable MobileNet parameters. The Linear SVM learns its own classifier parameters.*

The reported SVM time includes feature extraction and classifier training/selection.

Full Fine-Tuning achieved the best test accuracy and Macro F1-score. Classifier-Head Training required fewer trainable parameters and less training time. Feature Extraction + SVM had the highest measured computational cost in this experiment.

Small differences in test performance should not be interpreted as statistically significant without repeated experiments.

## Evaluation and Error Analysis

The final evaluation includes:

- Test accuracy
- Macro-averaged F1-score
- Confusion matrices for all three approaches
- Analysis of frequently confused breeds
- Correct and incorrect prediction examples
- Training and validation loss curves
- Training and validation accuracy curves

The test set was not used for model selection or hyperparameter tuning.

## Repository Structure

```text
Computer-Vision-Homework-2/
├── README.md
├── requirements.txt
├── homework2.ipynb
├── models.zip
├── results.zip
├── data_splits.npz
└── class_counts.csv
```

### Saved Models

The `models.zip` archive contains:

- `best_full_finetuning.pth`
- `best_classifier_head.pth`
- `best_svm_model.joblib`
- `standard_scaler.joblib`

### Results Archive

The `results.zip` archive contains:

- Comparison table
- Final test metrics
- Training history
- Class distribution
- Confusion matrices for all three approaches
- Training and validation curves
- Correct and incorrect prediction examples

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The notebook uses the Oxford-IIIT Pet dataset through:

```python
torchvision.datasets.OxfordIIITPet
```

The dataset is downloaded automatically when running the dataset preparation cells.

No manual dataset redistribution is required.

## Running the Project

The experiments can be run using Google Colab or a local Jupyter Notebook environment.

1. Clone or download this repository.
2. Open `homework2.ipynb` in Google Colab or Jupyter Notebook.
3. Install the required dependencies.
4. Run the notebook cells in order.
5. The Oxford-IIIT Pet dataset will be downloaded automatically.
6. Run the training, feature extraction, and evaluation cells to reproduce the three approaches and generate the results.

A GPU is recommended for faster execution of the neural-network experiments.

The saved checkpoints and result archives are also provided in the repository.

## Reproducibility

- The train/validation split uses `random_state=42`.
- The exact split indices are provided in `data_splits.npz`.
- Class distribution information is provided in `class_counts.csv`.
- The notebook contains the data preprocessing, training, feature extraction, and evaluation procedures.
- The original training experiments were not run with all global Python, NumPy, and PyTorch seeds explicitly configured before training.
- Deterministic algorithms were not enabled during the original experiments.
- GPU operations may introduce numerical variations between runs.

The saved checkpoints and results correspond to the experiments reported in this project.

## Environment

The experiments were conducted in Google Colab using a Tesla T4 GPU.

| Software | Version |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| TorchVision | 0.29.1+cu130 |
| Scikit-learn | 1.6.1 |
| NumPy | 2.4.6 |
| Pandas | 2.2.3 |
| Joblib | 1.6.0 |

## Author

Computer Vision – Homework 2
