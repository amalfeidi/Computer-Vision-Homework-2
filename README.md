# Computer Vision – Homework 2: Image Classification

## Project Overview

This project explores image classification using transfer learning and traditional machine learning on the **Oxford-IIIT Pet Dataset**.

The objective is to classify images into 37 different cat and dog breeds using three approaches based on the MobileNetV3-Small architecture.

## Dataset

- **Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **Number of classes:** 37
- **Training images:** 2,944
- **Validation images:** 736
- **Test images:** 3,669
- **Input image size:** 224 × 224 pixels

The official trainval and test partitions were used. The official trainval partition was divided into training and validation subsets using stratified sampling with `random_state=42` and a validation ratio of 20%.

The test set was kept separate and used only for final evaluation.

## Methods

### 1. Full Fine-Tuning

All MobileNetV3-Small parameters were trainable. The final classification layer was replaced to predict 37 classes.

### 2. Classifier-Head Training

The MobileNetV3-Small feature extractor was frozen, and only the classifier was trained.

### 3. Feature Extraction + Linear SVM

The pretrained MobileNetV3-Small network was frozen and used to extract 576-dimensional feature vectors.

A Linear SVM (`LinearSVC`) was trained on the extracted features. A `StandardScaler` was fitted using training features only and then applied to validation and test features.

## Experimental Configuration

| Parameter | Configuration |
|---|---|
| Architecture | MobileNetV3-Small |
| Pretrained weights | ImageNet |
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

### Data Preprocessing

Training images for the two neural-network approaches were augmented using:

- RandomResizedCrop (224 × 224)
- RandomHorizontalFlip
- ImageNet normalization

Validation and test images used resizing, center cropping, and ImageNet normalization.

For the SVM approach, the frozen feature extractor used the evaluation preprocessing pipeline without training-time augmentation.

## Results

The following table reports the best validation accuracy and final test performance.

| Approach | Validation Accuracy | Test Accuracy | Macro F1 |
|---|---:|---:|---:|
| Full Fine-Tuning | 87.77% | **84.71%** | **0.8419** |
| Classifier-Head | **88.18%** | 82.53% | 0.8240 |
| Feature Extraction + SVM | 87.50% | 83.35% | 0.8309 |

Full Fine-Tuning achieved the best test accuracy and Macro F1-score, while Classifier-Head Training achieved the highest validation accuracy.

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

### Results Archive

The `results.zip` archive contains:

- Comparison table
- Final test metrics
- Training history
- Class distribution
- Confusion matrices for all three approaches
- Training and validation curves
- Correct and incorrect prediction examples

### Saved Models

The `models.zip` archive contains:

- `best_full_finetuning.pth`
- `best_classifier_head.pth`
- `best_svm_model.joblib`
- `standard_scaler.joblib`

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

## Running the Project

1. Open `homework2.ipynb` in Google Colab or Jupyter Notebook.
2. Install the required dependencies.
3. Run the notebook cells in order.
4. The dataset will be downloaded automatically.
5. Run the training and evaluation cells to reproduce the experiments.

The saved model and result archives are provided separately in the repository.

## Reproducibility

- The train/validation split uses `random_state=42`.
- The split indices are provided in `data_splits.npz`.
- Class distribution information is provided in `class_counts.csv`.
- The notebook contains the preprocessing, training, feature extraction, and evaluation procedures.

GPU operations may introduce small numerical variations between runs.

## Environment

The experiments were conducted in Google Colab using a Tesla T4 GPU.

- Python 3.13.16
- PyTorch 2.14.1+cu130
- TorchVision 0.29.1+cu130
- Scikit-learn 1.6.1
- NumPy 2.4.6
- Pandas 2.2.3

## Evaluation

The final evaluation includes:

- Test accuracy
- Macro F1-score
- Confusion matrices
- Analysis of frequently confused breeds
- Correct and incorrect prediction examples
- Training and validation loss and accuracy curves

The test set was not used for model selection or hyperparameter tuning.

## Author

Computer Vision – Homework 2
