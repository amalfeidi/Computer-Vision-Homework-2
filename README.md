# Computer Vision – Homework 2: Image Classification

## Project Overview

This project explores image classification using transfer learning and traditional machine learning on the **Oxford-IIIT Pet Dataset**.

The objective is to classify images into 37 different cat and dog breeds using three approaches based on MobileNetV3-Small.

## Dataset

- Dataset: Oxford-IIIT Pet
- Number of classes: 37
- Training images: 2,944
- Validation images: 736
- Test images: 3,669
- Image input size: 224 × 224

The official trainval and test partitions were used. The trainval partition was divided into training and validation subsets using stratified sampling (`random_state=42`).

## Methods

### 1. Full Fine-Tuning
All MobileNetV3-Small parameters were trainable. The final classification layer was replaced to predict 37 classes.

### 2. Classifier-Head Training
The MobileNetV3 feature extractor was frozen, and only the classifier was trained.

### 3. Feature Extraction + Linear SVM
The pretrained MobileNetV3-Small network was frozen and used to extract 576-dimensional feature vectors. A Linear SVM was trained on these features.

A StandardScaler was fitted using training features only.

## Experimental Configuration

- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Full Fine-Tuning learning rate: 0.0001
- Classifier-Head learning rate: 0.001
- Batch size: 16
- Number of epochs: 10
- SVM: LinearSVC
- SVM C values: 0.01, 0.1, 1, 10
- Model selection: Best validation accuracy

## Results

| Approach | Validation Accuracy | Test Accuracy | Macro F1 |
|---|---:|---:|---:|
| Full Fine-Tuning | 87.77% | **84.71%** | **0.8419** |
| Classifier-Head | **88.18%** | 82.53% | 0.8240 |
| Feature Extraction + SVM | 87.50% | 83.35% | 0.8309 |

Full Fine-Tuning achieved the best test accuracy and macro F1-score.

## Repository Structure

```text
Computer-Vision-Homework-2/
├── README.md
├── requirements.txt
├── homework2.ipynb
└── models.zip
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The notebook uses the Oxford-IIIT Pet dataset through `torchvision.datasets.OxfordIIITPet`.

The dataset is downloaded automatically when running the dataset preparation cell.

## Running the Project

1. Open `homework2.ipynb` in Google Colab or Jupyter Notebook.
2. Install the dependencies.
3. Run the notebook cells in order.
4. Download the trained models and evaluation results.

## Saved Models

The `models.zip` archive contains:

- `best_full_finetuning.pth`
- `best_classifier_head.pth`
- `best_svm_model.joblib`
- `standard_scaler.joblib`

## Environment

The experiments were conducted in Google Colab using:

- Python 3.13.16
- PyTorch 2.14.1
- TorchVision 0.29.1
- Scikit-learn 1.6.1
- NumPy 2.4.6
- Pandas 2.2.3

## Evaluation

The final evaluation includes:

- Test accuracy
- Macro F1-score
- Confusion matrices
- Analysis of the most confused breed pairs
- Correct and incorrect prediction examples

The test set was kept separate from training and validation and used for final evaluation.

## Author

Computer Vision – Homework 2
