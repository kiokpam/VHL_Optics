# VHL_Optics

## Overview

VHL_Optics is an AI project designed to analyze optical data from images collected by various mobile devices. The project provides tools for data processing, image analysis, feature extraction, and machine learning model training for classification and prediction of optical parameters.

## Key Features

1. **Optical Data Processing**:
   - Organize and process image data from mobile devices (smartphones).
   - Automatically detect and segment regions of interest (ROI) containing chemical reactions.
   - Apply image normalization to reduce noise and improve input consistency.

2. **Feature Extraction**:
   - Extract color and texture features from reaction regions for use in models.
   - Utilize techniques such as GLCM, entropy, and color histograms.

3. **Model Training**:
   - Train classification and regression models based on extracted features.
   - Support for popular algorithms:
        + Random Forest
        + SVM
        + KNN
        + MLP (Neural Network)
        + XGBoost
        + Logistic Regression
        + Naive Bayes
   - Evaluate models using cross-validation (Stratified K-fold), accuracy statistics, F1-score, and standard deviation.
   
4. **Automated Pipeline**:
   - Execute the entire workflow from data processing, feature extraction, to model training with a single command.

## Project Structure
```
   VHL_Optics/
├── data/
│   ├── _uploadRGB_5phones_sorted/     # Raw data
│   ├── square image/                  # Normalized images
│   ├── csv/                           # Model input features
│   └── models/                        # .pkl model files and evaluation results
│   └── ...
├── evaluation/                        # Model evaluation charts and tables
│   ├── accuracy_per_model.png
│   ├── f1_macro_per_model.png
│   ├── average_accuracy_summary.csv
│   └── average_f1_macro_summary.csv
│   └── ...
├── src/                               # Main source code
│   ├── config.py
│   ├── processing.py
│   ├── normalize.py
│   ├── model.py
│   ├── classifiers.py
│   ├── evaluate_models.py
│   └── predict.py
│   └── ....
├── requirements.txt
└── README.md
```
## Installation Guide

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd VHL_Optics
   ```
2. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare the data**:
   Create a `data` directory and place the raw image data inside.

## Usage Guide
1. Run the full pipeline  
   Execute the entire workflow for data processing, feature extraction, and model training with:
   ```bash
   python src/main.py
   ```
2. Evaluate models and export charts
      ```bash
      python -m evaluation.evaluate_models
      ```

## System Requirements
 - Python 3.9 or higher.
 - Libraries listed in `requirements.txt`.

## License
This project is licensed under the MIT License.
