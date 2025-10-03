## VHL Optics – PPM Prediction (Global Model)

**VHL Optics** is an AI project for predicting the concentration (ppm) of chemical samples based on images captured from various mobile devices. In the initial version, a separate model was trained for each device type. However, the current version has been **refactored** to use **a single global model** for all devices. Device information is no longer used as a training feature, simplifying the workflow and making it easier to support new devices.

### Highlights

- **Automated data processing**: Scans raw data, generates metadata files, and crops regions of interest (ROI) to standardize images.
- **Feature extraction for regression**: Computes color channel statistics, GLCM contrast, entropy, edge density, etc. for each image.
- **Unified regression model**: Uses RandomForest or XGBoost trained on all data, without device columns.
- **User-friendly interface**: Streamlit app for uploading images and instantly receiving ppm predictions.
- **Batch prediction**: Predicts ppm for multiple samples from a feature file and saves results to CSV.

### Project Structure

```
├── app.py           # Simple Streamlit app for single and batch predictions
├── main.py          # Entry point: e2e pipeline, feature extraction, training, Streamlit
├── config.py        # Data paths and constants
├── loading.py       # Metadata creation and loading
├── processing.py    # Image processing, ROI and square cropping
├── normalize.py     # Regression feature extraction (no device info)
├── model.py         # Regression model training (per-device & global)
├── predict.py       # PPM prediction (single & batch)
├── roi.py           # ROI cropping utilities
├── squares.py       # Square contour detection
├── data/            # (auto-created) raw and processed data
```

### Installation

1. **Clone the repository** and install dependencies:

```bash
git clone <repository-url>
cd VHL_Optics_Regressor
pip install -r requirements.txt
```

If `requirements.txt` is missing, install at least:

```
numpy
pandas
opencv-python
scikit-learn
xgboost
streamlit
scikit-image
tqdm
```

2. **Prepare data**: Place raw images in `data/full/HP5_data` with the structure: *chemical_type / device / capture_id / image*. The app will automatically scan and generate metadata.

### Usage

#### Run the full pipeline (e2e)

This command will generate metadata, process images, extract features, and train the global model:

```bash
python main.py e2e
```

After running, feature files will be saved to `data/csv/features_all.csv` and trained models to `data/models/`.

#### Extract features and train separately

You can run each step individually:

```bash
# Feature extraction (when square images are ready)
python main.py feature

# Train the global model (when features_all.csv is available)
python main.py train
```

#### Predict ppm for a single image

Use the Streamlit interface:

```bash
streamlit run app.py
```

Or run `python main.py` (defaults to Streamlit). Users can upload an image, select a model (**RF** or **XGB**), and get the ppm result. No device info required.

You can also call the prediction function from Python:

```python
from predict import predict_regression_general

ppm = predict_regression_general('path/to/image.jpg', model_choice='RF')
print(f"Predicted ppm: {ppm:.2f}")
```

#### Batch prediction

For batch predictions, use the Streamlit app (`app.py`) or call:

```python
from predict import predict_test_set_general

results = predict_test_set_general(output_dir='batch_predictions')
print(results.head())
```

Results are saved to `predictions_general.csv` with columns: `id_img`, `true_ppm`, `pred_rf_ppm`, `pred_xgb_ppm`, `diff_rf_pct`, `diff_xgb_pct`.

### Note on Device Information

The current version **does not use** any device information as a training feature. Previous versions trained separate models per device, which made it difficult to support new devices. The global model is simpler and works for all images.

### System Requirements

* Python 3.8 or higher.
* Sufficient memory for image processing and model training (GPU recommended for XGBoost with large datasets).

### License
```
This project is distributed under the MIT License. You are free to use and modify the source code.

```

