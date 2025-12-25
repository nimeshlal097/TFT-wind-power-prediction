# ⚡ TFT Wind Power Prediction

A production-ready wind power forecasting system using **Temporal Fusion Transformers (TFT)** with integrated explainability (XAI), uncertainty quantification, and an interactive Dash dashboard.

## 📋 Overview

This repository implements a state-of-the-art deep learning pipeline for wind power prediction, combining:

- **Temporal Fusion Transformer**: Advanced attention-based architecture for multivariate time series forecasting
- **Probabilistic Forecasting**: Quantile regression for confidence intervals (2%, 10%, 50%, 90%, 98%)
- **Feature Attribution**: Variable selection networks and temporal attention weights for model interpretability
- **Interactive Dashboard**: Real-time forecasts, backtest validation, and feature importance visualization
- **Comprehensive Evaluation**: MAE, RMSE, MAPE, R² metrics with residual analysis

## 🎯 Features

### Model Capabilities
- ✅ Multi-step ahead forecasting (12-hour horizon by default)
- ✅ Uncertainty quantification with probabilistic predictions
- ✅ Interpretable predictions via attention mechanisms

### Dashboard Features
- 📊 **Forecast Tab**: Backtest validation + future 12h predictions with confidence bands
- 🧠 **Insights Tab**: Feature importance heatmaps + temporal attention visualization
- 📈 **Metrics Tab**: Performance cards (MAE, RMSE, MAPE, R²) + residual analysis + actual vs predicted scatter

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tft-wind-power-prediction.git
cd tft-wind-power-prediction

# Create virtual environment
conda create -n tft python=3.10 -y
conda activate tft

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
jupyter notebook dash.ipynb
```

Then:
1. Upload a CSV file with columns: `TIMESTAMP`, `TARGETVAR` (power output), `U10` (wind speed), and other meteorological features
2. Click **"🚀 Run Analysis"** to generate forecasts
3. Explore tabs for forecasts, insights, and metrics

### 3. Sample Data Format

Your CSV should have the following structure:

```csv
TIMESTAMP,TARGETVAR,U10,V10,T2M,RH2M,SSRD,group
2023-01-01 00:00,150.5,8.2,3.1,5.2,65,0,0
2023-01-01 01:00,165.2,8.5,3.3,5.1,66,0,0
...
```

**Required Columns:**
- `TIMESTAMP`: ISO format datetime
- `TARGETVAR`: Wind power output (MW)
- `U10`, `V10`,`U100`, `V100`: Wind components (m/s)


## 🔧 Configuration

Edit these constants in `dash.ipynb` to customize behavior:

```python
MODEL_CHECKPOINT = "tft_model.ckpt"          # Path to trained model
METADATA_FILE = "tft_metadata.pkl"           # Model metadata
DEFAULT_FORECAST_HORIZON = 12                # Hours ahead to forecast
CONTEXT_WINDOW = 24                          # Historical context (hours)
```

## 📊 Input Data Requirements

- **Minimum Length**: 48 hours (24h context + 24h buffer)
- **Frequency**: Hourly resampling (missing values interpolated)
- **Format**: CSV with datetime index
- **Missing Values**: Automatically handled via forward/backward fill + interpolation

## 🧠 Model Insights

### Feature Importance (Variable Selection Network)
The TFT's variable selection network learns which features are important:
- **Encoder variables**: Historical meteorological data used for context
- **Decoder variables**: Features used during forecasting horizon
- **Static variables**: Fixed metadata (location, capacity, etc.)

### Temporal Attention
Attention weights show which historical time steps the model focused on:
- Peak attention typically on recent past and same-hour patterns
- Reveals daily and weekly seasonality effects

### Interpretability Outputs
- Feature importance percentages
- Temporal attention curves
- Residual analysis with error bounds

## 📈 Model Performance Metrics

The dashboard displays:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MAE** | Mean(abs(actual - predicted)) | Average absolute error in MW |
| **RMSE** | √Mean((actual - predicted)²) | Penalizes larger errors |
| **MAPE** | Mean(abs((actual - predicted)/actual)) | Percentage error (scale-independent) |
| **R²** | 1 - SS_res/SS_tot | Proportion of variance explained (max: 1.0) |



Last Updated: January 2025
