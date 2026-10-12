# Tropospheric Ozone Prediction Using Machine Learning

## 📌 Project Overview

Tropospheric ozone (O₃) is a harmful air pollutant found near the Earth's surface. It forms when nitrogen oxides (NOx) and volatile organic compounds (VOCs) react in the presence of sunlight.

This project uses Machine Learning regression models to predict daily ground-level ozone concentration using weather conditions and air pollution measurements.

The project compares three regression models to evaluate their prediction performance on unseen data.

## 🎯 Objectives

- Predict daily ground-level ozone concentration.
- Study the relationship between weather conditions, precursor pollutants, and ozone levels.
- Implement and compare three Machine Learning regression models.
- Evaluate model performance using standard regression metrics.

## 📊 Dataset

**Source:** US Environmental Protection Agency (EPA) AirData

**Period:** 2016–2017

**Dataset size:** 9,651 processed records

### Input Features

- Temperature
- Atmospheric pressure
- Wind speed
- Month
- Nitrogen dioxide (NO₂)
- Volatile organic compounds (VOCs)

### Target Variable

- Daily ozone concentration (ppm)

The dataset is prepared by matching measurements using monitoring site and date, followed by data cleaning and preprocessing.

## 🤖 Machine Learning Models

### 1. Linear Regression
Used as the baseline model to learn linear relationships between input features and ozone concentration.

### 2. Support Vector Regression (SVR)
Uses a polynomial kernel to model nonlinear relationships between environmental features and ozone concentration.

### 3. Neural Network
Uses a multilayer perceptron with three hidden layers, 200 neurons per layer, and ReLU activation to learn complex patterns in the data.

## ⚙️ Methodology

1. Download daily environmental measurements from EPA AirData.
2. Clean the dataset and handle incomplete or duplicate records.
3. Match measurements using monitoring site and date.
4. Prepare input features and the target variable.
5. Split the data chronologically:
   - Training: 2016
   - Testing: 2017
6. Standardize input features using StandardScaler within the model pipelines.
7. Train and evaluate the regression models.
8. Compare actual and predicted ozone concentrations.

## 📈 Evaluation Metrics

- **R² Score:** Measures how much variation in ozone concentration is explained by the model. Higher is better.
- **Mean Absolute Error (MAE):** Measures average absolute prediction error. Lower is better.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors more strongly. Lower is better.

## 🏆 Results

Results on the unseen 2017 test dataset:

| Model | Test R² | Test MAE | Test RMSE |
|---|---:|---:|---:|
| Neural Network | 0.4473 | 0.00672 | 0.00865 |
| Linear Regression | 0.2938 | 0.00782 | 0.00978 |
| Polynomial SVR | -0.2445 | 0.01066 | 0.01298 |

The Neural Network achieved the best test performance among the three implemented models, with the highest R² and lowest MAE and RMSE.

These results apply to the selected dataset, preprocessing pipeline, model configurations, and test period.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Jupyter Notebook / VS Code

## 📁 Project Structure

```text
Tropospheric-Ozone-Prediction/
├── prepare_data.py
├── train_models.py
├── results/
│   ├── model_comparison.csv
│   └── actual_vs_predicted.png
├── models/
├── README.md
└── requirements.txt
```

*Note: Update this structure to match the actual files and folders in your repository.*

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible version
- pip package manager

### Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Project

Prepare the dataset:

```bash
python prepare_data.py
```

Train and evaluate the models:

```bash
python train_models.py
```

*Note: These commands assume the scripts support the stated workflow. Check the actual filenames, dependencies, and execution instructions before publishing.*

## 🔮 Future Enhancements

- Use a larger historical dataset.
- Include additional geographical and environmental features.
- Perform feature contribution analysis.
- Tune model hyperparameters.
- Evaluate additional regression algorithms.
- Extend the project to ozone-category classification.

## 👥 Contributors

- Tarun G K PES2UG24CS557
- Srinivas PES2UG24CS505

## 📄 Disclaimer

This is an academic Machine Learning mini-project. Predictions are experimental and are not intended to replace official air-quality monitoring or public health advisories.

## 🙏 Acknowledgements

- US Environmental Protection Agency (EPA) AirData
- Scikit-learn documentation
