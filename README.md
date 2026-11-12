# Machine Learning-Based Lithology Classification and Vshale Prediction

## 1. Project Overview

This project develops a machine learning workflow for lithology classification and continuous shale volume (Vshale) prediction using well-log data.

The workflow combines well-log data processing, feature engineering, exploratory data analysis, supervised machine learning, model evaluation, and interactive visualization through Streamlit.

The objective is to investigate how machine learning models can use conventional petrophysical well-log measurements to classify lithology and estimate shale volume.

## 2. Project Objectives

- Process and prepare well-log data for machine learning.
- Explore relationships between petrophysical measurements and lithology.
- Develop supervised classification models for lithology prediction.
- Develop a regression model for continuous Vshale prediction.
- Compare classification model performance.
- Visualize actual and predicted Vshale profiles.
- Build an interactive application for exploring predictions.

## 3. Dataset

**Dataset:** `weeks1_10_processed.csv`

**Source data:** WEEKS #1–10 well-log dataset.

The processed dataset contains 8,590 observations.

### Well-log features

| Feature | Description |
|---|---|
| DEPTH | Depth measurement |
| GR | Gamma Ray |
| RHOB | Bulk Density |
| DPOR | Density Porosity |
| CNPOR | Neutron Porosity |
| RILD | Deep Resistivity |
| RILM | Medium Resistivity |
| SP | Spontaneous Potential |
| PE | Photoelectric Factor |
| IGR | Gamma Ray Index |
| Vsh_Tertiary | Tertiary shale volume |
| Vsh_PreTertiary | Pre-Tertiary shale volume |
| Lith_Label | Encoded lithology class |

### Lithology classes

| Class | Lithology |
|---|---|
| 0 | Clean Sand |
| 1 | Silty Sand |
| 2 | Sandy Shale |
| 3 | Shale |

The target classes were derived from `Vsh_PreTertiary` using the following thresholds:

- Clean Sand: Vshale below 0.30
- Silty Sand: 0.30–0.50
- Sandy Shale: 0.50–0.80
- Shale: 0.80 and above

## 4. Machine Learning Models

### Classification

Three supervised classification algorithms are implemented:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

The models are trained and compared using classification performance metrics.

### Regression

A Random Forest Regressor is implemented to estimate continuous Vshale values.

The regression model uses:

- GR
- RHOB
- DPOR
- RILD
- SP
- PE

The target variable is `Vsh_PreTertiary`.

### Evaluation metrics

Classification:
- Accuracy
- Precision
- F1 Score
- Confusion Matrix
- Classification Report

Regression:
- R²
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## 5. Project Structure

```text
ml-lithology-classification/
│
├── app.py
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── 1055867994.las
│   │
│   └── processed/
│       └── weeks1_10_processed.csv
│
├── notebooks/
│   ├── Data_Processing_Feature_Engineering.ipynb
│   ├── Model_Development.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   └── 03_model_development.ipynb
│
├── src/
│   └── lithology/
│       └── models/
│           └── trainer.py
│
├── artifacts/
│   └── models/
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── requirements.txt
└── README.md
```

## 6. Technologies Used

| Technology | Application |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive well-log visualization |
| Scikit-learn | Machine learning and evaluation |
| XGBoost | Gradient boosting classification |
| Imbalanced-learn | Class balancing with SMOTE |
| Joblib | Model serialization |
| Streamlit | Interactive application |
| LASIO | LAS well-log data handling |

## 7. Installation and Setup

Clone the repository:

```powershell
git clone <repository-url>
cd ml-lithology-classification
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 8. Model Development

The model development workflow involves:

1. Loading the processed dataset.
2. Selecting predictor variables and target labels.
3. Splitting the dataset into training and testing sets.
4. Applying SMOTE to address class imbalance in the training data.
5. Training Logistic Regression, Random Forest and XGBoost.
6. Evaluating model performance.
7. Saving trained models using Joblib.
8. Training the Random Forest Vshale regressor.
9. Saving the regression model and evaluation results.

### Generated model artifacts

```text
artifacts/models/
│
├── logistic_regression.pkl
├── random_forest.pkl
├── xgboost.pkl
├── best_model.pkl
├── vsh_regressor.pkl
└── model_comparison.csv
```

## 9. Interactive Streamlit Application

The project includes a Streamlit application for exploring model predictions.

Run the application:

```powershell
streamlit run app.py
```

### Application capabilities

- Select between Logistic Regression, Random Forest and XGBoost.
- View classification performance metrics.
- Compare actual and predicted Vshale profiles.
- View actual and predicted lithology tracks.
- Explore depth intervals.
- Hover over the plots to inspect depth, Vshale values and lithology classes.
- View tabular prediction results.
- Download prediction results as CSV.

### Lithology colour scheme

| Lithology | Colour |
|---|---|
| Clean Sand | Yellow |
| Silty Sand | Orange |
| Sandy Shale | Brown |
| Shale | Dark Grey |

## 10. Important Methodological Considerations

The lithology labels in this project are derived from Vshale thresholds rather than independently verified core lithology measurements.

Because `Vsh_PreTertiary` is used to generate `Lith_Label`, including it as a classification predictor introduces direct target leakage.

Consequently, classification performance obtained with this feature should not be interpreted as evidence of independent geological prediction.

A further experiment should exclude `Vsh_PreTertiary` from classification predictors and compare the resulting performance against the baseline models.

The results should be described as predictions of Vshale-derived lithology classes unless independently validated against core or other geological reference data.

## 11. Current Project Scope

The project focuses on:

- Well-log data analysis.
- Petrophysical feature engineering.
- Supervised lithology classification.
- Continuous Vshale regression.
- Model evaluation and comparison.
- Interactive visualization.

## 12. Future Development

Potential extensions include:

- Testing additional classification algorithms.
- Feature importance and explainability analysis.
- Validation against independent wells.
- Improved generalization testing.
- Deployment of the trained models.

## 13. Conclusion

This project demonstrates an end-to-end machine learning workflow for well-log-based lithology classification and Vshale prediction.

By combining conventional petrophysical measurements with supervised learning and interactive visualization, the project provides a framework for investigating data-driven approaches to lithological interpretation.

**Project Category:** Machine Learning | Petroleum Engineering | Petrophysics | Well-Log Analysis