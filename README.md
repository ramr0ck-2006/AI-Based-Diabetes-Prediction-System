# AI-Based Diabetes Prediction System

## 1. Project Overview

The **AI-Based Diabetes Prediction System** is a machine-learning based application that estimates diabetes risk from health, lifestyle, demographic, and socioeconomic indicators.

The project uses supervised classification algorithms to learn patterns from the **CDC BRFSS Diabetes Health Indicators** dataset and provides an interactive Streamlit interface for generating model-based predictions.

The system also includes **SHAP-based explainability**, allowing users to understand which features contributed most to an individual prediction.

> **Important:** This project is an educational machine-learning system and is not intended to provide medical diagnosis or replace professional medical advice.

---

## 2. Objectives

The main objectives of the project are:

* Develop a binary classification system for diabetes prediction.
* Perform data preprocessing and exploratory data analysis.
* Remove exact duplicate records before model development.
* Compare multiple supervised machine-learning algorithms.
* Perform hyperparameter tuning using cross-validation.
* Evaluate models using Accuracy, Precision, Recall, F1-Score and ROC-AUC.
* Develop an interactive Streamlit web application.
* Provide model-estimated probability for individual inputs.
* Implement explainable AI using SHAP.
* Visualize important factors influencing predictions.

---

## 3. Dataset

### Dataset Used

**CDC BRFSS Diabetes Health Indicators**

Original dataset:

* Records: **253,680**
* Columns: **22**
* Target variable: `Diabetes_binary`
* Predictive features: **21**

### Target Variable

| Value | Meaning     |
| ----- | ----------- |
| `0`   | No diabetes |
| `1`   | Diabetes    |

### Features

The dataset contains the following predictive variables:

```text
HighBP
HighChol
CholCheck
BMI
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
GenHlth
MentHlth
PhysHlth
DiffWalk
Sex
Age
Education
Income
```

### Important Dataset Note

The dataset does **not** contain direct numerical blood-glucose measurements or numerical systolic/diastolic blood-pressure measurements.

Therefore, the application uses the actual dataset variables such as `HighBP`, `BMI`, `HighChol`, `GenHlth`, `Age`, lifestyle indicators, and other available features rather than creating unsupported input fields.

---

## 4. Data Preprocessing

The following preprocessing pipeline was used:

### Step 1 — Load Dataset

The dataset was loaded using Pandas.

### Step 2 — Duplicate Removal

Exact duplicate records were identified and removed.

```text
Original records       : 253,680
After duplicate removal: 229,474
Records removed        : 24,206
```

### Step 3 — Feature and Target Separation

The target variable:

```text
Diabetes_binary
```

was separated from the 21 predictive features.

### Step 4 — Train/Test Split

The dataset was divided using:

```text
Test size    : 20%
Random state : 42
Stratification: Yes
```

Final split:

```text
Training samples: 183,579
Testing samples : 45,895
```

Stratification was used to preserve the target-class distribution between training and testing data.

---

## 5. Exploratory Data Analysis

Exploratory analysis was performed to understand the dataset and identify important patterns.

The project generated:

* Target distribution visualization
* Correlation heatmap
* Diabetes distribution by BMI
* Diabetes distribution by age category
* Feature importance plots

The dataset is class-imbalanced, with the non-diabetic class substantially larger than the diabetic class. Therefore, accuracy alone was not considered sufficient for model evaluation.

---

## 6. Machine Learning Models

Three supervised classification algorithms were evaluated:

### 1. Logistic Regression

A linear classification baseline using:

* StandardScaler
* Balanced class weights
* Maximum iterations = 1000

### 2. Random Forest

The Random Forest model was configured with class balancing and multiple decision trees.

### 3. Gradient Boosting

Gradient Boosting was used to capture nonlinear relationships between health indicators and the target variable.

---

## 7. Baseline Model Results

The initial models were evaluated using the held-out test set.

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   71.43% |    31.83% | 76.01% | 44.87% |  0.8106 |
| Random Forest       |   77.46% |    36.19% | 62.10% | 45.73% |  0.8046 |
| Gradient Boosting   |   85.52% |    60.37% | 15.47% | 24.63% |  0.8187 |

The baseline results demonstrated that accuracy alone did not adequately represent the model's ability to identify the positive class.

---

## 8. Hyperparameter Tuning

Hyperparameter optimization was performed using:

```text
RandomizedSearchCV
StratifiedKFold
3-fold cross-validation
10 parameter combinations
Scoring metric: ROC-AUC
```

### Tuned Random Forest

Best parameters:

```python
{
    "n_estimators": 300,
    "min_samples_split": 2,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "max_depth": 10
}
```

Test ROC-AUC:

```text
0.8163
```

### Tuned Gradient Boosting

Best parameters:

```python
{
    "subsample": 0.8,
    "n_estimators": 250,
    "min_samples_split": 2,
    "min_samples_leaf": 4,
    "max_depth": 4,
    "learning_rate": 0.03
}
```

Test ROC-AUC:

```text
0.8193
```

---

## 9. Final Model

The **Tuned Gradient Boosting** model was selected as the primary model because it achieved the strongest overall combination of the evaluated metrics.

### Final Test Performance

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **80.31%** |
| Precision | **40.24%** |
| Recall    | **59.25%** |
| F1-Score  | **47.93%** |
| ROC-AUC   | **0.8193** |

The Random Forest model was retained as a comparison model.

### Selected Decision Threshold

The final classification threshold was selected through threshold analysis:

```text
Decision threshold = 0.25
```

At this threshold:

```text
Probability < 25% → Lower Risk classification
Probability ≥ 25% → Higher Risk classification
```

This threshold is a model-development decision based on the evaluation data and is **not a clinically validated risk boundary**.

---

## 10. Feature Importance

Feature importance analysis identified several influential variables.

For the Random Forest model, the most important features included:

1. BMI
2. Age
3. General Health
4. High Blood Pressure
5. Income
6. Physical Health
7. High Cholesterol
8. Education
9. Mental Health
10. Difficulty Walking

Gradient Boosting also identified major contributions from:

* High Blood Pressure
* General Health
* BMI
* Age
* High Cholesterol
* Difficulty Walking
* Heart Disease/Attack

Feature importance indicates model behavior and should not be interpreted as proof of medical causation.

---

## 11. Explainable AI

The application uses **SHAP (SHapley Additive exPlanations)** to provide instance-level explanations.

For each prediction, the application displays the features that contributed most strongly to the model output.

This improves transparency by helping users understand how individual input features influenced the prediction.

---

## 12. Streamlit Application

The application provides an interactive web interface containing:

### Patient Inputs

* High Blood Pressure
* High Cholesterol
* Cholesterol Check
* BMI
* Smoking
* Stroke history
* Heart Disease/Attack
* Physical Activity
* Fruit consumption
* Vegetable consumption
* Heavy Alcohol Consumption
* Healthcare Coverage
* Cost-related healthcare access
* General Health
* Mental Health Days
* Physical Health Days
* Difficulty Walking
* Sex
* Age Category
* Education Level
* Income Level

### Prediction Output

The application displays:

* Model-estimated probability
* Lower/Higher Risk classification
* Probability visualization
* Patient input summary
* SHAP explanation
* Model performance information

---

## 13. Probability Sanity Check

A probability-distribution analysis was performed on the 45,895 held-out test records.

Results:

| Statistic         | Probability |
| ----------------- | ----------: |
| Minimum           |       0.41% |
| Median            |       9.62% |
| Mean              |      15.45% |
| 90th percentile   |      39.56% |
| 95th percentile   |      48.27% |
| 99th percentile   |      63.93% |
| 99.9th percentile |      72.39% |
| Maximum           |      80.05% |

Only one test record reached or exceeded 80%.

This confirms that very high model probabilities are naturally uncommon for this trained model and dataset.

---

## 14. Project Structure

```text
AI-Based-Diabetes-Prediction-System/
│
├── .venv/
│
├── data/
│   └── diabetes_binary_health.csv
│
├── src/
│   ├── dataset_analysis.py
│   ├── eda_analysis.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── feature_analysis.py
│   ├── model_tuning.py
│   ├── final_model_evaluation.py
│   └── probability_sanity_check.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── tuned_random_forest.pkl
│   └── tuned_gradient_boosting.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── tuned_model_comparison.csv
│   ├── final_model_comparison.csv
│   ├── final_threshold_analysis.csv
│   ├── probability_sanity_check.csv
│   ├── confusion matrices
│   ├── ROC curves
│   ├── precision-recall curves
│   └── feature importance plots
│
├── app/
│   └── app.py
│
├── notebooks/
│
├── docs/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 15. Technologies Used

### Programming

* Python 3.12

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest
* Gradient Boosting

### Visualization

* Matplotlib
* Seaborn
* Plotly

### Explainable AI

* SHAP

### Web Application

* Streamlit

### Model Serialization

* Joblib

---

## 16. How to Run the Project

### Step 1 — Activate virtual environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2 — Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 3 — Run the application

```powershell
python -m streamlit run app/app.py
```

The Streamlit application will open in the browser.

---

## 17. Model Development Pipeline

```text
Dataset
   ↓
Data Inspection
   ↓
Duplicate Removal
   ↓
Exploratory Data Analysis
   ↓
Train/Test Split
   ↓
Baseline Models
   ├── Logistic Regression
   ├── Random Forest
   └── Gradient Boosting
   ↓
Model Evaluation
   ↓
Hyperparameter Tuning
   ↓
Threshold Analysis
   ↓
Final Gradient Boosting Model
   ↓
SHAP Explainability
   ↓
Streamlit Application
```

---

## 18. Limitations

The project has several important limitations:

1. The dataset does not contain direct blood-glucose measurements.
2. Blood pressure is represented as a binary `HighBP` indicator rather than numerical blood-pressure readings.
3. Age is represented using an ordinal category rather than exact age.
4. The dataset is class-imbalanced.
5. Model-estimated probabilities have not been clinically calibrated.
6. The selected threshold is based on machine-learning evaluation rather than a clinical guideline.
7. The system should not be interpreted as a medical diagnostic device.
8. Dataset patterns may not generalize equally to every population.

---

## 19. Future Improvements

Possible future extensions include:

* Probability calibration
* External validation using an independent dataset
* More advanced ensemble models
* Cross-dataset evaluation
* Additional clinical measurements such as glucose and HbA1c when appropriate data is available
* Model monitoring and drift detection
* Improved fairness and subgroup evaluation
* Secure deployment
* Database integration
* Patient history tracking
* Doctor/healthcare-provider dashboard

---

## 20. Conclusion

The AI-Based Diabetes Prediction System demonstrates an end-to-end machine-learning workflow for healthcare-oriented predictive analytics.

The project combines:

* Data preprocessing
* Exploratory analysis
* Multiple classification algorithms
* Hyperparameter optimization
* Threshold analysis
* Model evaluation
* Interactive prediction
* Explainable AI using SHAP
* Streamlit deployment

The final tuned Gradient Boosting model achieved an ROC-AUC of **0.8193** and an accuracy of **80.31%** on the held-out test set.

The resulting application provides an interactive and explainable interface while clearly communicating the limitations of machine-learning based health prediction.
