# E2E-ML: Customer Churn Prediction Pipeline

Complete ML pipeline for predicting customer churn using the ChurnModelling dataset. Includes data preprocessing, feature engineering, model training, and optimization.

## 🎯 Overview

End-to-end binary classification workflow with:
- Data preprocessing & EDA
- Feature engineering & encoding  
- Multiple ML algorithms
- Hyperparameter tuning & threshold optimization
- Cross-validation & SMOTE for class imbalance

## 📊 Dataset

**ChurnModelling** dataset with 10,000 bank customer records:

**Features:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary  
**Target:** Exited (0: retained, 1: churned)  
**Challenge:** Imbalanced classes

## � Quick Start

```bash
# Clone repository
git clone https://github.com/vishwajayawickrama/E2E-ML.git
cd E2E-ML

# Install dependencies
cd Explorarity_Data_Analysis && pip install -r requirements.txt
cd ../Model_Training_Evaluation && pip install -r requirements.txt

# Run EDA notebooks (0-4) then Model Training notebooks (0-5)
jupyter notebook
```

## 🔧 Pipeline

**EDA Phase:**
1. Missing values → Outliers → Feature binning → Encoding → Scaling

**ML Phase:**  
2. Data prep → Base models → K-fold validation → Multi-model comparison → Hyperparameter tuning → Threshold optimization

**Key Techniques:** SMOTE, Grid Search, ROC-AUC optimization

## 🛠️ Tech Stack

Python • Pandas • Scikit-learn • Jupyter • Matplotlib • Seaborn • Imbalanced-learn
