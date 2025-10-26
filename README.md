# 🛒 Customer Churn Prediction — Retail / E-commerce

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Modeling-yellow?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-EDA-9cf)
![XGBoost](https://img.shields.io/badge/XGBoost-Model%20Boosting-orange)
![GitHub](https://img.shields.io/badge/Project-Portfolio-green)

---

## 📌 Project Overview
This project analyzes customer data from an online retail store to predict which customers are likely to churn.  
By identifying high-risk customers, the business can proactively improve retention and revenue.

**Dataset:** [Online Retail Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/hassaneskikri/online-retail-customer-churn-dataset)

---

## 🎯 Business Problem
Customer churn directly impacts revenue. Predicting churn allows businesses to:  
- Target retention campaigns for high-risk customers  
- Enhance customer experience  
- Reduce revenue loss  

---

## 🧰 Tech Stack
| Category | Tools & Libraries |
|----------|------------------|
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| IDE | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

---

## 🔍 Project Workflow
1. **Data Understanding & Cleaning**  
   - Handle missing values, duplicates, and data types.  

2. **Exploratory Data Analysis (EDA)**  
   - Customer demographics and purchase patterns  
   - RFM (Recency, Frequency, Monetary) analysis  
   - Churn distribution visualization  

3. **Feature Engineering**  
   - Customer tenure, average basket size, frequency metrics  
   - Encoding categorical variables  

4. **Model Training & Evaluation**  
   - Train multiple models: Logistic Regression, Random Forest, XGBoost  
   - Evaluate with accuracy, precision, recall, F1-score, ROC-AUC  

5. **Insights & Recommendations**  
   - Identify key factors driving churn  
   - Provide actionable business strategies  

---
CustomerChurnProject/
│
├── data/
│   └── cleaned_churn_data.csv
│
├── notebooks/
│   ├── 01_EDA_and_Visualization.ipynb
│   └── 02_Model_Training_and_Evaluation.ipynb
│
├── scripts/
│   ├── feature_engineering.py
│   ├── Model_Prediction_Example.py
│   └── Best_Model_Prediction_and_Report.py
│
├── models/
│   └── LogisticRegression_model.pkl
│
├── reports/
│   ├── LogisticRegression_model_Predictions.csv
│   ├── LogisticRegression_model_Prediction_Report.pdf
│   ├── confusion_matrix.png
│   └── Model_Training_Summary.pdf
│
├── requirements.txt
└── README.md

## 📊 Model Performance

### 🧠 Model Training Summary (Actual Results)
| Model | Best Parameters | Accuracy | ROC-AUC |
|-------|-----------------|----------|----------|
| Random Forest | max_depth=10, n_estimators=100 | 0.475 | 0.4747 |
| XGBoost | max_depth=7, learning_rate=0.1 | 0.495 | 0.489 |
| Logistic Regression | C=0.1, solver=liblinear | 🏆 **0.515** | **0.512** |

✅ **Best Model:** Logistic Regression  
📂 **Saved As:** `models/LogisticRegression_model.pkl`

---

## 📊 Key Visual Insights
| Visualization | Description |
|---------------|-------------|
| 🧍‍♂️ **Churn Distribution** | Shows percentage of retained vs churned customers |
| 💸 **Top Predictive Features** | Identifies top drivers of churn (e.g., Total Spend, Last Purchase Days) |
| ⏱️ **Customer Tenure & Spend Patterns** | Shows how long-term vs new customers behave differently |

📁 *(All visualizations generated in `01_EDA_and_Visualization.ipynb` and stored under `/reports/`.)*

---

## 📂 Outputs & Reports
| Type | File | Description |
|------|------|-------------|
| 📘 **Model Summary PDF** | `reports/Model_Training_Summary.pdf` | Consolidated model evaluation |
| 📑 **Prediction Report PDF** | `reports/LogisticRegression_model_Prediction_Report.pdf` | Test predictions with visuals |
| 📊 **Predictions CSV** | `reports/LogisticRegression_model_Predictions.csv` | Actual vs predicted churn with probabilities |

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/bhavesh2418/Customer-Churn-Prediction-Retail.git

# Navigate to project directory
cd Customer-Churn-Prediction-Retail

# Install dependencies
pip install -r requirements.txt

# Run the main scripts
python scripts/Feature_Engineering.py
python scripts/Model_Training_and_Evaluation.py
python scripts/Best_Model_Prediction_and_Report.py
