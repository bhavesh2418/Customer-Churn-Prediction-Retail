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

## 📊 Model Performance (Example)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.82 | 0.78 | 0.74 | 0.76 |
| Random Forest | 0.86 | 0.83 | 0.80 | 0.81 |
| XGBoost | **0.88** | **0.85** | **0.83** | **0.84** |

*(Update with your actual results after model training)*

---

## 🖼️ Visualization Preview
- **Churn Distribution:** ![Churn Distribution](images/churn_distribution.png)  
- **Top Features Influencing Churn:** ![Feature Importance](images/feature_importance.png)  
- **RFM Segmentation:** ![RFM Plot](images/rfm_plot.png)  

*(Add actual plots after running your notebooks)*

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/bhavesh2418/Customer-Churn-Prediction-Retail.git

# Navigate to project directory
cd Customer-Churn-Prediction-Retail

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
