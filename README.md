# 📊 Predictive Modeling and Data Analysis Projects

This repository contains my work for three assignments completed as part of my data analysis coursework.  
Each assignment applies econometric and machine learning methods to real-world datasets, focusing on model building, evaluation, and interpretation.

---

## 🧮 Assignment 1 — Predicting Earnings per Hour

**Goal:**  
Analyze the CPS-Earnings dataset and build multiple linear regression models to predict individuals’ *earnings per hour*.

**Key tasks:**
- Select and filter a specific **occupation** from the CPS dataset.  
- Build **four OLS models** (from simple to complex) using demographic and employment-related predictors.  
- Compare models using:
  - Root Mean Squared Error (RMSE)
  - Cross-validated RMSE
  - Bayesian Information Criterion (BIC)
- Discuss the trade-off between **model complexity and performance**.
- Include a one-page summary report and reproducible code.

**Dataset:** [CPS-Earnings Dataset](https://osf.io/g8p9j/)  
**Occupation Codes:** [Available here](https://osf.io/57n9q/)  

---

##  Assignment 2 — Airbnb Pricing Model

**Business Case:**  
Develop a **pricing model** for a chain of Airbnb properties based on Inside Airbnb listings data.

**Part I – Modelling**
- Choose the ***Paris dataset*** (≥10,000 listings) and prepare the data:
  - Handle missing values, extract amenities, and engineer features.
- Build **five predictive models:**
  1. OLS  
  2. LASSO  
  3. Random Forest  
  4. Boosting model (of choice)  
- Compare model fit and computation time (create a “horse race” table).
- Analyze **feature importance** for Random Forest and Boosting models.

**Part II – Validity**
- Test model robustness using two additional datasets:
  1. A later time period of the same city  
  2. A different city from the same region  
- Compare model performance and discuss transferability.

---

## Assignment 3 — Predicting Firm Growth (Bisnode Dataset)

**Objective:**  
Build models to predict **fast-growing firms** using panel data from the Bisnode dataset (2010–2015).

**Steps:**
1. **Target Design:**  
   Define “fast growth” based on firm performance (e.g., revenue or size) between 2012–2013 or 2012–2014.  
   Discuss alternative definitions and justify your choice using corporate finance theory.
2. **Modeling:**
   - Build at least **three models**, including:
     - One **Logit model**
     - One **Random Forest**
   - Evaluate using **cross-validation** and **expected loss**.
3. **Classification:**
   - Define a business-oriented **loss function** (False Positives vs. False Negatives).
   - Identify optimal classification thresholds for each model.
4. **Discussion:**
   - Present a **confusion matrix** and interpret model usefulness.
   - Compare results across **manufacturing vs. services** industries.
---

##  Skills Demonstrated
- Data wrangling and feature engineering  
- Regression and machine learning (OLS, LASSO, RF, Boosting, Logit)  
- Model evaluation: RMSE, BIC, cross-validation, confusion matrices  
- Business and economic interpretation of results  
- Reproducible research workflow using Python and GitHub  

---
