# Data Science Project: Company Bankruptcy Prediction
## Project Overview

This project aims to develop a machine learning model that can predict the risk of company bankruptcy based on financial ratios. Using a dataset of Taiwanese companies with 95 financial features, this project implements various machine learning algorithms to provide an early warning system for stakeholders.

**Application (Streamlit) can be accessed at:** [https://bankruptprediction-project.streamlit.app/](https://bankruptprediction-project.streamlit.app/)

### Key Objectives

1.  **Identify factors** that cause company bankruptcy
2.  **Build the best prediction model** for bankruptcy prediction
3.  **Analyze model interpretability** and the impact of each financial variable

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset Information](#dataset-information)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Tech Stack](#tech-stack)
- [Business Recommendations](#business-recommendations)
- [Future Improvements](#future-improvements)

## Business Problem

### Problem Statement

Company bankruptcy creates a domino effect that harms various parties:

**Financial Impact:**
- Losses for creditors and banks
- Loss of investor capital
- Burden on the banking system

**Social Impact:**
- Mass layoffs
- Decrease in public purchasing power
- Increased burden on the government

**Operational Impact:**
- Supply chain disruption
- Chaos for customers
- Loss of productive assets

> *"The resulting impact can be so damaging for the company and other related parties. Bankruptcy is related to default, so the company's financial side is the main focus of this analysis. Identifying risks early becomes a crucial point as an alarm for the company to understand and comprehend the causes of bankruptcy or potential bankruptcy."*

### Research Questions

1.  What variables are the causes of bankruptcy?
2.  What is the best model for bankruptcy prediction?
3.  What is the impact of each variable on the model's prediction?

## Dataset Information
-   **Source:** [Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction/data) & UCI Machine Learning Repository
-   **Region:** Taiwan companies
-   **Total Features:** 95 financial ratios + 1 target variable
-   **Total Records:** 5,455 companies
-   **Target Distribution:** Highly imbalanced (~4% bankruptcy cases)

### Feature Categories:
-   Profitability ratios
-   Liquidity ratios
-   Leverage ratios
-   Activity/Efficiency ratios
-   Market-based ratios

## Methodology

### Data Preprocessing

-   **Outlier Detection & Treatment:** Statistical analysis with percentile capping (99th percentile)
-   **Feature Selection:** Multiple approaches used:
    -   Statistical testing (ANOVA F-test)
    -   Correlation analysis
    -   Literature-based selection
    -   Feature importance ranking
-   **Data Scaling:** RobustScaler implementation
-   **Class Imbalance:** SMOTE (Synthetic Minority Oversampling Technique)

### Models Evaluated

-   **XGBoost Classifier** **(Model Chosen)**
-   Gradient Boosting Classifier
-   AdaBoost Classifier
-   Extra Trees Classifier
-   Random Forest Classifier
-   Logistic Regression
-   K-Nearest Neighbors
-   Decision Tree Classifier
-   Support Vector Machine
-   LightGBM Classifier

### Model Selection Criteria

-   **Primary Metric:** Recall (minimize False Negatives)
-   **Secondary Metric:** F1-Score (balanced performance)
-   **Rationale:** High cost of missing actual bankruptcy cases
-   **Performance:** Fast performance, especially for hyperparameter tuning

## Key Results
### Model Performance After Tuning
```
Recall Score: 84%
F1-Score: Optimized for bankruptcy detection
Precision: Trade-off accepted for higher recall
```
### Top Risk Factors Identified

1.  **Total Debt/Total Net Worth** (Leverage) - Most Important
2.  **Persistent EPS** (Profitability)
3.  **Net Income to Total Assets** (ROA)
4.  **Borrowing Dependency** (Leverage)
5.  **Quick Ratio** (Liquidity)

### Key Insights from Model Interpretation

-   **Leverage Threshold:** Bankruptcy risk increases sharply when debt-to-net-worth ratio exceeds **1.5x**
-   **Borrowing Dependency:** Companies with borrowing dependency **>40%** show significantly higher risk
-   **ROA Critical Point:** Critical threshold at ROA **~0.6-0.7**
-   **EPS Warning:** EPS below **0.2** indicates a high bankruptcy risk

## Tech Stack

**Programming Language:**
-   Python 3.8+

**Key Libraries:**
-   `pandas` - Data manipulation
-   `numpy` - Numerical computing
-   `scikit-learn` - Machine learning
-   `xgboost` - Gradient boosting
-   `streamlit` - Web application
-   `matplotlib` & `seaborn` - Data visualization
-   `shap` - Model interpretation

## Business Recommendations

### For Management

-   **Debt Management:** Carefully monitor leverage ratios and borrowing dependency
-   **Profitability Focus:** Maintain ROA above critical thresholds
-   **Early Warning System:** Implement regular monitoring of identified risk factors
-   **Liquidity Planning:** Ensure sufficient liquid assets for operations

### For Investors & Creditors

-   **Due Diligence:** Focus on leverage and profitability metrics
-   **Risk Assessment:** Use model predictions as an additional screening tool
-   **Portfolio Monitoring:** Regular assessment of invested companies

## Future Improvements

### Technical Enhancements
-   **Feature Engineering:** Explore additional financial ratios and time-series features
-   **Deep Learning:** Experiment with Neural Networks using TensorFlow
-   **Ensemble Methods:** Combine multiple models for improved performance

### Data & Research
-   **Dataset Expansion:** Include more recent data and diverse geographic regions
-   **Expert Validation:** Collaborate with financial analysts and domain experts
-   **External Factors:** Incorporate macroeconomic indicators
-   **Longitudinal Analysis:** Track companies over multiple time periods

---
