# Credit Risk Probability Model Using Alternative Data

## Project Overview

This project aims to develop an end-to-end Credit Risk Probability Model for Bati Bank using alternative transaction data from an eCommerce platform. The goal is to support a Buy-Now-Pay-Later (BNPL) service by estimating the likelihood that a customer represents a credit risk.

Since the dataset does not contain a direct loan default indicator, a proxy target variable will be developed using customer behavioral patterns derived from Recency, Frequency, and Monetary (RFM) analysis.

The final solution will include:

* Customer risk classification
* Risk probability prediction
* Credit score generation
* Loan amount and duration recommendation
* Model deployment as a REST API
* Automated testing and CI/CD integration

---

## Dataset Description

The dataset contains transaction-level records collected from an eCommerce platform.

### Key Features

| Feature              | Description                           |
| -------------------- | ------------------------------------- |
| TransactionId        | Unique transaction identifier         |
| CustomerId           | Unique customer identifier            |
| Amount               | Transaction amount                    |
| Value                | Absolute value of transaction amount  |
| ProductCategory      | Product category purchased            |
| ChannelId            | Transaction channel                   |
| ProviderId           | Service provider                      |
| TransactionStartTime | Transaction timestamp                 |
| FraudResult          | Fraud flag (1 = Fraud, 0 = Non-Fraud) |

---

# Credit Scoring Business Understanding

## Basel II and Model Interpretability

The Basel II Accord emphasizes accurate risk measurement, transparency, documentation, and ongoing monitoring of credit risk models. As a result, credit scoring models should be interpretable enough for business stakeholders, auditors, and regulators to understand how risk decisions are made.

Well-documented and explainable models improve trust, support regulatory compliance, and facilitate model validation.

## Why a Proxy Target Variable Is Needed

The dataset does not contain a direct indicator of customer default behavior. Because supervised machine learning requires a target variable, a proxy measure of credit risk must be created.

This project will use customer transaction behavior, specifically Recency, Frequency, and Monetary (RFM) metrics, to identify customers who may represent higher or lower credit risk.

## Model Trade-Offs in a Regulated Environment

Simple models such as Logistic Regression offer strong interpretability and are easier to explain to regulators. More advanced models such as Gradient Boosting may achieve better predictive performance but are generally less transparent.

A key objective of this project is to balance predictive performance with explainability and regulatory requirements.

---

# Exploratory Data Analysis (EDA)

The dataset was explored to understand customer behavior, identify potential data quality issues, and guide future feature engineering.

## EDA Activities Completed

* Dataset structure inspection
* Data type validation
* Summary statistics analysis
* Numerical feature distribution analysis
* Categorical feature distribution analysis
* Correlation analysis
* Missing value assessment
* Outlier detection
* Customer activity analysis
* Time-based transaction analysis

---

## Key Insights

### 1. Transaction Amounts Are Highly Skewed

Most transactions involve relatively small amounts, while a small number of transactions have extremely large values. This indicates a right-skewed distribution and the presence of significant outliers.

### 2. Customer Activity Varies Significantly

Most customers perform only a few transactions, while a small number of customers are highly active. This finding supports the use of transaction frequency as an important behavioral feature.

### 3. Transaction Volume Changes Over Time

Transaction activity increased substantially during December before declining in January and February, suggesting potential seasonal purchasing patterns.

### 4. Large Transaction Outliers Exist

Several transactions contain unusually high values. These observations may represent legitimate customer behavior and will be further evaluated during feature engineering.

### 5. Behavioral Features May Be Strong Predictors

The observed differences in customer activity and spending behavior suggest that RFM-based features could provide meaningful signals for proxy credit risk modeling.

---

# Next Steps

The next phase of the project will focus on constructing customer-level RFM metrics and creating a proxy target variable to represent credit risk. This target will be used to train and evaluate multiple machine learning models for risk prediction.
