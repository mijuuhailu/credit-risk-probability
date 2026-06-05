# Credit Risk Probability Model Using Alternative Data

## Project Overview

This project develops an end-to-end Credit Risk Probability Model for Bati Bank using alternative transaction data from an eCommerce platform. The objective is to support a Buy-Now-Pay-Later (BNPL) service by estimating customer credit risk and identifying customers who may be eligible for credit products.

Since the dataset does not contain a direct loan default indicator, a proxy target variable is created using customer behavioral patterns based on Recency, Frequency, and Monetary (RFM) analysis.

---

## Business Problem

Bati Bank is partnering with an eCommerce platform to offer credit services to customers. Before granting credit, the bank needs a reliable method for assessing customer risk.

The project aims to:

* Identify high-risk and low-risk customers.
* Estimate the probability of credit risk.
* Generate a credit score from risk probabilities.
* Support data-driven lending decisions.
* Provide a deployable API for real-time credit risk prediction.

---

## Dataset Description

The dataset contains transaction-level information from an eCommerce platform.

### Key Features

| Feature              | Description                   |
| -------------------- | ----------------------------- |
| TransactionId        | Unique transaction identifier |
| CustomerId           | Unique customer identifier    |
| Amount               | Transaction amount            |
| ProductCategory      | Product category purchased    |
| ChannelId            | Transaction channel           |
| ProviderId           | Service provider              |
| TransactionStartTime | Transaction timestamp         |
| PricingStrategy      | Merchant pricing strategy     |
| FraudResult          | Fraud indicator               |

---

## Project Workflow

### 1. Business Understanding

* Reviewed credit scoring concepts and Basel II requirements.
* Analyzed the importance of model interpretability and documentation.
* Evaluated the challenges of building a model without a direct default label.

### 2. Exploratory Data Analysis (EDA)

Performed:

* Data structure inspection
* Summary statistics analysis
* Distribution analysis
* Correlation analysis
* Missing value assessment
* Outlier detection
* Customer activity analysis
* Transaction trend analysis

#### Key Findings

* Transaction amounts are highly skewed with several outliers.
* Customer transaction activity varies significantly.
* Transaction volumes show temporal patterns.
* Behavioral features appear useful for risk prediction.

### 3. Feature Engineering

Implemented:

* Customer-level aggregate features:

  * Total Transaction Amount
  * Average Transaction Amount
  * Transaction Count
  * Transaction Amount Standard Deviation

* Time-based features:

  * Transaction Hour
  * Transaction Day
  * Transaction Month
  * Transaction Year

* Missing value handling

* One-hot encoding of categorical variables

* Feature scaling using StandardScaler

### 4. Proxy Target Variable Engineering

Since no default label exists in the dataset:

* Calculated Recency, Frequency, and Monetary (RFM) metrics.
* Applied K-Means clustering to segment customers into behavioral groups.
* Identified the least-engaged customer segment.
* Created a binary target variable:

```text
is_high_risk = 1 → High-Risk Customer
is_high_risk = 0 → Low-Risk Customer
```

### 5. Model Training and Experiment Tracking

Trained and evaluated multiple machine learning models including:

* Logistic Regression
* Random Forest

Implemented:

* Train/Test Split
* Hyperparameter Tuning
* MLflow Experiment Tracking

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The best-performing model was registered and prepared for deployment.

### 6. Model Deployment

Developed a REST API using FastAPI.

Features:

* Loads the trained model
* Accepts customer feature inputs
* Returns risk probability predictions

Deployment tools:

* FastAPI
* Docker
* Docker Compose

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* MLflow
* FastAPI
* Docker
* Pytest
* GitHub Actions

---

## Conclusion

This project demonstrates the development of an end-to-end credit risk prediction system using alternative transaction data. By combining customer behavioral analytics, machine learning, experiment tracking, API deployment, and CI/CD practices, the solution provides a practical framework for supporting credit decision-making in environments where traditional credit history is unavailable.
