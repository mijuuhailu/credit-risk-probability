# Credit Scoring Business Understanding

## 1. Basel II and the Need for Interpretable Models

The Basel II Accord emphasizes rigorous measurement, documentation, and monitoring of credit risk. Financial institutions must be able to justify how credit decisions are made and demonstrate that their models are reliable, transparent, and consistent. As a result, credit risk models should be interpretable enough for risk managers, auditors, and regulators to understand the factors influencing predictions.

An interpretable model enables the bank to explain why a customer received a particular risk score, supports regulatory compliance, facilitates model validation, and helps identify potential sources of bias or instability. Comprehensive documentation of data sources, feature engineering steps, model assumptions, and performance metrics is therefore an essential component of the credit scoring process.

## 2. Necessity and Risks of a Proxy Default Variable

The available eCommerce transaction dataset does not contain a direct measure of loan default or repayment behavior. Since supervised machine learning models require a target variable, a proxy variable must be constructed to represent credit risk.

This project proposes using customer behavioral indicators derived from Recency, Frequency, and Monetary (RFM) analysis. Customers exhibiting low transaction frequency, low monetary value, and long periods of inactivity may be classified as higher-risk, while active and engaged customers may be classified as lower-risk.

However, proxy-based targets introduce several risks. The proxy may not perfectly reflect actual default behavior, leading to label noise and prediction errors. Some customers classified as high-risk may ultimately repay loans successfully, while some low-risk customers may default. Additionally, proxy definitions may introduce unintended biases if behavioral patterns differ across customer groups. These limitations should be acknowledged and monitored throughout model development and deployment.

## 3. Trade-offs Between Interpretable and High-Performance Models

A Logistic Regression model combined with Weight of Evidence (WoE) encoding offers high interpretability and transparency. The contribution of each predictor can be easily understood, making the model suitable for regulatory review and scorecard development. However, its predictive performance may be limited when relationships between variables are highly nonlinear.

In contrast, Gradient Boosting models such as XGBoost or LightGBM often achieve superior predictive accuracy by capturing complex interactions and nonlinear patterns in the data. The trade-off is reduced interpretability, making it more difficult to explain individual predictions and satisfy regulatory expectations.

In a regulated financial environment, model selection requires balancing predictive performance with transparency, governance, explainability, and compliance requirements. A common industry practice is to benchmark advanced machine learning models against interpretable baseline models and use explainability techniques such as feature importance analysis or SHAP values when deploying more complex models.
