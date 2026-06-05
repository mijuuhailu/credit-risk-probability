# src/data_processing.py

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def create_customer_aggregates(df):
    """
    Create customer-level aggregate features.
    """

    customer_features = (
        df.groupby("CustomerId")
        .agg(
            TotalTransactionAmount=("Amount", "sum"),
            AverageTransactionAmount=("Amount", "mean"),
            TransactionCount=("TransactionId", "count"),
            StdTransactionAmount=("Amount", "std"),
        )
        .reset_index()
    )

    customer_features["StdTransactionAmount"] = (
        customer_features["StdTransactionAmount"].fillna(0)
    )

    return customer_features


def create_time_features(df):
    """
    Extract time-based features.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["TransactionHour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["TransactionDay"] = (
        df["TransactionStartTime"].dt.day
    )

    df["TransactionMonth"] = (
        df["TransactionStartTime"].dt.month
    )

    df["TransactionYear"] = (
        df["TransactionStartTime"].dt.year
    )

    return df


# =====================================================
# RFM TARGET ENGINEERING
# =====================================================

def calculate_rfm(df):
    """
    Calculate Recency, Frequency, Monetary metrics.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (snapshot_date - x.max()).days,
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    return rfm


def create_proxy_target(rfm):
    """
    Create high-risk proxy target using KMeans clustering.
    """

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )

    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = (
        rfm.groupby("Cluster")
        [["Recency", "Frequency", "Monetary"]]
        .mean()
    )

    print("\nCluster Summary")
    print(cluster_summary)

    # High Recency + Low Frequency + Low Monetary
    high_risk_cluster = (
        cluster_summary["Recency"]
        .idxmax()
    )

    rfm["is_high_risk"] = (
        rfm["Cluster"] == high_risk_cluster
    ).astype(int)

    return rfm


# =====================================================
# BUILD CUSTOMER-LEVEL DATASET
# =====================================================

def build_customer_dataset(df):
    """
    Combine engineered features and target variable.
    """

    df = create_time_features(df)

    customer_features = create_customer_aggregates(df)

    rfm = calculate_rfm(df)

    rfm = create_proxy_target(rfm)

    # Use first categorical observation per customer
    customer_cats = (
        df.groupby("CustomerId")
        .agg(
            CurrencyCode=("CurrencyCode", "first"),
            ProviderId=("ProviderId", "first"),
            ProductCategory=("ProductCategory", "first"),
            ChannelId=("ChannelId", "first"),
            CountryCode=("CountryCode", "first"),
            PricingStrategy=("PricingStrategy", "first"),
            TransactionHour=("TransactionHour", "mean"),
            TransactionDay=("TransactionDay", "mean"),
            TransactionMonth=("TransactionMonth", "mean"),
            TransactionYear=("TransactionYear", "mean"),
        )
        .reset_index()
    )

    dataset = (
        customer_features
        .merge(rfm, on="CustomerId")
        .merge(customer_cats, on="CustomerId")
    )

    return dataset


# =====================================================
# PREPROCESSING PIPELINE
# =====================================================

def build_pipeline(df):
    """
    Returns:
        processed_df
        fitted_pipeline
    """

    dataset = build_customer_dataset(df)

    numeric_features = [
        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "TransactionCount",
        "StdTransactionAmount",
        "Recency",
        "Frequency",
        "Monetary",
        "CountryCode",
        "PricingStrategy",
        "TransactionHour",
        "TransactionDay",
        "TransactionMonth",
        "TransactionYear",
    ]

    categorical_features = [
        "CurrencyCode",
        "ProviderId",
        "ProductCategory",
        "ChannelId",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    X = dataset.drop(
        columns=[
            "CustomerId",
            "Cluster",
            "is_high_risk",
        ]
    )

    y = dataset["is_high_risk"]

    transformed = preprocessor.fit_transform(X)

    cat_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(
            categorical_features
        )
    )

    feature_names = (
        numeric_features
        + list(cat_names)
    )

    processed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
    )

    processed_df["is_high_risk"] = y.values

    return processed_df, preprocessor


# =====================================================
# EXAMPLE USAGE
# =====================================================

if __name__ == "__main__":

    df = pd.read_csv("../data/data.csv")

    processed_df, pipeline = build_pipeline(df)

    # Save processed dataset
    processed_df.to_csv(
        "../data/processed_data.csv",
        index=False
    )

    print("Processed dataset saved!")

    print(processed_df.head())

    print(
        processed_df["is_high_risk"]
        .value_counts(normalize=True)
    )