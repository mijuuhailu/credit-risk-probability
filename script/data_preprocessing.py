import pandas as pd
import numpy as np

from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)

class AggregateFeatures(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        customer_features = (
            X.groupby("CustomerId")
            .agg(
                TotalAmount=("Amount","sum"),
                AvgAmount=("Amount","mean"),
                TxCount=("Amount","count"),
                StdAmount=("Amount","std")
            )
            .reset_index()
        )

        return customer_features
    

class TimeFeatures(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["TransactionStartTime"] = pd.to_datetime(
            X["TransactionStartTime"]
        )

        X["Hour"] = (
            X["TransactionStartTime"]
            .dt.hour
        )

        X["Day"] = (
            X["TransactionStartTime"]
            .dt.day
        )

        X["Month"] = (
            X["TransactionStartTime"]
            .dt.month
        )

        X["Year"] = (
            X["TransactionStartTime"]
            .dt.year
        )

        return X
    

numeric_features = [
    "TotalAmount",
    "AvgAmount",
    "TxCount",
    "StdAmount"
]



categorical_features = [
    "CurrencyCode",
    "ProviderId",
    "ProductCategory",
    "ChannelId"
]


from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

num_pipeline = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


cat_pipeline = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    [
        (
            "num",
            num_pipeline,
            numeric_features
        ),

        (
            "cat",
            cat_pipeline,
            categorical_features
        )
    ]
)