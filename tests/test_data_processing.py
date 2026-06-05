from src.data_processing import (
    create_customer_aggregates
)

import pandas as pd

def test_aggregate_columns():

    sample = pd.DataFrame(
        {
            "CustomerId":[1,1,2],
            "Amount":[10,20,30],
            "TransactionId":[1,2,3]
        }
    )

    result = create_customer_aggregates(
        sample
    )

    expected_cols = [
        "CustomerId",
        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "TransactionCount",
        "StdTransactionAmount"
    ]

    for col in expected_cols:
        assert col in result.columns


from src.data_processing import (
    calculate_rfm
)

def test_rfm_creation():

    sample = pd.DataFrame(
        {
            "CustomerId":[1,1],
            "TransactionId":[1,2],
            "Amount":[100,200],
            "TransactionStartTime":[
                "2024-01-01",
                "2024-01-02"
            ]
        }
    )

    result = calculate_rfm(
        sample
    )

    assert "Recency" in result.columns
    assert "Frequency" in result.columns
    assert "Monetary" in result.columns