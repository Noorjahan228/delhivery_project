import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from shipment_prep import clean_shipments, build_features, FEATURES

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    raw_rows = len(raw)
    n_dupes_removed = raw.duplicated().sum()

    dedup = raw.drop_duplicates().reset_index(drop=True)
    missing_dist_before = dedup["distance_km"].isna().sum()
    sentinel_weight_rows = (dedup["parcel_weight_kg"] > 100).sum()
    missing_weight_before = dedup["parcel_weight_kg"].isna().sum()
    missing_value_before = dedup["declared_value"].isna().sum()
    missing_pay_before = dedup["payment_mode"].isna().sum()
    service_spellings_before = dedup["service_type"].nunique()

    clean_df, medians = clean_shipments(raw)
    service_spellings_after = clean_df["service_type"].nunique()
    rows_after_dedup = len(clean_df)

    model_df = build_features(clean_df, hubs)
    rows_after_hub_merge = len(model_df)

    model_df["is_delayed"] = (
        model_df["delivery_hours"] > model_df["promised_hours"]
    ).astype(int)

    model_df.to_csv("data/modelling_table.csv", index=False)

    nulls_in_features_and_target = (
        model_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
    )
    delayed_count = int(model_df["is_delayed"].sum())
    delay_rate = round(float(model_df["is_delayed"].mean()), 4)

    decl_mean = round(float(model_df["declared_value_inr"].mean()), 4)
    decl_median = round(float(model_df["declared_value_inr"].median()), 4)
    log_decl_mean = round(float(model_df["log_declared_value"].mean()), 4)
    log_decl_median = round(float(model_df["log_declared_value"].median()), 4)

    q1 = float(np.percentile(model_df["delivery_hours"], 25))
    q3 = float(np.percentile(model_df["delivery_hours"], 75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(
        (
            (model_df["delivery_hours"] < lower_bound)
            | (model_df["delivery_hours"] > upper_bound)
        ).sum()
    )

    checks = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", n_dupes_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_dist_before),
        ("missing_parcel_weight_before", missing_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_value_before),
        ("missing_payment_mode_before", missing_pay_before),
        ("service_type_spellings_before", service_spellings_before),
        ("service_type_spellings_after", service_spellings_after),
        ("declared_value_mean_inr", round(decl_mean, 2)),
        ("log_declared_value_mean", round(log_decl_mean, 4)),
        ("median_distance_km", round(medians["distance_km"], 2)),
        ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 2)),
        ("median_declared_value_inr", round(medians["declared_value_inr"], 2)),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", delay_rate),
        ("hours_q1", round(q1, 2)),
        ("hours_q3", round(q3, 2)),
        ("hours_lower_bound", round(lower_bound, 2)),
        ("hours_upper_bound", round(upper_bound, 2)),
        ("hours_outlier_count", outlier_count),
        ("declared_value_median_inr", round(decl_median, 2)),
        ("log_declared_value_median", round(log_decl_median, 4)),
    ]

    checks_df = pd.DataFrame(checks, columns=["metric", "value"])
    checks_df.to_csv("outputs/part_a_checks.csv", index=False)

    bins = [0, 200, 400, 600, 2000]
    labels = ["0-200", "201-400", "401-600", "601+"]
    model_df["distance_band"] = pd.cut(
        model_df["distance_km"], bins=bins, labels=labels, right=True
    )

    records = []
    for dim, col in [
        ("service_type", "service_type"),
        ("hub_tier", "hub_tier"),
        ("distance_band", "distance_band"),
    ]:
        for grp, sub in model_df.groupby(col, observed=False):
            records.append(
                {
                    "dimension": dim,
                    "group": str(grp),
                    "shipments": len(sub),
                    "delayed": int(sub["is_delayed"].sum()),
                    "delay_rate": round(float(sub["is_delayed"].mean()), 4),
                    "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
                }
            )

    delay_group_df = pd.DataFrame(records)
    delay_group_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)

    num_cols = model_df.select_dtypes(include=[np.number]).columns
    corr_df = model_df[num_cols].corr().round(4)
    corr_df.to_csv("outputs/part_a_correlations.csv", index=True)

    print("SUCCESS: Part A run completed!")

if __name__ == "__main__":
    main()