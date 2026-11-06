import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

def clean_shipments(df, medians=None):
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    
    clean_val = (
        df["declared_value"]
        .astype(str)
        .str.replace("\u20b9", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["declared_value_inr"] = pd.to_numeric(clean_val, errors="coerce")
    df = df.drop(columns=["declared_value"])
    
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    median_cols = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    if medians is None:
        computed_medians = {col: float(df[col].median()) for col in median_cols}
    else:
        computed_medians = medians

    for col in median_cols:
        df[col] = df[col].fillna(computed_medians[col])

    return df, computed_medians

def build_features(clean, hubs):
    df = clean.copy()
    df = df.merge(
        hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
        on="hub_id",
        how="left",
    )
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df