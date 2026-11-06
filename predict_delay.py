"""
src/predict_delay.py
Task 6: Scoring script for new incoming shipments
"""
import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from shipment_prep import clean_shipments, build_features

def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load incoming raw new shipments and metadata
    raw_new = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    model = joblib.load("models/delay_model.pkl")
    thresh_df = pd.read_csv("models/delay_threshold.csv")
    threshold = float(thresh_df["threshold"].iloc[0])
    impute_df = pd.read_csv("models/impute_values.csv").set_index("column")["median"].to_dict()
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    hub_segments = pd.read_csv("outputs/hub_segments.csv")

    # 2. Clean using stored medians
    clean_new, _ = clean_shipments(raw_new, medians=impute_df)

    # 3. Build features
    features_new = build_features(clean_new, hubs)

    # 4. Predict probabilities using ordered features
    X_score = features_new[feature_order]
    probs = model.predict_proba(X_score)[:, 1]

    # 5. Build output dataframe
    res = pd.DataFrame()
    res["shipment_id"] = clean_new["shipment_id"]
    res["hub_id"] = clean_new["hub_id"]

    # Join hub segment
    res = res.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")
    res["delay_probability"] = [round(float(p), 4) for p in probs]
    res["flagged"] = (res["delay_probability"] >= threshold).astype(int)

    # Action rule
    def assign_action(row):
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                return "Escalate to hub manager"
            else:
                return "Proactive customer call"
        else:
            return "No action"

    res["action"] = res.apply(assign_action, axis=1)

    # Save to outputs/new_shipment_actions.csv
    cols = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
    res[cols].to_csv("outputs/new_shipment_actions.csv", index=False)

    print("SUCCESS: Predictions generated!")

if __name__ == "__main__":
    main()