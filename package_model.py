import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(__file__))
from shipment_prep import clean_shipments, FEATURES

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    k_val = int(config["selector_k"])
    n_est = int(config["n_estimators"])
    m_depth = None if str(config["max_depth"]).lower() == "unlimited" else int(config["max_depth"])
    min_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    clf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=m_depth,
        min_samples_leaf=min_leaf,
        random_state=42
    )
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=k_val)),
        ("model", clf)
    ])
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "models/delay_model.pkl")
    pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)
    impute_df = pd.DataFrame([
        {"column": col, "median": round(medians[col], 4)} for col in medians
    ])
    impute_df.to_csv("models/impute_values.csv", index=False)

    reloaded = joblib.load("models/delay_model.pkl")
    in_mem_probs = pipeline.predict_proba(X_test)[:, 1]
    reloaded_probs = reloaded.predict_proba(X_test)[:, 1]
    max_diff = float(np.max(np.abs(in_mem_probs - reloaded_probs)))

    reload_check = pd.DataFrame([
        {"metric": "test_rows_scored", "value": len(X_test)},
        {"metric": "reload_max_abs_probability_diff", "value": max_diff},
        {"metric": "threshold", "value": threshold}
    ])
    reload_check.to_csv("outputs/part_d_reload_check.csv", index=False)

    print("SUCCESS: Model packaged successfully!")

if __name__ == "__main__":
    main()