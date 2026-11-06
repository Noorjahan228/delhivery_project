import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(__file__))
from shipment_prep import FEATURES


def main():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_train_scaled, axis=0))))
    test_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_test_scaled, axis=0))))

    b_checks = pd.DataFrame([
        {"metric": "train_rows", "value": len(X_train)},
        {"metric": "val_rows", "value": len(X_val)},
        {"metric": "test_rows", "value": len(X_test)},
        {"metric": "train_scaled_max_abs_mean", "value": round(train_scaled_max_abs_mean, 10)},
        {"metric": "test_scaled_max_abs_mean", "value": round(test_scaled_max_abs_mean, 4)},
    ])
    b_checks.to_csv("outputs/part_b_checks.csv", index=False)

    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_records = []
    best_val_rmse = float("inf")
    best_depth = None

    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        pred_train = dt.predict(X_train)
        pred_val = dt.predict(X_val)

        tr_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
        v_rmse = np.sqrt(mean_squared_error(y_val, pred_val))

        label = "unlimited" if d is None else str(d)
        tree_records.append({
            "max_depth": label,
            "train_rmse": round(float(tr_rmse), 4),
            "val_rmse": round(float(v_rmse), 4)
        })

        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_depth = d

    pd.DataFrame(tree_records).to_csv("outputs/part_b_tree_depth.csv", index=False)

    y_train_mean = y_train.mean()
    base_pred_val = np.full_like(y_val, fill_value=y_train_mean)
    base_mae = mean_absolute_error(y_val, base_pred_val)
    base_mse = mean_squared_error(y_val, base_pred_val)
    base_rmse = np.sqrt(base_mse)
    base_r2 = r2_score(y_val, base_pred_val)

    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    lr_val_pred = lr_pipe.predict(X_val)
    lr_mae = mean_absolute_error(y_val, lr_val_pred)
    lr_mse = mean_squared_error(y_val, lr_val_pred)
    lr_rmse = np.sqrt(lr_mse)
    lr_r2 = r2_score(y_val, lr_val_pred)

    best_dt = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    best_dt.fit(X_train, y_train)
    dt_val_pred = best_dt.predict(X_val)
    dt_mae = mean_absolute_error(y_val, dt_val_pred)
    dt_mse = mean_squared_error(y_val, dt_val_pred)
    dt_rmse = np.sqrt(dt_mse)
    dt_r2 = r2_score(y_val, dt_val_pred)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    rf_mae = mean_absolute_error(y_val, rf_val_pred)
    rf_mse = mean_squared_error(y_val, rf_val_pred)
    rf_rmse = np.sqrt(rf_mse)
    rf_r2 = r2_score(y_val, rf_val_pred)

    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_val_pred = gb.predict(X_val)
    gb_mae = mean_absolute_error(y_val, gb_val_pred)
    gb_mse = mean_squared_error(y_val, gb_val_pred)
    gb_rmse = np.sqrt(gb_mse)
    gb_r2 = r2_score(y_val, gb_val_pred)

    val_metrics = [
        {"model": "mean_baseline", "split": "val", "MAE": round(base_mae, 4), "MSE": round(base_mse, 4), "RMSE": round(base_rmse, 4), "R2": round(base_r2, 4)},
        {"model": "linear_regression", "split": "val", "MAE": round(lr_mae, 4), "MSE": round(lr_mse, 4), "RMSE": round(lr_rmse, 4), "R2": round(lr_r2, 4)},
        {"model": "decision_tree", "split": "val", "MAE": round(dt_mae, 4), "MSE": round(dt_mse, 4), "RMSE": round(dt_rmse, 4), "R2": round(dt_r2, 4)},
        {"model": "random_forest", "split": "val", "MAE": round(rf_mae, 4), "MSE": round(rf_mse, 4), "RMSE": round(rf_rmse, 4), "R2": round(rf_r2, 4)},
        {"model": "gradient_boosting", "split": "val", "MAE": round(gb_mae, 4), "MSE": round(gb_mse, 4), "RMSE": round(gb_rmse, 4), "R2": round(gb_r2, 4)},
    ]

    lr_rates = [0.01, 0.1, 1.0]
    gb_lr_records = []
    for rate in lr_rates:
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=rate, random_state=42)
        gbr.fit(X_train, y_train)
        pred_v = gbr.predict(X_val)
        v_rmse = np.sqrt(mean_squared_error(y_val, pred_v))
        v_r2 = r2_score(y_val, pred_v)
        gb_lr_records.append({
            "learning_rate": rate,
            "val_rmse": round(float(v_rmse), 4),
            "val_r2": round(float(v_r2), 4)
        })
    pd.DataFrame(gb_lr_records).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv_scores = cross_val_score(rf, X_train_val, y_train_val, cv=cv, scoring="r2")
    gb_cv_scores = cross_val_score(gb, X_train_val, y_train_val, cv=cv, scoring="r2")

    cv_df = pd.DataFrame([
        {"model": "random_forest", "mean_r2": round(float(rf_cv_scores.mean()), 4), "std_r2": round(float(rf_cv_scores.std()), 4)},
        {"model": "gradient_boosting", "mean_r2": round(float(gb_cv_scores.mean()), 4), "std_r2": round(float(gb_cv_scores.std()), 4)},
    ])
    cv_df.to_csv("outputs/part_b_cv.csv", index=False)

    candidates = [
        ("linear_regression", lr_rmse, lr_pipe),
        ("decision_tree", dt_rmse, best_dt),
        ("random_forest", rf_rmse, rf),
        ("gradient_boosting", gb_rmse, gb),
    ]
    candidates.sort(key=lambda x: x[1])
    winner_name, _, winner_model = candidates[0]

    test_pred = winner_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_pred)

    val_metrics.append({
        "model": winner_name,
        "split": "test",
        "MAE": round(test_mae, 4),
        "MSE": round(test_mse, 4),
        "RMSE": round(test_rmse, 4),
        "R2": round(test_r2, 4)
    })
    pd.DataFrame(val_metrics).to_csv("outputs/part_b_metrics.csv", index=False)

    print("SUCCESS: Part B complete!")


if __name__ == "__main__":
    main()