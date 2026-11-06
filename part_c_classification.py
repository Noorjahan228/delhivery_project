"""
src/part_c_classification.py
Part C: Delay Classifier and Alert Threshold
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(__file__))
from shipment_prep import FEATURES, NOISE_COLUMNS


def get_clf_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return int(tn), int(fp), int(fn), int(tp), round(float(acc), 4), round(float(prec), 4), round(float(rec), 4), round(float(f1), 4)


def main():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # 1. Stratified split (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 2. Blind baseline and three classifiers
    models_records = []

    # Blind all-on-time baseline
    pred_blind = np.zeros(len(y_val), dtype=int)
    tn, fp, fn, tp, acc, _, rec, f1 = get_clf_metrics(y_val, pred_blind)
    models_records.append({
        "model": "blind_all_on_time", "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": 0.0, "recall": rec, "f1": f1
    })

    # Logistic Regression pipeline
    pipe_lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    pipe_lr.fit(X_train, y_train)
    pred_lr = pipe_lr.predict(X_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_clf_metrics(y_val, pred_lr)
    models_records.append({
        "model": "logistic_regression", "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })

    # Decision Tree depth tuning
    dt_candidates = []
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        pred_dt = dt.predict(X_val)
        val_f1 = f1_score(y_val, pred_dt, zero_division=0)
        dt_candidates.append((d, val_f1, pred_dt, dt))

    dt_candidates.sort(key=lambda x: (-x[1], x[0]))
    chosen_depth, _, best_dt_pred, _ = dt_candidates[0]
    tn, fp, fn, tp, acc, prec, rec, f1 = get_clf_metrics(y_val, best_dt_pred)
    models_records.append({
        "model": f"decision_tree_depth_{chosen_depth}", "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })

    # Default Random Forest
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_default.fit(X_train, y_train)
    pred_rf_def = rf_default.predict(X_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_clf_metrics(y_val, pred_rf_def)
    models_records.append({
        "model": "random_forest_default", "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })

    pd.DataFrame(models_records).to_csv("outputs/part_c_models.csv", index=False)

    # 3. Feature signals
    rf_feat = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_feat.fit(X_train, y_train)
    rf_imp = rf_feat.feature_importances_

    skb_all = SelectKBest(f_classif, k="all")
    skb_all.fit(X_train, y_train)
    f_scores = skb_all.scores_

    feature_scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": [round(float(s), 4) for s in f_scores],
        "rf_importance": [round(float(imp), 4) for imp in rf_imp],
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES]
    })
    feature_scores_df["f_rank"] = feature_scores_df["f_score"].rank(ascending=False, method="min").astype(int)
    feature_scores_df = feature_scores_df.sort_values(by="f_rank").reset_index(drop=True)
    feature_scores_df = feature_scores_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    feature_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

    k_records = []
    for k in [5, 7, 10, 12, 17]:
        pipe_k = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe_k.fit(X_train, y_train)
        pred_k = pipe_k.predict(X_val)
        val_f1 = f1_score(y_val, pred_k, zero_division=0)
        k_records.append({"k": k, "val_f1": round(float(val_f1), 4)})
    pd.DataFrame(k_records).to_csv("outputs/part_c_selectk.csv", index=False)

    # 4. Hyperparameter tuning on top 7 features
    selector_7 = SelectKBest(f_classif, k=7)
    X_train_7 = selector_7.fit_transform(X_train, y_train)
    X_val_7 = selector_7.transform(X_val)
    X_test_7 = selector_7.transform(X_test)

    rf_7_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_7_def.fit(X_train_7, y_train)
    rf_7_def_val_f1 = f1_score(y_val, rf_7_def.predict(X_val_7), zero_division=0)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }

    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        n_jobs=1
    )
    gs.fit(X_train_7, y_train)
    gs_best_model = gs.best_estimator_
    gs_val_f1 = f1_score(y_val, gs_best_model.predict(X_val_7), zero_division=0)

    rs = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=9,
        cv=5,
        random_state=42,
        n_jobs=1
    )
    rs.fit(X_train_7, y_train)
    rs_best_model = rs.best_estimator_
    rs_val_f1 = f1_score(y_val, rs_best_model.predict(X_val_7), zero_division=0)

    tuning_df = pd.DataFrame([
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(gs.best_params_),
            "best_cv_accuracy": round(float(gs.best_score_), 4),
            "val_f1": round(float(gs_val_f1), 4),
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rs.best_params_),
            "best_cv_accuracy": round(float(rs.best_score_), 4),
            "val_f1": round(float(rs_val_f1), 4),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": "",
            "val_f1": round(float(rf_7_def_val_f1), 4),
        },
    ])
    tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

    # 5. Choose final configuration
    candidates_tuning = [
        ("default_forest_same_7_features", rf_7_def_val_f1, rf_7_def, 100, None, 1),
        ("grid_search", gs_val_f1, gs_best_model, gs.best_params_["n_estimators"], gs.best_params_["max_depth"], gs.best_params_["min_samples_leaf"]),
        ("randomized_search", rs_val_f1, rs_best_model, rs.best_params_["n_estimators"], rs.best_params_["max_depth"], rs.best_params_["min_samples_leaf"]),
    ]
    priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
    candidates_tuning.sort(key=lambda x: (-x[1], priority[x[0]]))
    chosen_name, _, chosen_clf, chosen_n_est, chosen_depth_val, chosen_leaf = candidates_tuning[0]

    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", chosen_clf)
    ])
    final_pipeline.fit(X_train, y_train)

    # 6. Threshold sweep on validation set
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_records = []
    eligible_thresholds = []

    for t in threshold_list:
        t_clean = round(t, 1)
        pred_t = (val_probs >= t_clean).astype(int)
        tn, fp, fn, tp, acc, prec, rec, f1 = get_clf_metrics(y_val, pred_t)
        thresh_records.append({
            "threshold": t_clean, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        })
        if rec >= 0.80:
            eligible_thresholds.append((t_clean, rec))

    pd.DataFrame(thresh_records).to_csv("outputs/part_c_thresholds.csv", index=False)

    if eligible_thresholds:
        eligible_thresholds.sort(key=lambda x: x[0], reverse=True)
        chosen_threshold = eligible_thresholds[0][0]
    else:
        chosen_threshold = 0.1

    # 7. One test evaluation
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    test_pred_final = (test_probs >= chosen_threshold).astype(int)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_clf_metrics(y_test, test_pred_final)

    test_eval_df = pd.DataFrame([{
        "threshold": chosen_threshold,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    }])
    test_eval_df.to_csv("outputs/part_c_final_test.csv", index=False)

    final_config_df = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": chosen_n_est,
        "max_depth": "unlimited" if chosen_depth_val is None else chosen_depth_val,
        "min_samples_leaf": chosen_leaf,
        "threshold": chosen_threshold,
        "chosen_from": chosen_name
    }])
    final_config_df.to_csv("outputs/part_c_final_config.csv", index=False)

    print("SUCCESS: Part C complete!")


if __name__ == "__main__":
    main()