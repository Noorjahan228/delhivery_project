"""
src/part_d_segments.py
Part D: Hub Segmentation and Clustering Diagnostics
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist

def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data
    df = pd.read_csv("data/modelling_table.csv")

    # 2. Hub profile: Group by hub_id, 24 rows sorted ascending by hub_id
    profile = df.groupby("hub_id", observed=False).agg(
        shipments=("delivery_hours", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    ).reset_index().sort_values("hub_id").reset_index(drop=True)

    profile_to_save = profile.copy()
    profile_to_save["mean_delivery_hours"] = profile_to_save["mean_delivery_hours"].round(4)
    profile_to_save["delay_rate"] = profile_to_save["delay_rate"].round(4)
    profile_to_save["mean_distance_km"] = profile_to_save["mean_distance_km"].round(4)
    profile_to_save["mean_n_stops"] = profile_to_save["mean_n_stops"].round(4)
    profile_to_save.to_csv("outputs/hub_profile.csv", index=False)

    # 3. K-Means separation evidence for K in [2, 3, 4, 5]
    feature_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(profile[feature_cols])

    n_hubs = len(profile)
    separation_records = []
    kmeans_models = {}

    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_matrix)
        kmeans_models[k] = (km, labels)

        # Cluster sizes in descending order joined by hyphens
        counts = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
        cluster_sizes_str = "-".join(map(str, counts))

        # Across-to-within distance ratio
        # All 276 pairs
        within_dists = []
        across_dists = []
        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                d = np.linalg.norm(scaled_matrix[i] - scaled_matrix[j])
                if labels[i] == labels[j]:
                    within_dists.append(d)
                else:
                    across_dists.append(d)

        mean_within = np.mean(within_dists)
        mean_across = np.mean(across_dists)
        ratio = round(float(mean_across / mean_within), 3)

        separation_records.append({
            "K": k,
            "across_over_within": ratio,
            "cluster_sizes": cluster_sizes_str
        })

    pd.DataFrame(separation_records).to_csv("outputs/part_d_separation.csv", index=False)

    # 4. K = 3 fixed by business
    km3, labels3 = kmeans_models[3]
    profile["cluster_raw"] = labels3

    # Name clusters by mean delay_rate:
    # lowest -> Standard, middle -> Watch, highest -> Escalate
    cluster_means = profile.groupby("cluster_raw")["delay_rate"].mean().sort_values()
    name_map = {
        cluster_means.index[0]: "Standard",
        cluster_means.index[1]: "Watch",
        cluster_means.index[2]: "Escalate"
    }
    profile["segment"] = profile["cluster_raw"].map(name_map)

    # Load hubs info to merge hub_city and hub_tier
    hubs = pd.read_csv("data/hubs.csv")
    hub_seg = profile[["hub_id", "segment"]].merge(
        hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left"
    )
    hub_seg = hub_seg[["hub_id", "segment", "hub_city", "hub_tier"]]
    hub_seg.to_csv("outputs/hub_segments.csv", index=False)

    # Segment profile (outputs/segment_profile.csv)
    seg_prof = profile.groupby("segment").agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    ).loc[["Standard", "Watch", "Escalate"]].reset_index()

    seg_prof["mean_delivery_hours"] = seg_prof["mean_delivery_hours"].round(4)
    seg_prof["delay_rate"] = seg_prof["delay_rate"].round(4)
    seg_prof["mean_distance_km"] = seg_prof["mean_distance_km"].round(4)
    seg_prof["mean_n_stops"] = seg_prof["mean_n_stops"].round(4)
    seg_prof.to_csv("outputs/segment_profile.csv", index=False)

    # Tier vs Segment mismatch
    # Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    expected_seg = {1: "Standard", 2: "Watch", 3: "Escalate"}
    hub_seg["expected_segment"] = hub_seg["hub_tier"].map(expected_seg)
    mismatch = hub_seg[hub_seg["segment"] != hub_seg["expected_segment"]].drop(columns=["expected_segment"])
    mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

    print("SUCCESS: Part D complete!")
    print(f"Mismatches found: {len(mismatch)} hubs")

if __name__ == "__main__":
    main()