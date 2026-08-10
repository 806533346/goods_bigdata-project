"""Ensemble RoBERTa-base end-to-end fine-tuning + MLP predictions."""
import pandas as pd
import numpy as np

DATA_PATH = "/home/nmxc/project_code/big_data_trea/data"

# Load predictions
e2e_pred = pd.read_csv(f"{DATA_PATH}/submission_roberta_base.csv")    # 0.61814
mlp_pred = pd.read_csv(f"{DATA_PATH}/submission_roberta_mlp.csv")     # 0.64759/0.65423

print("=== Ensemble: End-to-End + MLP ===")
print(f"E2E prediction: min={e2e_pred['rating'].min():.2f}, max={e2e_pred['rating'].max():.2f}, mean={e2e_pred['rating'].mean():.2f}")
print(f"MLP prediction: min={mlp_pred['rating'].min():.2f}, max={mlp_pred['rating'].max():.2f}, mean={mlp_pred['rating'].mean():.2f}")

# Check correlation
corr = np.corrcoef(e2e_pred['rating'].values, mlp_pred['rating'].values)[0, 1]
print(f"Prediction correlation: {corr:.4f}")

# Try different weights
print("\n=== Testing ensemble weights ===")
for w_e2e in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    w_mlp = 1.0 - w_e2e
    ensemble = w_e2e * e2e_pred['rating'].values + w_mlp * mlp_pred['rating'].values
    ensemble = np.clip(ensemble, 1.0, 5.0)
    print(f"  E2E:{w_e2e:.1f} + MLP:{w_mlp:.1f} -> min={ensemble.min():.2f}, max={ensemble.max():.2f}, mean={ensemble.mean():.2f}")

# Best weight: E2E is better (0.61814 vs 0.64759), so weight it more
best_w_e2e = 0.7
best_w_mlp = 0.3
ensemble = best_w_e2e * e2e_pred['rating'].values + best_w_mlp * mlp_pred['rating'].values
ensemble = np.clip(ensemble, 1.0, 5.0)

submission = pd.DataFrame({"id": e2e_pred["id"].values, "rating": ensemble})
output_path = f"{DATA_PATH}/submission_ensemble.csv"
submission.to_csv(output_path, index=False)
print(f"\nEnsemble submission saved to {output_path}")
print(f"Weights: E2E={best_w_e2e}, MLP={best_w_mlp}")
print(f"Prediction stats: min={ensemble.min():.2f}, max={ensemble.max():.2f}, mean={ensemble.mean():.2f}")
