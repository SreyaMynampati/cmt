"""
predictive_modeling.py

Loads the synthetic dataset, trains RandomForestClassifier and a small MLP, evaluates performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pickle

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    cat_cols = ["gene_variant_id", "mutation_type"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[cat_cols])
    numeric = df[["age_of_onset", "motor_score", "sensory_score"]].values.astype(float)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(numeric)
    X = np.hstack([X_num, X_cat])
    label_map = {"mild":0, "moderate":1, "severe":2}
    y = df["disease_severity"].map(label_map).values
    return X, y, encoder, scaler, label_map, df

def train_and_evaluate(X, y, out_dir, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=random_state)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(out_dir / "mlp_model.pkl", "wb") as f:
        pickle.dump(mlp, f)
    with open(out_dir / "rf_report.txt", "w") as f:
        f.write(f"Accuracy: {acc_rf:.4f}\n\n{classification_report(y_test, y_pred_rf, target_names=['mild','moderate','severe'])}\n")
    with open(out_dir / "mlp_report.txt", "w") as f:
        f.write(f"Accuracy: {acc_mlp:.4f}\n\n{classification_report(y_test, y_pred_mlp, target_names=['mild','moderate','severe'])}\n")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].imshow(cm_rf)
    axes[0].set_title(f"RandomForest Confusion Matrix (acc={acc_rf:.3f})")
    axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true")
    axes[1].imshow(cm_mlp)
    axes[1].set_title(f"MLP Confusion Matrix (acc={acc_mlp:.3f})")
    axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrices.png")
    plt.close(fig)
    return {"rf_acc":acc_rf, "mlp_acc":acc_mlp}

if __name__ == "__main__":
    X, y, encoder, scaler, label_map, df = load_and_preprocess("cmt_synthetic_dataset.csv")
    results = train_and_evaluate(X, y, "model_outputs")
    print("Training complete. Results:", results)
