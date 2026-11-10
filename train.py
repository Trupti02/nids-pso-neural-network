import json, time
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import config as cfg
from data_loader import load_dataset, get_feature_lists, build_preprocessor
from pso import PSO
from svm_model import SVMClassifier

def dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def main():
    cfg.MODEL_DIR.mkdir(exist_ok=True)
    cfg.STATIC_DIR.mkdir(exist_ok=True)

    train_df, test_df = load_dataset(cfg.TRAIN_FILE), load_dataset(cfg.TEST_FILE)
    print("Train shape:", train_df.shape, "| Test shape:", test_df.shape)
    feat, num = get_feature_lists(train_df)

    pre = build_preprocessor(num)
    X_full = dense(pre.fit_transform(train_df[feat])); y_full = train_df["y"].values
    X_test = dense(pre.transform(test_df[feat])); y_test = test_df["y"].values
    print("Features after encoding:", X_full.shape[1])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.3, stratify=y_full, random_state=42)
    rng = np.random.RandomState(1)
    i_tr = rng.choice(len(X_tr), cfg.SVM_FIT_TRAIN, replace=False)
    i_val = rng.choice(len(X_val), cfg.SVM_FIT_VAL, replace=False)
    X_sub, y_sub = X_tr[i_tr], y_tr[i_tr]
    X_vs, y_vs = X_val[i_val], y_val[i_val]

    def fitness(p):
        svc = SVC(kernel="rbf", C=10 ** p[0], gamma=10 ** p[1], cache_size=1000)
        svc.fit(X_sub, y_sub)
        return accuracy_score(y_vs, svc.predict(X_vs))

    start = time.time()
    pso = PSO(fitness, cfg.SVM_LOWER, cfg.SVM_UPPER, cfg.PSO_PARTICLES,
                cfg.PSO_ITERATIONS, cfg.PSO_W, cfg.PSO_C1, cfg.PSO_C2)
    best, best_fit, history = pso.run()
    C, gamma = float(10 ** best[0]), float(10 ** best[1])
    print(f"Best parameters -> C={C:.4f}, gamma={gamma:.6f} (PSO took {time.time() - start:.0f}s)")

    n_final = min(cfg.SVM_FINAL_SAMPLES, len(X_full))
    sel = np.random.RandomState(2).choice(len(X_full), n_final, replace=False)
    X_fin, y_fin = X_full[sel], y_full[sel]

    print(f"Training final SVM on {n_final} rows...")
    final_svc = SVC(kernel="rbf", C=C, gamma=gamma, cache_size=1000).fit(X_fin, y_fin)
    print("Training default SVM, Random Forest, Logistic Regression...")
    default_svc = SVC(kernel="rbf", cache_size=1000).fit(X_fin, y_fin)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0).fit(X_full, y_full)
    lr = LogisticRegression(max_iter=500).fit(X_full, y_full)

    pred = final_svc.predict(X_test)
    results = {
        "Default SVM": accuracy_score(y_test, default_svc.predict(X_test)),
        "Logistic Regression": accuracy_score(y_test, lr.predict(X_test)),
        "Random Forest": accuracy_score(y_test, rf.predict(X_test)),
        "PSO-optimized SVM": accuracy_score(y_test, pred),
    }
    for name, acc in results.items():
        print(f"{name:22s} test accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, target_names=["Normal", "Attack"]))
    cm = confusion_matrix(y_test, pred).tolist()
    print("Confusion matrix:", cm)

    plt.figure(figsize=(5, 3.5))
    plt.plot(history, marker="o")
    plt.xlabel("Iteration"); plt.ylabel("Best validation accuracy")
    plt.title("PSO convergence"); plt.tight_layout()
    plt.savefig(cfg.PLOT_PATH, dpi=130)

    defaults = {c: train_df[c].mode()[0] for c in cfg.CATEGORICAL}
    defaults.update({c: float(train_df[c].median()) for c in num})
    artifacts = {
        "pre": pre, "clf": SVMClassifier(final_svc), "feat": feat, "defaults": defaults,
        "choices": {c: sorted(train_df[c].unique().tolist()) for c in cfg.CATEGORICAL},
        "samples": {
            "normal": test_df[test_df.y == 0].sample(30, random_state=1)[feat].to_dict("records"),
            "attack": test_df[test_df.y == 1].sample(30, random_state=1)[feat].to_dict("records"),
        }}
    joblib.dump(artifacts, cfg.MODEL_PATH)
    with open(cfg.METRICS_PATH, "w") as f:
        json.dump({"C": C, "gamma": gamma,
                    "acc_base": float(results["Default SVM"]),
                    "acc_lr": float(results["Logistic Regression"]),
                    "acc_rf": float(results["Random Forest"]),
                    "acc_pso": float(results["PSO-optimized SVM"]),
                    "cm": cm, "best_val": best_fit}, f)
    print("Saved model and metrics.")

if __name__ == "__main__":
    main()