import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import config as cfg
from data_loader import load_dataset, get_feature_lists, build_preprocessor
from pso import PSO

def main():
    cfg.MODEL_DIR.mkdir(exist_ok=True)
    cfg.STATIC_DIR.mkdir(exist_ok=True)

    train_df, test_df = load_dataset(cfg.TRAIN_FILE), load_dataset(cfg.TEST_FILE)
    print("Train shape:", train_df.shape, "| Test shape:", test_df.shape)
    feat, num = get_feature_lists(train_df)

    pre = build_preprocessor(num)
    X_full = pre.fit_transform(train_df[feat]); y_full = train_df["y"].values
    X_test = pre.transform(test_df[feat]); y_test = test_df["y"].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.3, stratify=y_full, random_state=42)
    idx = np.random.RandomState(1).choice(X_tr.shape[0], cfg.FITNESS_SUBSET, replace=False)
    X_sub, y_sub = X_tr[idx], y_tr[idx]

    def fitness(p):
        model = MLPClassifier(hidden_layer_sizes=(int(p[0]),), learning_rate_init=10 ** p[1],
                              alpha=10 ** p[2], max_iter=cfg.FITNESS_MAX_ITER, random_state=0)
        model.fit(X_sub, y_sub)
        return accuracy_score(y_val, model.predict(X_val))

    pso = PSO(fitness, cfg.LOWER_BOUNDS, cfg.UPPER_BOUNDS, cfg.PSO_PARTICLES,
            cfg.PSO_ITERATIONS, cfg.PSO_W, cfg.PSO_C1, cfg.PSO_C2)
    best, best_fit, history = pso.run()
    neurons, lr, alpha = int(best[0]), float(10 ** best[1]), float(10 ** best[2])
    print(f"Best parameters -> neurons={neurons}, lr={lr:.5f}, alpha={alpha:.6f}")

    final = MLPClassifier(hidden_layer_sizes=(neurons,), learning_rate_init=lr, alpha=alpha,
                        max_iter=cfg.FINAL_MAX_ITER, random_state=0).fit(X_full, y_full)
    baseline = MLPClassifier(max_iter=cfg.FINAL_MAX_ITER, random_state=0).fit(X_full, y_full)

    pred = final.predict(X_test)
    acc_base = accuracy_score(y_test, baseline.predict(X_test))
    acc_pso = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred).tolist()
    print(f"Baseline test accuracy: {acc_base:.4f} | PSO-NN test accuracy: {acc_pso:.4f}")
    print(classification_report(y_test, pred, target_names=["Normal", "Attack"]))
    print("Confusion matrix:", cm)

    plt.figure(figsize=(5, 3.5))
    plt.plot(history, marker="o")
    plt.xlabel("Iteration"); plt.ylabel("Best validation accuracy")
    plt.title("PSO convergence"); plt.tight_layout()
    plt.savefig(cfg.PLOT_PATH, dpi=130)

    defaults = {c: train_df[c].mode()[0] for c in cfg.CATEGORICAL}
    defaults.update({c: float(train_df[c].median()) for c in num})
    artifacts = {
        "pre": pre, "clf": final, "feat": feat, "defaults": defaults,
        "choices": {c: sorted(train_df[c].unique().tolist()) for c in cfg.CATEGORICAL},
        "samples": {
            "normal": test_df[test_df.y == 0].sample(30, random_state=1)[feat].to_dict("records"),
            "attack": test_df[test_df.y == 1].sample(30, random_state=1)[feat].to_dict("records"),
        }}
    joblib.dump(artifacts, cfg.MODEL_PATH)
    with open(cfg.METRICS_PATH, "w") as f:
        json.dump({"neurons": neurons, "lr": lr, "alpha": alpha, "acc_base": acc_base,
                "acc_pso": acc_pso, "cm": cm, "best_val": best_fit}, f)
    print("Saved model and metrics.")

if __name__ == "__main__":
    main()