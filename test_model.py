import joblib, pandas as pd
import config as cfg
import svm_model
from data_loader import load_dataset

art = joblib.load(cfg.MODEL_PATH)
pre, clf, feat = art["pre"], art["clf"], art["feat"]

df = load_dataset(cfg.TEST_FILE).sample(20, random_state=7)
X = pre.transform(df[feat])
if hasattr(X, "toarray"):
    X = X.toarray()

score = clf.predict_proba(X)[:, 1]
out = pd.DataFrame({"actual": df["y"].map({0: "Normal", 1: "Attack"}).values,
                    "predicted": ["Attack" if s >= 0.5 else "Normal" for s in score],
                    "attack_score_%": (score * 100).round(1)})
out["correct"] = out["actual"] == out["predicted"]
print(out.to_string(index=False))
print("Correct:", out["correct"].sum(), "/ 20")