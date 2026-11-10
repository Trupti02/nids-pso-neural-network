import json, random
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
import config as cfg
import svm_model  # needed so the saved model can be loaded

app = Flask(__name__)

artifacts = joblib.load(cfg.MODEL_PATH)
with open(cfg.METRICS_PATH) as f:
    metrics = json.load(f)

pre, clf, feat = artifacts["pre"], artifacts["clf"], artifacts["feat"]
defaults, choices, samples = artifacts["defaults"], artifacts["choices"], artifacts["samples"]

@app.route("/")
def home():
    return render_template("index.html", choices=choices, defaults=defaults,
                            m=metrics, nums=cfg.FORM_NUMERIC_FIELDS)

@app.route("/sample/<kind>")
def sample(kind):
    if kind not in samples:
        return jsonify({"error": "invalid kind"}), 400
    return jsonify(random.choice(samples[kind]))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    row = dict(defaults)
    try:
        for k in feat:
            if k in data and data[k] != "":
                row[k] = data[k] if k in choices else float(data[k])
        X = pre.transform(pd.DataFrame([row])[feat])
        if hasattr(X, "toarray"):
            X = X.toarray()
        score = float(clf.predict_proba(X)[0][1])
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"label": "Attack" if score >= 0.5 else "Normal",
                    "attack_probability": round(score * 100, 2)})

if __name__ == "__main__":
    app.run(debug=False)