import json, random
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
import config as cfg

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
        prob = float(clf.predict_proba(pre.transform(pd.DataFrame([row])[feat]))[0][1])
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"label": "Attack" if prob >= 0.5 else "Normal",
                    "attack_probability": round(prob * 100, 2)})

if __name__ == "__main__":
    app.run(debug=False)