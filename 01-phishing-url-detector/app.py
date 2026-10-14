"""Minimal REST API for the phishing URL detector.

Run:
    python -m src.train          # once, to produce the model
    python app.py                # starts the server on :5000

Endpoints:
    GET  /health                 -> {"status": "ok", "model_loaded": bool}
    POST /predict  {"url": "..."}            -> single prediction
    POST /predict  {"urls": ["...", "..."]}  -> batch prediction
"""

from __future__ import annotations

from flask import Flask, jsonify, request

from src.predict import ModelNotTrained, classify

app = Flask(__name__)


@app.get("/health")
def health():
    try:
        classify("http://example.com")
        loaded = True
    except ModelNotTrained:
        loaded = False
    return jsonify(status="ok", model_loaded=loaded)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    urls = payload.get("urls")
    single = payload.get("url")

    if single is None and not urls:
        return jsonify(error="provide 'url' (string) or 'urls' (list)"), 400

    try:
        if urls:
            if not isinstance(urls, list):
                return jsonify(error="'urls' must be a list"), 400
            results = [_slim(classify(u)) for u in urls]
            return jsonify(results=results)
        return jsonify(_slim(classify(single)))
    except ModelNotTrained as exc:
        return jsonify(error=str(exc)), 503


def _slim(result: dict) -> dict:
    """Drop the verbose feature dump from API responses by default."""
    return {k: v for k, v in result.items() if k != "features"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
