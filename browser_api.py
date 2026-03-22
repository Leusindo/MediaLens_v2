from flask import Flask, request, jsonify
from flask_cors import CORS

from core.classifier import NewsClassifier

app = Flask(__name__)
CORS(app)

classifier = NewsClassifier()
MODELS_READY = classifier.load_models()


@app.route("/classify", methods=["POST", "OPTIONS"])
def classify():
    global MODELS_READY

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    try:
        if not MODELS_READY:
            MODELS_READY = classifier.load_models()
        if not MODELS_READY:
            return jsonify({"error": "Models are not available"}), 503
        result = classifier.classify(text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Unexpected error during classification")
        return jsonify({"error": f"Classification failed: {exc}"}), 500
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
