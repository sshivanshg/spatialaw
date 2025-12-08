from flask import Flask, request, jsonify


def create_app():
    app = Flask(__name__)

    @app.route("/alerts", methods=["POST"])
    def alerts():
        payload = request.get_json(silent=True) or {}
        app.logger.info("Received alert: %s", payload)
        return jsonify({"status": "ok", "received": payload}), 200

    @app.route("/healthz", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"}), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5055, debug=True)

