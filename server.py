from flask import Flask, request, jsonify
from flask_cors import CORS
from forecast_refill_lib import get_next_refill
import os

app = Flask(__name__,
            static_folder="pet-feeder-ui/build",
            static_url_path="/")
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    last = data.get('last')  # expecting 'YYYY-MM-DD'
    try:
        next_dt, interval = get_next_refill(last)
        return jsonify(
            next_refill    = next_dt.strftime('%Y-%m-%d %H:%M:%S'),
            interval_hours = round(interval, 2)
        )
    except Exception as e:
        return jsonify(error=str(e)), 400

# Serve React build
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    build_dir = app.static_folder
    full = os.path.join(build_dir, path)
    if path and os.path.exists(full):
        return app.send_static_file(path)
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
