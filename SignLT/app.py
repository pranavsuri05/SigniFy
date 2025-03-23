from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/recognize', methods=['GET'])
def recognize_sign():
    # Dummy response for testing
    return jsonify({"text": "Hello from Flask!"})

if __name__ == '__main__':
    app.run(debug=True)