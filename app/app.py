from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_features = np.array(data["features"]).reshape(1,-1)
    prediction = model.predict(input_features)
    probability = model.predict_proba(input_features)
    return jsonify({"prediction": int(prediction[0]),
		    "confidence": float(probability[0][prediction][0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000) #check your port number ( if it is in use, change the port number)
