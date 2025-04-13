from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("housing_random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    # Validate the "features" key
    if "features" not in data:
        return jsonify({"error": '"features" key is missing'}), 400

    
    input_features = data["features"]
    predictions = []
    # Check that it's a list of lists with exactly 12 float-compatible values
    if not isinstance(input_features, list):
        return jsonify({"error": '"features" should be a list of lists',
                        "data_type": str(type(input_features))}), 400
    
    input_features = np.array(data["features"])
    for i, sample in enumerate(input_features):

        if len(sample) != 12:
            return jsonify({"error": f"Sample at index {i} does not have exactly 12 values"}), 400
        try:
            # Check float conversion
            [float(x) for x in sample]
        except ValueError:
            return jsonify({"error": f"Sample at index {i} contains non-numeric values"}), 400
        
        reshaped_sample = sample.reshape(1,-1)
        prediction = model.predict(reshaped_sample)
        predictions.append(prediction.item())

    return jsonify({"prediction": predictions})

@app.route("/health")
def health():
    return jsonify({"status":"ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001) #check your port number ( if it is in use, change the port number)
