from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
with open("accident_severity_model.pkl", "rb") as file:
    model_components = pickle.load(file)

xgb_model = model_components["xgboost_model"]
label_encoder = model_components["label_encoder"]

# Severity mapping (ensure it matches the training labels)
severity_mapping = {0: "Slight", 1: "Serious", 2: "Fatal"}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Collect user input from form
        light_conditions = int(request.form["light_conditions"])
        num_casualties = int(request.form["num_casualties"])
        num_vehicles = int(request.form["num_vehicles"])
        road_surface = int(request.form["road_surface"])
        speed_limit = int(request.form["speed_limit"])
        urban_rural = int(request.form["urban_rural"])
        vehicle_type = int(request.form["vehicle_type"])

        # Default values for missing features
        missing_features = [
            1,  # Default for weather_conditions
            0,  # Default for road_type
            2,  # Default for junction_detail
            0,  # Default for pedestrian_crossing
            1,  # Default for special_conditions
            0,  # Default for carriageway_hazards
        ]

        # Combine all features
        user_input = [light_conditions, num_casualties, num_vehicles, road_surface,
                      speed_limit, urban_rural, vehicle_type] + missing_features

        # Convert to NumPy array and reshape
        input_array = np.array(user_input).reshape(1, -1)

        # Make prediction
        prediction_encoded = xgb_model.predict(input_array)[0]

        # Convert numeric prediction to severity label
        prediction = severity_mapping.get(prediction_encoded, "Unknown")

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True,port=5000)
