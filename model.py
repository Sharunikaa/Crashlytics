import pickle
import numpy as np

# Load the trained model and encoders
with open("accident_severity_model.pkl", "rb") as file:
    model_components = pickle.load(file)

xgb_model = model_components["xgboost_model"]
ordinal_encoder = model_components["ordinal_encoder"]
label_encoder = model_components["label_encoder"]
categorical_columns = model_components["categorical_columns"]

# Define severity mapping
severity_mapping = {0: "Slight", 1: "Serious", 2: "Fatal"}

def predict_severity(user_input):
    """
    Predict accident severity based on user input.
    :param user_input: List of input features [Light_Conditions, Num_Casualties, Num_Vehicles, Road_Surface, Speed_Limit, Urban_Rural, Vehicle_Type]
    :return: Predicted accident severity label
    """

    # Convert input to NumPy array and reshape
    input_array = np.array(user_input).reshape(1, -1)

    # Make prediction
    prediction_encoded = xgb_model.predict(input_array)[0]

    # Convert numeric prediction to severity level
    prediction_label = severity_mapping.get(prediction_encoded, "Unknown")

    return prediction_label
