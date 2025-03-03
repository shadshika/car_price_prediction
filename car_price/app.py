import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model
model_path = "C:/car_price/model/random_forest_model.pkl"
model = joblib.load(model_path)

# Load the training dataset to create label encoders
train_df = pd.read_csv("C:/car_price/dataset/train.csv")

# Create label encoders for categorical features
categorical_columns = ['model', 'motor_type', 'wheel', 'color', 'type', 'status']
label_encoders = {}

# Initialize label encoders and fit them to the categorical columns
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(train_df[col])
    label_encoders[col] = le

# Function to preprocess the form input
def preprocess_input(form_data):
    # Convert running distance to kilometers
    def distance_to_km(distance_str):
        cleaned_str = distance_str.lower().replace(",", "").strip()
        if "miles" in cleaned_str:
            return float(cleaned_str.replace("miles", "").strip()) * 1.60934  # miles to km
        elif "km" in cleaned_str:
            return float(cleaned_str.replace("km", "").strip())
        return np.nan

    form_data["running"] = distance_to_km(form_data["running"])

    # Encoding categorical features
    for col in ['model', 'motor_type', 'wheel', 'color', 'type', 'status']:
        le = label_encoders[col]
        try:
            # Try transforming the label
            form_data[col] = le.transform([form_data[col]])[0]
        except ValueError:
            # If the label is unseen, assign it to the default value (e.g., -1 or 0)
            form_data[col] = -1  # You can adjust this as needed, or use a default category.

    # Ensure numeric fields are correctly converted
    form_data["motor_volume"] = float(form_data["motor_volume"])  # Ensure motor_volume is a float
    form_data["year"] = int(form_data["year"])  # Ensure year is an integer

    return form_data

# Routes for different pages
@app.route('/')
def index():
    return render_template('index.html')

# Home route to display the form
@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Extract form data
        form_data = {
            "model": request.form["model"],
            "year": int(request.form["year"]),
            "motor_type": request.form["motor_type"],
            "running": request.form["running"],
            "wheel": request.form["wheel"],
            "color": request.form["color"],
            "type": request.form["type"],
            "status": request.form["status"],
            "motor_volume": float(request.form["motor_volume"])
        }

        # Preprocess input data
        form_data = preprocess_input(form_data)

        # Convert the form_data dictionary to a 2D array (1 sample, multiple features)
        input_features = [list(form_data.values())]

        # Ensure that all values are numeric
        input_features = np.array(input_features, dtype=np.float64)

        # Make the prediction
        prediction = model.predict(input_features)[0]

        return render_template('result.html', prediction=round(prediction, 0))
    
    return render_template('prediction.html')


if __name__ == '__main__':
    app.run(debug=True)