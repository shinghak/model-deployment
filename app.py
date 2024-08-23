from flask import Flask, request, render_template, jsonify
from pycaret.regression import load_model, predict_model
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and ensure it includes preprocessing steps
model = load_model('price_pipeline')

# Define columns based on the model's expected input
cols = [
    'flat_type', 
    'storey_range', 
    'floor_area_sqm', 
    'flat_model', 
    'cbd_dist', 
    'min_dist_mrt', 
    'lease_left'
]

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        int_features = [x for x in request.form.values()]
        
        # Convert features to a DataFrame
        data_unseen = pd.DataFrame([int_features], columns=cols)
        
        # Ensure data is in the correct format for PyCaret
        for col in cols:
            if col not in data_unseen.columns:
                data_unseen[col] = np.nan  # Add missing columns with NaN

        # Predict using the model
        prediction = predict_model(model, data=data_unseen)
        
        # Get the predicted value and format it
        prediction_value = prediction['prediction_label'][0]
        formatted_value = "${:,.2f}".format(prediction_value)
        
        # Update the output message
        output_message = f'Predicted Resale price will be {formatted_value}'
    except Exception as e:
        output_message = f'Error in prediction: {str(e)}'
    
    return render_template('home.html', pred=output_message)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get the data from the request
        data = request.get_json(force=True)
        
        # Convert data to DataFrame
        data_unseen = pd.DataFrame([data])
        
        # Ensure data is in the correct format for PyCaret
        for col in cols:
            if col not in data_unseen.columns:
                data_unseen[col] = np.nan  # Add missing columns with NaN
        
        # Predict using the model
        prediction = predict_model(model, data=data_unseen)
        
        # Get the predicted value
        output = prediction['prediction_label'][0]
    except Exception as e:
        output = f'Error in prediction: {str(e)}'
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
