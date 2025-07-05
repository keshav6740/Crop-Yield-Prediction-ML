from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import logging
import os
import shap

# --- Setup ---
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__) # The static_folder is now correctly inferred
CORS(app)

# --- Global variables for model and data ---
model_components = {}
historical_df = None

# --- Loading Functions ---
def load_model_and_data():
    """Loads the model components and historical data into memory at startup."""
    global model_components, historical_df
    
    # Load Model
    try:
        with open('model.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        
        model_components['model'] = loaded_data['model']
        model_components['model_lower'] = loaded_data.get('model_lower')
        model_components['model_upper'] = loaded_data.get('model_upper')
        model_components['label_encoders'] = loaded_data['label_encoders']
        model_components['scaler'] = loaded_data['scaler']
        model_components['feature_names'] = loaded_data['feature_names']
        model_components['categorical_mappings'] = loaded_data['categorical_mappings']
        model_components['explainer'] = loaded_data.get('explainer')
        model_components['model_name'] = loaded_data.get('model_name', 'Crop Yield Prediction Model')
        
        logging.info("Model and components loaded successfully.")
    except FileNotFoundError:
        logging.error("Model file (model.pkl) not found.")
        raise
    except Exception as e:
        logging.error(f"Error loading model components: {str(e)}")
        raise

    # Load Historical Data CSV
    try:
        # Corrected filename and load once
        historical_df = pd.read_csv('crop_yield.csv') 
        historical_df['State'] = historical_df['State'].str.strip()
        historical_df['Crop'] = historical_df['Crop'].str.strip()
        logging.info("Historical data (crop_yield.csv) loaded successfully.")
    except FileNotFoundError:
        logging.error("Historical data file (crop_yield.csv) not found.")
        # You might want to allow the app to run without historical data
        historical_df = pd.DataFrame() # Empty dataframe to prevent errors
    except Exception as e:
        logging.error(f"Error loading historical data: {str(e)}")
        raise

# Load components when the app starts
load_model_and_data()

# --- Routes ---
@app.route('/')
def serve_index():
    # Renders the HTML file from the 'templates' folder
    return render_template('index.html')

@app.route('/get-categories', methods=['GET'])
def get_categories():
    if not model_components:
        return jsonify({'success': False, 'error': 'Model not loaded properly'}), 500
    return jsonify({
        'success': True,
        'categories': model_components['categorical_mappings']
    })

@app.route('/predict', methods=['POST'])
def predict():
    # This function remains largely the same, but it's good practice
    # to check that the model is loaded.
    if not all(k in model_components for k in ['model', 'scaler', 'label_encoders']):
        return jsonify({'success': False, 'error': 'Server not ready, model components missing.'}), 503

    try:
        data = request.get_json()
        logging.info(f"Received prediction request: {data}")

        # ... (Your existing prediction logic here is fine) ...
        # (Create DataFrame, encode, scale, predict, get SHAP values)

        input_df_dict = {}
        for feature in model_components['feature_names']:
            form_field_map = {
                'State': 'state', 'Crop': 'cropType', 'Season': 'season',
                'Area': 'Area', 'Crop_Year': 'Crop_Year',
                'Annual_Rainfall': 'Annual_Rainfall',
                'Fertilizer': 'Fertilizer', 'Pesticide': 'Pesticide'
            }
            input_df_dict[feature] = [data[form_field_map.get(feature, feature)]]
        
        input_data = pd.DataFrame(input_df_dict)
        
        numerical_features = ['Area', 'Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        for col in numerical_features:
            if col in input_data.columns:
                 input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

        for col, le in model_components['label_encoders'].items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col])
        
        input_scaled = model_components['scaler'].transform(input_data)
        prediction = model_components['model'].predict(input_scaled)[0]
        
        pred_lower, pred_upper = None, None
        if model_components.get('model_lower') and model_components.get('model_upper'):
            pred_lower = model_components['model_lower'].predict(input_scaled)[0]
            pred_upper = model_components['model_upper'].predict(input_scaled)[0]
        
        feature_contributions_list = []
        if model_components.get('explainer'):
            input_scaled_df_for_shap = pd.DataFrame(input_scaled, columns=model_components['feature_names'])
            shap_values_instance = model_components['explainer'].shap_values(input_scaled_df_for_shap)
            contributions = shap_values_instance[0]
            feature_contributions = sorted(zip(model_components['feature_names'], contributions), key=lambda x: abs(x[1]), reverse=True)
            feature_contributions_list = [{'feature': f, 'contribution': round(c, 3)} for f, c in feature_contributions[:5]]

        response_payload = {
            'success': True,
            'predicted_yield': round(float(prediction), 2),
            'model_used': model_components['model_name']
        }
        if pred_lower is not None:
            response_payload['predicted_yield_lower'] = round(float(pred_lower), 2)
        if pred_upper is not None:
            response_payload['predicted_yield_upper'] = round(float(pred_upper), 2)
        if feature_contributions_list:
            response_payload['feature_contributions'] = feature_contributions_list
            response_payload['shap_base_value'] = float(model_components['explainer'].expected_value)
        
        logging.info(f"Successful prediction: {response_payload}")
        return jsonify(response_payload)
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get-historical-yield', methods=['GET'])
def get_historical_yield():
    state = request.args.get('state')
    crop = request.args.get('cropType')
    
    if not state or not crop:
        return jsonify({'success': False, 'error': 'State and CropType are required'}), 400

    if historical_df is None or historical_df.empty:
        return jsonify({'success': False, 'error': 'Historical data not available on server.'}), 503

    try:
        filtered_data = historical_df[(historical_df['State'] == state) & (historical_df['Crop'] == crop)]
        if filtered_data.empty:
             return jsonify({'success': True, 'years': [], 'yields': [], 'message': 'No historical data found.'})

        data = filtered_data.groupby('Crop_Year')['Yield'].mean().sort_index()
        
        return jsonify({
            'success': True,
            'years': data.index.tolist(),
            'yields': data.values.tolist()
        })
    except Exception as e:
        logging.error(f"Error in get-historical-yield: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # This part is for local execution only. 
    # A production server will call the 'app' object directly.
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)