from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import logging
# from datetime import datetime # Not used in this version
import os
import shap # For SHAP explainer

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__, static_folder='static')
CORS(app)

# Global variable to hold model data
model_components = {}

def load_model_components():
    global model_components
    try:
        with open('model.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        
        model_components['model'] = loaded_data['model']
        model_components['model_lower'] = loaded_data.get('model_lower') # .get for backward compatibility if not present
        model_components['model_upper'] = loaded_data.get('model_upper')
        model_components['label_encoders'] = loaded_data['label_encoders']
        model_components['scaler'] = loaded_data['scaler']
        model_components['feature_names'] = loaded_data['feature_names']
        model_components['categorical_mappings'] = loaded_data['categorical_mappings']
        model_components['explainer'] = loaded_data.get('explainer') # SHAP explainer
        model_components['model_name'] = loaded_data.get('model_name', 'Crop Yield Prediction Model')
        
        logging.info("Model and components loaded successfully")
        return True
    except FileNotFoundError:
        logging.error("Model file (model.pkl) not found. Please train the model first.")
        return False
    except Exception as e:
        logging.error(f"Error loading model components: {str(e)}")
        return False

# Load model components at startup
if not load_model_components():
    # If model loading fails, the app might not be fully functional.
    # You might want to raise an error or have a fallback.
    print("CRITICAL: Model loading failed. The application might not work as expected.")


@app.route('/')
def serve_index():
    return send_from_directory(model_components.get('static_folder', 'static'), 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(model_components.get('static_folder', 'static'), path)

@app.route('/get-categories', methods=['GET'])
def get_categories():
    if not model_components: # Check if model_components were loaded
        return jsonify({'success': False, 'error': 'Model not loaded properly'}), 500
    try:
        return jsonify({
            'success': True,
            'categories': model_components['categorical_mappings']
        })
    except KeyError as e:
        logging.error(f"Missing component in get-categories: {str(e)}")
        return jsonify({'success': False, 'error': f'Server configuration error: {str(e)} missing.'}), 500
    except Exception as e:
        logging.error(f"Error in get-categories: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if not model_components or not all(k in model_components for k in ['model', 'label_encoders', 'scaler', 'feature_names']):
        return jsonify({'success': False, 'error': 'Model not loaded or critical components missing.'}), 500
        
    try:
        data = request.get_json()
        logging.info(f"Received prediction request: {data}")

        # Create DataFrame in the order specified by feature_names from model.pkl
        # This ensures consistency with the training data structure.
        input_df_dict = {}
        for feature in model_components['feature_names']:
            # Map form field names to expected feature names if they differ
            # For example, if form sends 'areaCultivated' but model expects 'Area'
            form_field_map = {
                'State': 'state', 'Crop': 'cropType', 'Season': 'season',
                'Area': 'Area', 'Crop_Year': 'Crop_Year', 
                'Annual_Rainfall': 'Annual_Rainfall', 
                'Fertilizer': 'Fertilizer', 'Pesticide': 'Pesticide'
            }
            input_df_dict[feature] = [data[form_field_map.get(feature, feature)]]
        
        input_data = pd.DataFrame(input_df_dict)
        
        # Convert types for numerical features
        numerical_features = ['Area', 'Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        for col in numerical_features:
            if col in input_data.columns:
                 input_data[col] = pd.to_numeric(input_data[col], errors='coerce')


        # Encode categorical variables
        for col in model_components['label_encoders']:
            if col in input_data.columns:
                # Use a copy to avoid SettingWithCopyWarning
                input_data_col_copy = input_data[col].copy()
                input_data_col_copy = model_components['label_encoders'][col].transform(input_data_col_copy)
                input_data[col] = input_data_col_copy
            else:
                logging.warning(f"Categorical column {col} not found in input_data for encoding.")


        # Ensure columns are in the exact order as during training before scaling
        # This uses the feature_names that were AFTER encoding but BEFORE scaling during training
        # If feature_names stored are original names, this reordering might be complex here.
        # It's better if feature_names stored in model.pkl are the final numeric feature names
        # in the correct order. Assuming feature_names are the original ones for now.
        # The DataFrame was already created with feature_names_ordered from the notebook.
        
        # Scale features
        input_scaled = model_components['scaler'].transform(input_data)

        # Make prediction
        prediction = model_components['model'].predict(input_scaled)[0]
        
        # Confidence interval predictions
        pred_lower, pred_upper = None, None
        if model_components.get('model_lower') and model_components.get('model_upper'):
            pred_lower = model_components['model_lower'].predict(input_scaled)[0]
            pred_upper = model_components['model_upper'].predict(input_scaled)[0]
        
        # SHAP explanations
        feature_contributions_list = []
        if model_components.get('explainer'):
            try:
                # Convert input_scaled (numpy array) to DataFrame for SHAP if explainer needs feature names
                input_scaled_df_for_shap = pd.DataFrame(input_scaled, columns=model_components['feature_names'])
                shap_values_instance = model_components['explainer'].shap_values(input_scaled_df_for_shap)
                
                # shap_values_instance might be a 2D array if multi-output, or 1D for single output.
                # For single output regression, it's usually shap_values_instance[0] for the first (and only) sample.
                contributions = shap_values_instance[0] if isinstance(shap_values_instance, np.ndarray) and shap_values_instance.ndim > 1 else shap_values_instance

                # Map original feature names to SHAP values
                original_feature_names = model_components['feature_names'] # These are the names BEFORE encoding
                
                feature_contributions = sorted(
                    zip(original_feature_names, contributions), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                feature_contributions_list = [{'feature': f, 'contribution': round(c, 3)} for f, c in feature_contributions[:5]]

            except Exception as shap_e:
                logging.error(f"SHAP explanation error: {str(shap_e)}")


        response_payload = {
            'success': True,
            'predicted_yield': round(float(prediction), 2),
            'model_used': model_components['model_name']
        }
        if pred_lower is not None and pred_upper is not None:
            response_payload['predicted_yield_lower'] = round(float(pred_lower), 2)
            response_payload['predicted_yield_upper'] = round(float(pred_upper), 2)
        if feature_contributions_list:
            response_payload['feature_contributions'] = feature_contributions_list
            response_payload['shap_base_value'] = float(model_components['explainer'].expected_value)


        logging.info(f"Successful prediction: {response_payload} for input: {data}")
        return jsonify(response_payload)

    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        logging.error(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 400
    except ValueError as e:
        error_msg = f"Invalid value: {str(e)}"
        logging.error(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 400
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logging.error(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/get-historical-yield', methods=['GET'])
def get_historical_yield():
    state = request.args.get('state')
    crop = request.args.get('cropType') # Match form field name
    
    if not state or not crop:
        return jsonify({'success': False, 'error': 'State and CropType are required parameters'}), 400

    try:
        # This should ideally not load the CSV every time.
        # For a production app, use a database or load data into memory at startup.
        df_hist = pd.read_csv('crop_yield - Copy.csv') # Make sure this CSV is available
        
        filtered_data = df_hist[(df_hist['State'] == state) & (df_hist['Crop'] == crop)]
        if filtered_data.empty:
             return jsonify({'success': True, 'years': [], 'yields': [], 'message': 'No historical data found for this combination.'})

        historical_data = filtered_data.groupby('Crop_Year')['Yield'].mean().reset_index()
        historical_data = historical_data.sort_values(by='Crop_Year')
        
        return jsonify({
            'success': True,
            'years': historical_data['Crop_Year'].tolist(),
            'yields': historical_data['Yield'].tolist()
        })
    except FileNotFoundError:
        logging.error("Historical data CSV not found.")
        return jsonify({'success': False, 'error': 'Historical data source not available.'}), 500
    except Exception as e:
        logging.error(f"Error in get-historical-yield: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    if not model_components:
        print("Shutting down: Model could not be loaded.")
    else:
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=True, host='0.0.0.0', port=port)