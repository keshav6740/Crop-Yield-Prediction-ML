# Enhanced Crop Yield Prediction ML Project

![Crop Yield Prediction Banner](<banner.jpeg>) <!-- Optional: Create a banner image for your project -->

This project leverages machine learning to predict crop yields based on various agricultural and environmental factors. It features an interactive web application built with Flask, advanced regression models for prediction, confidence interval estimation, and SHAP-based explainability to understand the key drivers behind each prediction.

**Repository:** [https://github.com/keshav6740/Crop-Yield-Prediction-ML](https://github.com/keshav6740/Crop-Yield-Prediction-ML)

## Table of Contents
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Model Training & Selection](#model-training--selection)
  - [Confidence Intervals](#confidence-intervals)
  - [Model Explainability (XAI)](#model-explainability-xai)
  - [Yield Trend Classification (Exploratory)](#yield-trend-classification-exploratory)
- [Technology Stack](#technology-stack)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
  - [Training the Model](#training-the-model)
  - [Starting the Flask Server](#starting-the-flask-server)
- [Web Application Usage](#web-application-usage)
- [Model Performance](#model-performance)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Key Features

*   **Accurate Yield Prediction:** Utilizes an XGBoost Regressor model, fine-tuned for optimal performance.
*   **Confidence Intervals:** Provides a 90% confidence interval for predictions using LightGBM Quantile Regression, offering a measure of prediction uncertainty.
*   **Explainable AI (XAI):** Integrates SHAP (SHapley Additive exPlanations) to display the top factors influencing each prediction and their impact (increase/decrease).
*   **Interactive Web Interface:** A user-friendly frontend built with HTML, CSS, and JavaScript, served by a Flask backend.
*   **Dynamic Inputs:** Categorical options (State, Crop, Season) are dynamically populated from the trained model's context.
*   **Historical Data Visualization:** Displays a line chart of historical average yields for the selected State and Crop, plotting the current prediction for comparison using Chart.js.
*   **Robust Preprocessing:** Includes handling of missing values, outlier capping, label encoding for categorical features, and feature scaling.
*   **Comprehensive EDA:** The Jupyter notebook includes detailed exploratory data analysis to understand feature distributions and relationships.
*   **Exploratory Classification:** An additional analysis to predict whether yield will increase or decrease compared to the previous year.

## Project Structure

<pre>
Crop-Yield-Prediction-ML/
├── README.md                     # This file
├── app.py                        # Flask application (backend)
├── crop_yield.csv                # Dataset
├── cropyield.ipynb               # Jupyter Notebook for EDA, model training, and evaluation
├── model.pkl                     # Saved ML model, scaler, encoders, SHAP explainer, etc.
├── app.log                       # Application log file (generated at runtime)
├── index.html                    # Main HTML page
├── script.js                     # Client-side JavaScript
└── styles.css                    # CSS for styling
</pre>


## Dataset

The primary dataset used is `crop_yield.csv`. It contains historical data on crop production, including:

*   **Categorical Features:**
    *   `Crop`: Type of crop (e.g., Rice, Wheat, Maize).
    *   `Season`: Sowing season (e.g., Kharif, Rabi).
    *   `State`: Indian state where the crop was grown.
*   **Numerical Features:**
    *   `Crop_Year`: Year of cultivation.
    *   `Area`: Cultivated area in hectares (ha).
    *   `Annual_Rainfall`: Annual rainfall in millimeters (mm).
    *   `Fertilizer`: Amount of fertilizer used in kilograms per hectare (kg/ha).
    *   `Pesticide`: Amount of pesticide used in kilograms per hectare (kg/ha).
*   **Target Variable:**
    *   `Yield`: Crop yield in tons per hectare (tons/ha). (Calculated as Production/Area in the original dataset).

## Methodology

### Data Preprocessing
1.  **Loading Data:** The dataset is loaded using Pandas.
2.  **Cleaning:** The 'Production' column is dropped as 'Yield' is the direct target. Whitespace is trimmed from object columns.
3.  **Missing Value Imputation:**
    *   Numerical features: Median imputation.
    *   Categorical features: Mode imputation.
4.  **Outlier Handling (Regression):** Outliers in numerical features are capped using the Interquartile Range (IQR) method (values outside 1.5 * IQR from Q1 or Q3 are capped).
5.  **Categorical Encoding:** `LabelEncoder` from scikit-learn is used to convert categorical features ('State', 'Crop', 'Season') into numerical representations. Mappings are saved for frontend dropdowns.
6.  **Feature Scaling:** `StandardScaler` from scikit-learn is applied to numerical features to standardize them (mean 0, variance 1) after splitting the data into training and testing sets.

### Exploratory Data Analysis (EDA)
The `cropyield.ipynb` notebook contains visualizations and statistical summaries, including:
*   Distribution of the target variable (Yield).
*   Correlation matrix to understand linear relationships between features.
*   Box plots showing Yield distribution across different States, Crops, and Seasons (top 10 categories).
*   Line plot of average crop yield over the years.
*   Pairplots for selected numerical features against Yield.

### Model Training & Selection

#### Regression (Yield Prediction)
1.  **Train-Test Split:** Data is split into 80% training and 20% testing sets.
2.  **Model Choice:** XGBoost Regressor (`XGBRegressor`) was chosen as the primary model due to its high performance on tabular data.
3.  **Hyperparameter Tuning:** `GridSearchCV` is used to find the optimal hyperparameters for the XGBoost model, optimizing for R² score. The search space includes `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.
4.  **Evaluation Metrics:** Root Mean Squared Error (RMSE), R² Score, and Mean Absolute Error (MAE) are used to evaluate the regression model on the test set.

#### Confidence Intervals
To provide a measure of uncertainty with the predictions:
1.  **Quantile Regression:** LightGBM (`LGBMRegressor`) is trained with a 'quantile' objective.
2.  Two models are trained:
    *   One for the lower bound (alpha=0.05, predicting the 5th percentile).
    *   One for the upper bound (alpha=0.95, predicting the 95th percentile).
3.  This provides a 90% confidence interval for the predicted yield.
4.  Coverage and average interval width are checked on the test set.

### Model Explainability (XAI)
SHAP (SHapley Additive exPlanations) is used to interpret the predictions of the XGBoost regression model:
1.  A `shap.TreeExplainer` is initialized with the trained XGBoost model.
2.  **Global Explanations:** SHAP summary plots (bar and dot plots) are generated to show the overall importance of each feature.
3.  **Local Explanations:** For individual predictions, SHAP waterfall plots can illustrate how each feature value contributes to moving the prediction away from the base (expected) value. The top 5 contributing features and their impact are displayed in the web app.

### Yield Trend Classification (Exploratory)
An additional classification task was explored in the notebook to predict whether the yield for a specific crop in a state would increase or decrease compared to its previous year.
1.  **Target Variable Creation:** A binary target is created (1 if current year's yield > previous year's yield, 0 otherwise). This requires grouping by 'State' and 'Crop' and using `shift(1)` on the 'Yield' column.
2.  **Feature Engineering:** Difference features (e.g., `Diff_Annual_Rainfall`) were created.
3.  **Data Handling:** Rows with NaN values (due to the shift operation) are dropped.
4.  **Class Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to address potential class imbalance.
5.  **Models Tested:** Decision Tree, Random Forest, and XGBoost classifiers were trained and evaluated.
6.  **Evaluation:** Accuracy, Precision, Recall, F1-score, and confusion matrices are used.
*Note: This classification model is currently for analysis and not integrated into the primary prediction endpoint of the Flask app, but its results are shown in the notebook.*

## Technology Stack

*   **Backend:**
    *   Python 3.12
    *   Flask (for the web server and API)
    *   Flask-CORS (for handling Cross-Origin Resource Sharing)
*   **Machine Learning & Data Processing:**
    *   Pandas (for data manipulation and analysis)
    *   NumPy (for numerical operations)
    *   Scikit-learn (for preprocessing, model evaluation, LabelEncoder, StandardScaler, GridSearchCV, DecisionTreeClassifier, RandomForestClassifier/Regressor)
    *   XGBoost (for the primary regression model and classification)
    *   LightGBM (for quantile regression)
    *   SHAP (for model explainability)
    *   Imbalanced-learn (for SMOTE)
*   **Frontend:**
    *   HTML5
    *   CSS3
    *   JavaScript (ES6+)
    *   Anime.js (for simple animations)
    *   Chart.js (for plotting historical data)
*   **Development Environment:**
    *   Jupyter Notebook (for model development and EDA)
    *   Pickle (for saving and loading the trained model and components)
*   **Logging:** Python's built-in `logging` module.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/keshav6740/Crop-Yield-Prediction-ML.git
    cd Crop-Yield-Prediction-ML
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Dataset:**
    Ensure the `crop_yield.csv` file is present in the root directory of the project.

## Running the Application

### Training the Model
If `model.pkl` is not present or you wish to retrain the model with new data or parameters:
1.  Open the `cropyield.ipynb` Jupyter Notebook.
2.  Ensure the `crop_yield.csv` file is correctly referenced.
3.  Run all cells in the notebook. This will perform EDA, train the regression and quantile models, create the SHAP explainer, and save all necessary components into `model.pkl`.
    *The classification part of the notebook is for analysis and its model is not currently saved in the main `model.pkl` for the app.*

### Starting the Flask Server
Once `model.pkl` is generated:
1.  Navigate to the project's root directory in your terminal.
2.  Run the Flask application:
    ```bash
    python app.py
    ```
3.  The application will typically be available at `http://localhost:5000` (or the port specified in the console). Open this URL in your web browser.

## Web Application Usage

1.  **Open the Web Interface:** Navigate to `http://localhost:5000`.
2.  **Input Crop Details:**
    *   Select the `State`, `Crop Type`, and `Season` from the dropdown menus. These are populated based on the data used for training.
    *   Enter numerical values for `Area Cultivated`, `Crop Year`, `Annual Rainfall`, `Fertilizer Used`, and `Pesticide Used`. Default values are provided as examples.
3.  **Predict Yield:** Click the "Predict Yield" button.
4.  **View Results:**
    *   A loading indicator will appear while the prediction is processed.
    *   The results section will display:
        *   **Predicted Yield:** The primary yield prediction from the XGBoost model.
        *   **Confidence Interval (90%):** The estimated lower and upper bounds for the yield, generated by the LightGBM quantile models.
        *   **Model Used:** Name of the model that generated the prediction.
        *   **Top Factors Influencing Prediction:** A list of the top 5 features and their SHAP contribution, indicating how they influenced the prediction (increased or decreased).
        *   **Historical Yield Data:** A line chart showing the average historical yield for the selected State and Crop, with the current year's prediction plotted for comparison.

## Model Performance

**Regression Model (XGBoost on Test Set):**
*   **R² Score:** `0.8860`
*   **RMSE:** `0.7068`
*   **MAE:** `0.3777`
*   **Quantile Coverage (90% CI):** `<89%>` - indicates how often the true value falls within the predicted interval.


## Future Enhancements

*   **More Granular Data:** Incorporate district-level data instead of state-level for more precise predictions.
*   **Advanced Feature Engineering:**
    *   Soil type and quality data.
    *   More detailed weather parameters (e.g., temperature, humidity, sunshine hours, specific rainfall patterns during growing season).
    *   Interaction terms between features.
    *   Lagged variables.
*   **Time Series Models:** Explore models like ARIMA, SARIMA, or LSTMs if sufficient historical data per crop/location is available.
*   **Ensemble Methods:** Combine predictions from multiple strong models.
*   **User Authentication:** Allow users to save their farm details and track predictions.
*   **Deployment:** Dockerize the application and deploy it to a cloud platform (AWS, Google Cloud, Azure, Heroku).
*   **Continuous Integration/Continuous Deployment (CI/CD):** Set up a pipeline for automated testing and deployment.
*   **API for External Use:** Expose the prediction endpoint as a secure API.
*   **Improved UI/UX:** Further enhance the user interface with more visualizations and user guidance.
*   **Feedback Mechanism:** Allow users to provide feedback on prediction accuracy to help refine the model.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.


## Acknowledgements

*   Dataset source (if applicable, or mention it's a common type of dataset).
*   Libraries used: Scikit-learn, Pandas, NumPy, XGBoost, LightGBM, SHAP, Flask, Matplotlib, Seaborn, Chart.js.
*   Inspiration or tutorials that helped.
