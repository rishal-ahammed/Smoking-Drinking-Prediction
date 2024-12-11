from flask import Flask, render_template, request
import pickle
import numpy as np

# Load models
with open('models.pkl', 'rb') as f:
    models = pickle.load(f)

# Load scalers
scalers = {}
for target_col in [-2, -1]:
    with open(f"scaler_{target_col}.pkl", 'rb') as f:
        scalers[f"target_{target_col}"] = pickle.load(f)

app = Flask(__name__)

# Define the safe_convert function
def safe_convert(value, default=0.0):
    """
    Safely converts a string value to a float.
    Returns a default value if conversion fails.
    """
    try:
        return float(value) if value else default
    except ValueError:
        return default

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    try:
        # Gather input data
        feature_names = ['age', 'gender', 'height', 'weight', 'waistline',
                         'sight_left', 'sight_right', 'hear_left', 'hear_right',
                         'sbp', 'dbp', 'blds', 'tot_chole', 'hdl_chole',
                         'ldl_chole', 'triglyceride', 'hemoglobin',
                         'urine_protein', 'serum_creatinine', 'sgot_ast',
                         'sgot_alt', 'gamma_gtp']
        data = [safe_convert(request.form.get(name)) for name in feature_names]
        arr = np.array([data])

        # Make predictions
        predictions = {}
        for target_col in [-2, -1]:
            scaler = scalers[f"target_{target_col}"]
            model = models[f"target_{target_col}"]
            arr_scaled = scaler.transform(arr)
            pred = model.predict(arr_scaled)[0]
            predictions[target_col] = pred

        # Map predictions to labels
        result_details = {
            "smoking": {1: "Never smoked", 2: "Used to smoke but quit", 3: "Still smoke"},
            "drinking": {1: "Yes", 0: "No"}
        }
        result_text = []
        if -2 in predictions:
            result_text.append(f"Smoking status: {result_details['smoking'].get(predictions[-2], 'Unknown')}")
        if -1 in predictions:
            result_text.append(f"Drinking status: {result_details['drinking'].get(predictions[-1], 'Unknown')}")

        detailed_result = "<br>".join(result_text)
        print("Detailed result being sent to template:")
        print(detailed_result)  # Debugging

        return render_template('after.html', data=detailed_result)
    except Exception as e:
        return f"Error occurred: {e}", 400



if __name__ == "__main__":
    app.run(debug=True)
