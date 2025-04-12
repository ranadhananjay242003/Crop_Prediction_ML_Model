
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__, static_folder='static')

MODEL_PATH = 'crop_prediction_model.joblib'

def train_crop_model():
    """
    Train the crop prediction model using XGBoost.
    """
    try:
        df = pd.read_csv('cr2.csv')
        print("✅ Dataset loaded.")

        # Ensure all necessary features are present
        expected_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall', 'label']
        for col in expected_columns:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']]
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        print('✅ Model training completed.')

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'🎯 Final Accuracy: {accuracy * 100:.2f}%')

        joblib.dump(model, MODEL_PATH)
        print(f"💾 Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error during training: {e}")

def predict_crop(user_input):
    """
    Predict suitable crop based on user input.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not trained yet.")

        model = joblib.load(MODEL_PATH)

        features = np.array([[user_input['N'], user_input['P'], user_input['K'],
                              user_input['temperature'], user_input['humidity'],
                              user_input['pH'], user_input['rainfall']]])
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return f"Error: {e}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = {
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'pH': float(request.form['pH']),
            'rainfall': float(request.form['rainfall'])
        }
        prediction = predict_crop(user_input)
        return render_template('result2.html', prediction=prediction)
    except Exception as e:
        return render_template('result2.html', prediction=f"Error: {e}")

if __name__ == "__main__":
    train_crop_model()  # Train once before running app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
