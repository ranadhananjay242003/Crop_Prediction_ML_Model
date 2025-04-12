import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, render_template, request, url_for

app = Flask(__name__, static_folder='static')

def train_crop_model():
    """
    Train a simple crop prediction model using mock data.
    Save the model for reuse.
    """
    try:
        # Load dataset
        df = pd.read_csv('cr2.csv')
        print("Dataset loaded successfully.")

        # Check for missing 'pH' column and handle it
        if 'pH' not in df.columns:
            df['pH'] = 0  # or another default value
            print("'pH' column was missing and has been added with default values.")

        # Features and target
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']]
        y = df['label']

        print("Features and target separated.")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Training and test data split completed.")

        # Model training
        model = RandomForestClassifier()
        print("Training the RandomForestClassifier...")
        model.fit(X_train, y_train)

        # Test accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Save the model
        joblib.dump(model, 'crop_prediction_model.joblib')
        print("Model saved to 'crop_prediction_model.joblib'.")
    
    except FileNotFoundError:
        print("Error: 'cr2.csv' file not found.")
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def predict_crop(user_input):
    """
    Predict the suitable crop based on user input.
    """
    try:
        print("Loading the trained model...")
        # Load the trained model
        model = joblib.load('crop_prediction_model.joblib')
        print("Model loaded successfully.")

        print("Preparing features for prediction...")
        # Extract features for prediction
        features = np.array([[
            user_input['N'],
            user_input['P'],
            user_input['K'],
            user_input['temperature'],
            user_input['humidity'],
            user_input['pH'],
            user_input['rainfall']
        ]])
        print(f"Features: {features}")

        # Make prediction
        predicted_crop = model.predict(features)
        print(f"Prediction successful. Predicted crop: {predicted_crop[0]}")
        return predicted_crop[0]

    except FileNotFoundError:
        print("Error: Model file 'crop_prediction_model.joblib' not found.")
        return "Error: Model file not found."
    except KeyError as e:
        print(f"KeyError during prediction: {e}")
        return f"Error in prediction: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return f"Error in prediction: {e}"

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
        print(f"Received input: {user_input}")
        result = predict_crop(user_input)
        return render_template('result2.html', prediction=result)
    except ValueError as e:
        print(f"ValueError processing request: {e}")
        return render_template('result2.html', prediction=f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return render_template('result2.html', prediction=f"Error: {e}")

if __name__ == "__main__":
    train_crop_model()  # Train once before running app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
