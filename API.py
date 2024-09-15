from flask import Flask, request, jsonify, send_file
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from joblib import load
import pandas as pd
import pytesseract
import cv2
import re
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from flask_socketio import SocketIO, emit

# Initialize the Flask application
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Swagger UI configuration
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Heart Disease Prediction API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Load the pre-trained model
best_model = load('model.joblib')

# Load dataset for plotting
dataset = pd.read_csv("heart.csv")

# Calculate the mean of each feature in the dataset
feature_means = dataset.drop("target", axis=1).mean()

# Extract patient info from text
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load chatbot resources
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.json.get('message')
        if not message:
            return jsonify({"error": "No message provided"}), 400

        ints = predict_class(message)
        res = get_response(ints, intents)
        return jsonify({"response": res})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def extract_info(text):
    patterns = {
        "age": r"Age\s*[:\-]?\s*(\d+)",
        "sex": r"Sex\s*[:\-]?\s*(\d+)",
        "cp": r"Cp\s*[:\-]?\s*(\d+)",
        "trestbps": r"Trestbps\s*[:\-]?\s*(\d+)",
        "chol": r"Chol\s*[:\-]?\s*(\d+)",
        "fbs": r"Fbs\s*[:\-]?\s*(\d+)",
        "restecg": r"Restecg\s*[:\-]?\s*(\d+|O)",
        "thalach": r"Thalach\s*[:\-]?\s*(\d+)",
        "exang": r"Exang\s*[:\-]?\s*(\d+)",
        "oldpeak": r"Oldpeak\s*[:\-]?\s*([\d.]+)",
        "slope": r"Slope\s*[:\-]?\s*(\d+|O)",
        "ca": r"Ca\s*[:\-]?\s*(\d+)",
        "thal": r"Thal\s*[:\-]?\s*(\d+)"
    }

    data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            if value.isdigit():
                value = int(value)
            elif re.match(r"^[\d.]+$", value):
                value = float(value)
            elif value == 'O':
                value = 0
            data[key] = value
            print(f"Extracted {key}: {value}")  # Debug print
        else:
            print(f"Pattern not found for {key}")  # Debug print
    return data

@app.route('/extract_from_image', methods=['POST'])
def extract_from_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        # Read image
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not open or find the image"}), 400

        # Perform OCR
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(img, config=config)

        # Extract and format data
        info = extract_info(text)
        return jsonify(info)
    return jsonify({"error": "Invalid file format"}), 400

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# API route to handle patient data and predict heart disease
@app.route('/predict', methods=['POST'])
def predict_heart_disease():
    data = request.json
    try:
        # Extracting the patient data
        patient_data = np.array([[data['age'], data['sex'], data['cp'], data['trestbps'],
                                  data['chol'], data['fbs'], data['restecg'], data['thalach'],
                                  data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']]])

        # Prediction using the best model
        prediction = best_model.predict(patient_data)

        # Returning prediction result
        result = "Possible" if prediction[0] == 1 else "No ability."
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# API route to plot the prediction result
@app.route('/plot', methods=['POST'])
def plot_prediction():
    data = request.json
    try:
        # Extracting the patient data
        patient_data = np.array([[data['age'], data['sex'], data['cp'], data['trestbps'],
                                  data['chol'], data['fbs'], data['restecg'], data['thalach'],
                                  data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']]])

        # Prediction using the best model
        prediction = best_model.predict(patient_data)
        result = "Có khả năng" if prediction[0] == 1 else "Không có khả năng"

        # Create a plot
        fig, ax = plt.subplots()
        ax.bar(['Prediction'], [1] if prediction[0] == 1 else [0], color='blue' if prediction[0] == 1 else 'red')
        ax.set_ylim(0, 1.5)
        ax.set_ylabel('Probability')
        ax.set_title('Heart Disease Prediction')

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return send_file(buf, mimetype='image/png', as_attachment=True, download_name='plot.png')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# API route to plot model accuracy
@app.route('/plot_model_accuracy', methods=['GET'])
def plot_model_accuracy():
    try:
        # Prepare dataset
        X = dataset.drop("target", axis=1)
        y = dataset["target"]
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Define and train models
        models = {
            'Logistic Regression': LogisticRegression(),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(kernel='linear'),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Decision Tree': DecisionTreeClassifier(random_state=0),
            'Random Forest': RandomForestClassifier(random_state=0),
            'XGBoost': xgb.XGBClassifier(objective="binary:logistic", random_state=0),
            'Neural Network': Sequential([
                Dense(11, activation='relu', input_dim=13),
                Dense(1, activation='sigmoid')
            ])
        }

        scores = []
        for name, model in models.items():
            if name == 'Neural Network':
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X_train, Y_train, epochs=300, verbose=0)
                Y_pred = model.predict(X_test)
                Y_pred = [round(x[0]) for x in Y_pred]
            else:
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)

            accuracy = accuracy_score(Y_test, Y_pred)
            scores.append(accuracy * 100)

        # Plot accuracy
        fig, ax = plt.subplots()
        ax.bar(models.keys(), scores, color='skyblue')
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Model Accuracy Comparison')
        plt.xticks(rotation=45, ha='right')

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return send_file(buf, mimetype='image/png', as_attachment=True, download_name='model_accuracy.png')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# API route to plot target distribution
@app.route('/plot_target_distribution', methods=['GET'])
def plot_target_distribution():
    try:
        y = dataset["target"]

        fig, ax = plt.subplots()
        ax.hist(y, bins=2, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Target')
        ax.set_ylabel('Frequency')
        ax.set_title('Target Distribution')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return send_file(buf, mimetype='image/png', as_attachment=True, download_name='target_distribution.png')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# API route to plot data distribution
@app.route('/plot_data_distribution', methods=['GET'])
def plot_data_distribution():
    try:
        features = ['age', 'chol', 'trestbps']
        fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(15, 5))

        for ax, feature in zip(axes, features):
            ax.hist(dataset[feature], bins=20, color='lightblue', edgecolor='black')
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return send_file(buf, mimetype='image/png', as_attachment=True, download_name='data_distribution.png')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# API route to plot a generic chart
@app.route('/plot_chart', methods=['GET'])
def plot_chart():
    try:
        fig, ax = plt.subplots()
        ax.scatter(dataset['age'], dataset['chol'], c=dataset['target'], cmap='bwr', alpha=0.7)
        ax.set_xlabel('Age')
        ax.set_ylabel('Cholesterol')
        ax.set_title('Age vs Cholesterol')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return send_file(buf, mimetype='image/png', as_attachment=True, download_name='chart.png')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# API route to plot comparison of user data with average feature values
@app.route('/plot_feature_comparison', methods=['POST'])
def plot_feature_comparison():
    data = request.json
    try:
        # Extracting the patient data
        patient_data = np.array([
            data['age'], data['sex'], data['cp'], data['trestbps'],
            data['chol'], data['fbs'], data['restecg'], data['thalach'],
            data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
        ])

        # Create a DataFrame for plotting
        comparison_df = pd.DataFrame({
            'Feature': dataset.drop("target", axis=1).columns,
            'User Data': patient_data,
            'Average Value': feature_means
        })

        # Create a plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        index = np.arange(len(comparison_df['Feature']))

        bar1 = ax.bar(index - bar_width/2, comparison_df['User Data'], bar_width, label='User Data')
        bar2 = ax.bar(index + bar_width/2, comparison_df['Average Value'], bar_width, label='Average Value')

        ax.set_xlabel('Features')
        ax.set_ylabel('Values')
        ax.set_title('Comparison of User Data with Average Feature Values')
        ax.set_xticks(index)
        ax.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right')
        ax.legend()

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return send_file(buf, mimetype='image/png', as_attachment=True, download_name='feature_comparison.png')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Socket.IO event handler example

@socketio.on('message')
def handle_message(data):
    message = data.get('message')
    if not message:
        emit('response', {'response': 'No message provided'})
        return

    try:
        # Predict the class of the message
        ints = predict_class(message)
        # Get a response based on the predicted class
        response = get_response(ints, intents)
        # Send the response back to the client
        emit('response', {'response': response})
    except Exception as e:
        emit('response', {'response': f'Error: {str(e)}'})

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
