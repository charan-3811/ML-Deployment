# Importing essential libraries
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

# Load the Random Forest Classifier model
filename = 'heart-disease.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        age = int(data['age'])
        sex = data['sex']
        cp = data['cp']
        trestbps = int(data['trestbps'])
        chol = int(data['chol'])
        fbs = data['fbs']
        restecg = int(data['restecg'])
        thalach = int(data['thalach'])
        exang = data['exang']
        oldpeak = float(data['oldpeak'])
        slope = data['slope']
        ca = int(data['ca'])
        thal = data['thal']

        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        print(input_data)
        my_prediction = model.predict(input_data)
        print(my_prediction)
        return jsonify({'prediction': int(my_prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
