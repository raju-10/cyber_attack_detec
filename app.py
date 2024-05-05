from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
import numpy as np
import os
import pickle


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the trained model
model = load_model('my_model.h5')

# Load the StandardScaler and LabelEncoder objects
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/')
def home():
    return render_template('inputs.html')

@app.route('/toknowinputs.html')
def to_know_inputs():
    return render_template('toknowinputs.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the form
    input_features = [float(x) for x in request.form.values()]

    # Preprocess the input features
    input_features = np.array([input_features])
    input_features = scaler.transform(input_features)

    # Make the prediction
    prediction = model.predict(input_features)[0]

    # Get the predicted traffic type
    predicted_traffic_type = label_encoder.classes_[np.argmax(prediction)]

    print(predicted_traffic_type)
    # Render the result template with the prediction

    # Redirect to the result page with the predicted traffic type as a parameter
    return redirect(url_for('result', predicted_traffic_type=predicted_traffic_type))

@app.route('/result.html')
def result():
    # Extract the predicted traffic type from the URL parameter
    predicted_traffic_type = request.args.get('predicted_traffic_type')

    # Render the result template with the prediction
    return render_template('result.html', prediction=f'The predicted traffic type is: {predicted_traffic_type}', predicted_traffic_type= predicted_traffic_type)
if __name__ == '__main__':
    app.run(debug=True)
