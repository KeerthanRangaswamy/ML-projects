import pickle
import numpy as np
from flask import Flask, request, render_template

# Load model and label encoder
model = pickle.load(open('mlmodel.pkl', 'rb'))
location_encoder = pickle.load(open('location_label_encoder.pkl', 'rb'))
#size_encoder = pickle.load(open('size_encoder.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    size = int(request.form['size'])
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    balcony = int(request.form['balcony'])

    # Encode inputs
    location_encoded = location_encoder.transform([location])[0]
    #size_encoded = size_encoder.transform([size])[0]

    # Prepare data
    features = np.array([[location_encoded, size, sqft, bath, balcony]])
    
    # Predict
    predicted_price = model.predict(features)[0]
    return render_template('index.html', prediction_text=f'Estimated Price: ₹{predicted_price:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
