import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers

app = Flask(__name__)
model = None

# Define the allowed image extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Function to check if an image file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load the model
def load_my_model():
    global model
    # Load the model from the saved file
    model = load_model('dog_emotion_model.h5')
    # Compile the model with the same settings as before
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        metrics=['accuracy']
    )

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input size of the model
    image = image.resize((96, 96))
    # Convert the image to an array and normalize its values
    image = np.array(image) / 255.
    # Expand the dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    return image

# Define the API route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request has an image file
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'})
    image_file = request.files['image']
    # Check if the image file is allowed
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'File type not allowed'})
    # Read the image file
    image_bytes = image_file.read()
    # Convert the image bytes to a PIL image
    image = Image.open(io.BytesIO(image_bytes))
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Make a prediction with the model
    prediction = model.predict(preprocessed_image)
    # Convert the prediction to a list
    prediction_list = prediction.tolist()
    # Return the prediction as JSON
    return jsonify({'prediction': prediction_list})

if __name__ == '__main__':
    # Load the model
    load_my_model()
    # Run the Flask app
    app.run(debug=True)
