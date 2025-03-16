from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
import tensorflow as tf

from PIL import Image

import base64
from io import BytesIO
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow informational logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

@app.template_filter('pil_to_b64')
def pil_to_b64(pil_img):
    img_buffer = BytesIO()
    pil_img.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str

model = keras.models.load_model('model/imageclassifier_keras3.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['image']
            if not file:
                return "No file uploaded"

            # Process image
            img = Image.open(file.stream).convert('RGB')  # Ensure image is in RGB format
            img = img.resize((256, 256))  # Resize using PIL
            img_array = np.array(img) / 255.0  # Normalize pixel values

            # Make prediction
            prediction = model.predict(np.expand_dims(img_array, axis=0))
            emotions = 'Happy' if prediction[0] > 0.5 else 'Sad'
            
            return render_template('result.html', prediction=prediction[0], emotions=emotions, imgPath=img)

        except Exception as e:
            return f"Error processing image: {e}"

if __name__ == '__main__':
    # Start the Flask app on the default port provided by Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
