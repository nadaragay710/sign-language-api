from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max

# Load your model (update paths as needed)
MODEL_PATH = 'sign_language_model_final.h5'
CLASS_MAPPING_PATH = 'class_mapping.json'

# Load model and class mapping
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_MAPPING_PATH, 'r') as f:
        class_mapping = json.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    class_mapping = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Adjust based on your model
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        if model:
            processed_image = preprocess_image(filepath)
            if processed_image is not None:
                prediction = model.predict(processed_image)
                predicted_class = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class])
                
                # Get letter from class mapping
                letter = class_mapping.get(str(predicted_class), 'Unknown')
                
                # Return result
                return jsonify({
                    'success': True,
                    'letter': letter,
                    'confidence': confidence,
                    'image_url': f'/static/uploads/{filename}'
                })
        
        return jsonify({'error': 'Prediction failed'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
