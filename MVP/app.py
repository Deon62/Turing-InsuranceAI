from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from huggingface_hub import snapshot_download
import os
from PIL import Image
import io
import easyocr
import cv2

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = Flask(__name__, static_folder='.')
CORS(app)

# Initialize models only once
model = None
ocr_reader = None

def load_models():
    global model, ocr_reader
    if model is None:
        print("Loading damage detection model...")
        model_dir = snapshot_download("chinesemusk/car-damage-resnet50")
        model = tf.keras.layers.TFSMLayer(model_dir, call_endpoint="serving_default")
        print("Damage detection model loaded successfully!")
    
    if ocr_reader is None:
        print("Loading OCR model...")
        ocr_reader = easyocr.Reader(['en'], gpu=False)
        print("OCR model loaded successfully!")

def prepare_image(img_data):
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(img_data))
    # Resize image
    img = img.resize((224, 224))
    # Convert to array
    img_array = image.img_to_array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return send_from_directory('.', 'claim.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/analyze-damage', methods=['POST'])
def analyze_damage():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Ensure model is loaded
        if model is None:
            load_models()
        
        file = request.files['image']
        img_data = file.read()
        
        # Prepare image for model
        img_preprocessed = prepare_image(img_data)
        
        # Run inference
        output = model(img_preprocessed)
        logits = output['output_0']
        probs = tf.sigmoid(logits).numpy()[0]
        
        # Get predictions
        labels = ['dent', 'scratch', 'crack', 'broken_light', 'glass_damage', 'no_damage']
        predicted_class_index = np.argmax(probs)
        confidence = float(probs[predicted_class_index])
        predicted_class = labels[predicted_class_index]
        
        # Format results
        results = {
            'prediction': predicted_class,
            'confidence': confidence,
            'all_probabilities': {
                label: float(prob) for label, prob in zip(labels, probs)
            }
        }
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-ocr', methods=['POST'])
def analyze_ocr():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Ensure OCR model is loaded
        if ocr_reader is None:
            load_models()
        
        file = request.files['image']
        img_data = file.read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform OCR
        results = ocr_reader.readtext(img)
        
        # Extract text and confidence scores
        extracted_data = []
        for (bbox, text, prob) in results:
            # Convert bbox coordinates to regular Python lists and int32 to int
            bbox_list = [[int(x) for x in point] for point in bbox]
            extracted_data.append({
                'text': text,
                'confidence': float(prob),
                'bbox': bbox_list
            })
        
        # Format results
        response = {
            'success': True,
            'data': extracted_data,
            'full_text': ' '.join([item['text'] for item in extracted_data])
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during OCR analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load models before starting server
    load_models()
    app.run(debug=True, port=5000) 