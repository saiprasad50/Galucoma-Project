import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# --- Configuration ---
# Set the path for the saved model
MODEL_PATH = 'glaucoma_cnn_model.h5'
IMG_WIDTH, IMG_HEIGHT = 128, 128
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the model globally once
try:
    # Disable eager execution for better performance on some servers
    #tf.compat.v1.disable_eager_execution() 
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the expected class indices (based on your training output)
# You MUST verify these from the output of your training script!
CLASS_INDICES = {0: 'Glaucoma Detected', 1: 'Normal/No Glaucoma'}
# OR if your classes were the other way around, use:
# CLASS_INDICES = {0: 'Normal/No Glaucoma', 1: 'Glaucoma Detected'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    """Loads an image, preprocesses it, and makes a prediction."""
    try:
        # Load the image and resize to target size
        img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        
        # Convert image to numpy array
        img_array = image.img_to_array(img)
        
        # Add a batch dimension (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Rescale the image (as done in the ImageDataGenerator: rescale=1./255)
        img_array = img_array / 255.0
        
        # Make prediction
        prediction = model.predict(img_array)[0][0] # Returns a single value between 0 and 1
        
        # Determine the predicted class and confidence
        if prediction >= 0.5:
            # Class 1 (or 0) - check your indices
            predicted_class_index = 1 
            confidence = prediction 
        else:
            # Class 0 (or 1) - check your indices
            predicted_class_index = 0
            confidence = 1.0 - prediction
            
        predicted_label = CLASS_INDICES[predicted_class_index]
        
        return predicted_label, float(confidence * 100)
    
    except Exception as e:
        return f"Prediction Error: {e}", 0.0


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty part
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Save the file temporarily
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if model is None:
                return render_template('index.html', result="ERROR: Model failed to load. Check console.", image_path=None)

            # Make prediction
            predicted_label, confidence = predict_image(filepath)
            
            # Clean up the temporary file (optional but good practice)
            os.remove(filepath) 
            
            result_text = f"Result: {predicted_label} |"
            
            return render_template('index.html', result=result_text)
            
    return render_template('index.html', result=None)

if __name__ == '__main__':
    # When deploying, use a production WSGI server like Gunicorn
    # For local testing, use debug=True
    app.run(debug=True)