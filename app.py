from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io

app = Flask(__name__)

# Load the trained model
model = load_model('fashion_mnist_model.h5')

class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image_data):
    
    img = Image.open(io.BytesIO(image_data))
    
    
    img = img.convert('L')
    
    
    width, height = img.size
    new_size = max(width, height)
    
    
    background = Image.new('L', (new_size, new_size), 255)
    
    
    offset = ((new_size - width) // 2, (new_size - height) // 2)
    background.paste(img, offset)
    
    
    img = background.resize((28, 28), Image.Resampling.LANCZOS)
    
    img_array = np.array(img)
    
    if np.mean(img_array[0:3, 0:3]) < 127:  # Check top-left corner
        img_array = 255 - img_array
    
    
    img_array = img_array.astype('float32') / 255.0
    
    img_array = 1 - img_array
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    image_data = image_file.read()
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]
        confidence = float(predictions[0][predicted_class] * 100)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'label': class_labels[idx],
                'confidence': float(predictions[0][idx] * 100)
            }
            for idx in top_3_indices
        ]
        
        return jsonify({
            'prediction': predicted_label,
            'confidence': f'{confidence:.2f}%',
            'top_3_predictions': top_3_predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 