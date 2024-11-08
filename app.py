from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import base64
from PIL import Image
import io
import torch
import re
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

app = Flask(__name__)
CORS(app)

# Define model architectures (same as training)
class NeuralNet1(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, num_classes=10):
        super(NeuralNet1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class NeuralNet2(nn.Module):
    def __init__(self, input_size=784, hidden_size=300, num_classes=10):
        super(NeuralNet2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class BPNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(BPNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size//2, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Initialize models and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {
    'nn1': NeuralNet1().to(device),
    'nn2': NeuralNet2().to(device),
    'bpnn': BPNN().to(device)
}


# ... existing imports and model definitions ...

# Load the saved models and scaler
print("Loading models and scaler...")
try:
    # Load models
    for name, model in models.items():
        model.load_state_dict(torch.load(f'{name}_model.pth', map_location=device))
        model.eval()
    
    # Load scaler
    scaler = StandardScaler()
    scaler_state = torch.load('scaler.pth')
    scaler.mean_ = scaler_state['scaler_mean_']
    scaler.scale_ = scaler_state['scaler_scale_']
    scaler.n_samples_seen_ = scaler_state['n_samples_seen_']
    scaler.var_ = scaler_state['var_']
    scaler.n_features_in_ = scaler_state['n_features_in_']
    
    print("Models and scaler loaded successfully")
except Exception as e:
    print(f"Error loading models or scaler: {e}")

def preprocess_image(image_data):
    """
    Preprocess the received image data for model prediction
    Args:
        image_data: Base64 encoded image string
    Returns:
        PyTorch tensor ready for model input
    """
    # Remove base64 image header
    image_data = re.sub('^data:image/png;base64,', '', image_data)
    # Decode base64 string to bytes
    image_bytes = base64.b64decode(image_data)
    
    # Convert bytes to grayscale image
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Ensure image is 28x28 (MNIST format)
    if image.size != (28, 28):
        image = image.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Invert colors (MNIST expects white digits on black background)
    img_array = 255 - img_array
    
    # Normalize pixel values to [0,1]
    img_array = img_array / 255.0
    
    # Flatten image and reshape for scaler
    img_array = img_array.reshape(1, -1)
    
    # Apply same scaling as training data
    img_array = scaler.transform(img_array)
    
    # Convert to PyTorch tensor and move to appropriate device
    img_tensor = torch.FloatTensor(img_array)
    
    return img_tensor.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Get predictions from individual models
        individual_preds = []
        all_probs = []
        
        with torch.no_grad():
            for name, model in models.items():
                outputs = model(processed_image)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                individual_preds.append(pred)
                all_probs.append(probs)
        
        # Average the probabilities for ensemble prediction
        ensemble_probs = torch.mean(torch.stack(all_probs), dim=0)
        ensemble_pred = torch.argmax(ensemble_probs).item()
        confidence = round(float(ensemble_probs[0][ensemble_pred] * 100), 2)
        
        return jsonify({
            'prediction': ensemble_pred,
            'confidence': confidence,
            'individual_predictions': individual_preds
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
