# MNIST Digit Recognition with Ensemble Neural Networks

This project implements a web-based handwritten digit recognition system using an ensemble of three neural networks trained on the MNIST dataset. Users can draw digits on a canvas, and the system will predict the digit using a combination of predictions from multiple models.

## Features
- Interactive drawing canvas
- Real-time digit prediction
- Ensemble prediction using three different neural network architectures
- Confidence score for predictions
- Individual model predictions display

## Prerequisites
- Python 3.8 or higher
- Git
- Web browser (Chrome/Firefox recommended)

## Installation

1. Clone the repository:
```
bash
git clone https://github.com/yourusername/mnist-ensemble.git
cd mnist-ensemble
```

2. Set up a Python virtual environment:

### For Linux/macOS:
```
bash
Create virtual environment ->
python3 -m venv venv

Activate virtual environment ->
source venv/bin/activate
```

### For Windows:
```
bash
Create virtual environment ->
python -m venv venv

Activate virtual environment ->
venv\Scripts\activate
```

3. Install required packages:
```
pip install -r requirements.txt
```

## Running the Application

1. Ensure your virtual environment is activated (see steps above)

2. Start the Flask application:
```
python app.py
```

3. Open your web browser and navigate to `http://127.0.0.1:5000/` to access the digit recognition application.


## Usage
1. Draw a digit (0-9) on the canvas using your mouse or touch screen
2. Click the "Predict" button to get the prediction
3. View the predicted digit, confidence score, and individual model predictions
4. Use the "Clear" button to reset the canvas

## Project Structure
MNIST-Digit-Recognition/
├── app.py # Flask application
├── train.py # Model training script
├── requirements.txt # Python dependencies
├── nn1_model.pth # Trained model 1
├── nn2_model.pth # Trained model 2
├── bpnn_model.pth # Trained model 3
├── scaler.pth # Scaler parameters
└── templates/
    └── index.html # Web interface


## Models
The system uses three different neural network architectures:
- NeuralNet1: Single hidden layer (100 neurons)
- NeuralNet2: Single hidden layer (300 neurons)
- BPNN: Two hidden layers (128 and 64 neurons)

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Make sure you've activated the virtual environment and installed all requirements:
```
pip install -r requirements.txt
```

2. **Port already in use**: If port 5000 is already in use, modify the port in `app.py`:
```
app = Flask(__name__, static_folder='static', template_folder='templates')
app.run(port=5001)
```

3. **Model loading errors**: Ensure all model files (nn1_model.pth, nn2_model.pth, bpnn_model.pth, scaler.pth) are in the correct directory

## Contributing
Feel free to submit issues and enhancement requests!

## License
[MIT License](LICENSE)