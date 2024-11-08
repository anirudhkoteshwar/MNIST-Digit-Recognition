import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive for persistent storage
drive.mount('/content/drive')

# Create directory for saving trained models
save_path = '/content/drive/MyDrive/mnist_models/'
os.makedirs(save_path, exist_ok=True)

# Configure logging to track training progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Custom Dataset class for MNIST data handling
class MNISTDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.FloatTensor(X)  # Features (images)
        self.y = torch.LongTensor(y)   # Labels

    def __len__(self):
        # Return the total number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single sample and its label
        return self.X[idx], self.y[idx]

# First Neural Network architecture with smaller hidden layer
class NeuralNet1(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, num_classes=10):
        super(NeuralNet1, self).__init__()
        # Sequential model with one hidden layer
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),    # Input layer to hidden layer
            nn.ReLU(),                            # Activation function
            nn.Dropout(0.2),                      # Dropout for regularization
            nn.Linear(hidden_size, num_classes)    # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)

# Second Neural Network architecture with larger hidden layer
class NeuralNet2(nn.Module):
    def __init__(self, input_size=784, hidden_size=300, num_classes=10):
        super(NeuralNet2, self).__init__()
        # Similar structure to NN1 but with larger hidden layer
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Back Propagation Neural Network with two hidden layers
class BPNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(BPNN, self).__init__()
        # More complex architecture with two hidden layers
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),           # First hidden layer
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),       # Second hidden layer (half size)
            nn.ReLU(),
            nn.Dropout(0.3),                             # Stronger dropout
            nn.Linear(hidden_size//2, num_classes)        # Output layer
        )

    def forward(self, x):
        return self.model(x)

# Function to train the neural network models
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    model.train()  # Set model to training mode
    history = []   # Store loss and accuracy for each epoch

    # Create progress bar for training
    pbar = tqdm(range(num_epochs), desc=f"Training {model.__class__.__name__}")

    for epoch in pbar:
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate over batches
        for inputs, labels in train_loader:
            # Move data to appropriate device (CPU/GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()                 # Clear previous gradients
            outputs = model(inputs)              # Get model predictions
            loss = criterion(outputs, labels)    # Calculate loss
            loss.backward()                      # Backpropagation
            optimizer.step()                     # Update weights

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history.append((epoch_loss, epoch_acc))

        # Update progress bar
        pbar.set_postfix({'loss': f'{epoch_loss:.4f}', 'acc': f'{epoch_acc:.2f}%'})

    return history

# Function to evaluate model performance
def evaluate_model(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels for further analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return correct / total, all_predictions, all_labels

def plot_training_history(history, title):
    losses, accuracies = zip(*history)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title(f'{title} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title(f'{title} - Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.show()

def main():
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess MNIST dataset
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    logging.info(f"Dataset loaded. Shape: {X.shape}")

    # Convert labels to integers and normalize pixel values
    y = y.astype(np.int32)
    X = X / 255.0  # Normalize pixel values to [0,1]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create dataset objects and data loaders
    train_dataset = MNISTDataset(X_train_scaled, y_train)
    test_dataset = MNISTDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize models and move them to appropriate device
    nn1 = NeuralNet1().to(device)
    nn2 = NeuralNet2().to(device)
    bpnn = BPNN().to(device)

    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(nn1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(nn2.parameters(), lr=0.001)
    optimizer_bpnn = optim.Adam(bpnn.parameters(), lr=0.001)

    # Train and evaluate each model individually
    print("\nTraining Neural Network 1...")
    history1 = train_model(nn1, train_loader, criterion, optimizer1, device)
    plot_training_history(history1, "Neural Network 1")

    print("\nTraining Neural Network 2...")
    history2 = train_model(nn2, train_loader, criterion, optimizer2, device)
    plot_training_history(history2, "Neural Network 2")

    print("\nTraining BPNN...")
    history_bpnn = train_model(bpnn, train_loader, criterion, optimizer_bpnn, device)
    plot_training_history(history_bpnn, "BPNN")

    # Evaluate individual models
    print("\nEvaluating individual models...")
    models = {'NN1': nn1, 'NN2': nn2, 'BPNN': bpnn}

    for name, model in models.items():
        train_acc, _, _ = evaluate_model(model, train_loader, device)
        test_acc, _, _ = evaluate_model(model, test_loader, device)
        print(f"{name} - Training accuracy: {train_acc:.3f}, Test accuracy: {test_acc:.3f}")

    # Evaluate ensemble performance (average predictions of all models)
    print("\nEvaluating ensemble...")
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get predictions from all models
            outputs1 = nn1(inputs)
            outputs2 = nn2(inputs)
            outputs_bpnn = bpnn(inputs)

            # Average the predictions
            ensemble_outputs = (outputs1 + outputs2 + outputs_bpnn) / 3

            _, predicted = torch.max(ensemble_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    ensemble_acc = correct / total
    print(f"Three-Model Ensemble - Test accuracy: {ensemble_acc:.3f}")

    # Save models and scaler for later use
    print("\nSaving models...")
    torch.save(nn1.state_dict(), f'{save_path}nn1_model.pth')
    torch.save(nn2.state_dict(), f'{save_path}nn2_model.pth')
    torch.save(bpnn.state_dict(), f'{save_path}bpnn_model.pth')
    
    # Save scaler parameters for preprocessing new data
    scaler_dict = {
        'scaler_mean_': scaler.mean_,
        'scaler_scale_': scaler.scale_,
        'n_samples_seen_': scaler.n_samples_seen_,
        'var_': scaler.var_,
        'n_features_in_': scaler.n_features_in_
    }
    torch.save(scaler_dict, f'{save_path}scaler.pth')
    
    print("Training complete!")

if __name__ == "__main__":
    main()
