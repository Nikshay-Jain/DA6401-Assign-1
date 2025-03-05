# Quuestion 2:

import numpy as np
import wandb
from wandb_setup import setup_wandb

# Initialize Weights & Biases
wandb_run = setup_wandb(run_name="Q2-feedforward-nn")
config = wandb.config
config.update({
    "epochs": 10,
    "learning_rate": 0.01,
    "batch_size": 64,
    "hidden_layers": [128, 64]
})

def load_fashion_mnist():
    with np.load("fashion-mnist.npz") as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]
    return (x_train, y_train), (x_test, y_test)

# Activation functions
def relu(x):
    return np.maximum(0, x)

def diff_relu(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Initialize params for nn
def init_nn(input_size, hidden_layers, output_size):
    layers = [input_size] + hidden_layers + [output_size]
    weights = [np.random.randn(layers[i], layers[i+1]) * 0.01 for i in range(len(layers) - 1)]
    biases = [np.zeros((1, layers[i+1])) for i in range(len(layers) - 1)]
    return {"weights": weights, "biases": biases}

# Forward propagation
def forward_pass(x, params):
    a = [x.reshape(x.shape[0], -1) / 255.0]    # Flatten and normalize input
    z = []
    for w, b in zip(params["weights"][:-1], params["biases"][:-1]):
        z.append(np.dot(a[-1], w) + b)
        a.append(relu(z[-1]))
    z.append(np.dot(a[-1], params["weights"][-1]) + params["biases"][-1])
    a.append(softmax(z[-1]))
    return a, z

# Backward propagation
def backward_pass(y_true, a, z, params, eta):
    m = y_true.shape[0]
    y_one_hot = np.eye(10)[y_true]            # Convert to one-hot encoding
    dz = a[-1] - y_one_hot                    # Softmax derivative
    
    for i in range(len(params["weights"]) - 1, -1, -1):
        dw = np.dot(a[i].T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        
        if i > 0:
            dz = np.dot(dz, params["weights"][i].T) * diff_relu(z[i-1])
        
        params["weights"][i] -= eta * dw
        params["biases"][i] -= eta * db

# Training function
def train_nn(x_train, y_train, params, epochs, eta, batch_size):
    for epoch in range(epochs):
        indices = np.random.permutation(len(x_train))
        x_train, y_train = x_train[indices], y_train[indices]
        
        for i in range(0, len(x_train), batch_size):
            x_batch, y_batch = x_train[i:i+batch_size], y_train[i:i+batch_size]
            a, z = forward_pass(x_batch, params)
            backward_pass(y_batch, a, z, params, eta)
        
        preds = predict(x_train, params)
        acc = np.mean(preds == y_train)
        wandb.log({"epoch": epoch+1, "train_accuracy": acc})
        print(f"Epoch {epoch+1}/{epochs}, Accuracy: {acc:.4f}")

# Prediction function
def predict(x, params):
    a, _ = forward_pass(x, params)
    return np.argmax(a[-1], axis=1)

# Load data
(x_train, y_train), (x_test, y_test) = load_fashion_mnist()

# Initialize params and train nn
params = init_nn(input_size=28*28, hidden_layers=[128, 64], output_size=10)
train_nn(x_train, y_train, params, epochs=config.epochs, eta=config.learning_rate, batch_size=config.batch_size)

# Evaluate on test data
y_pred = predict(x_test, params)
test_acc = np.mean(y_pred == y_test)
wandb.log({"test_accuracy": test_acc})
print(f"Test Accuracy: {test_acc:.4f}")