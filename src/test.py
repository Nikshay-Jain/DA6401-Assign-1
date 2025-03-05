import numpy as np
import wandb
from wandb_setup import setup_wandb

# Define sweep configuration
sweep_config = {
    "method": "random",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5, 10]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "batch_size": {"values": [16, 32, 64]},
        "hidden_layers": {"values": [[32, 32], [64, 64], [128, 128]]},
        "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "weight_init": {"values": ["random", "xavier"]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="DL-Assign-1")

def train():
    wandb.init()
    config = wandb.config

    # Load dataset
    def load_fashion_mnist():
        with np.load("fashion-mnist.npz") as data:
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]
        return (x_train, y_train), (x_test, y_test)

    # Split validation set (10% of training data)
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    val_size = int(0.1 * len(x_train))
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    # Activation functions
    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return (x > 0).astype(float)

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Cross-entropy loss function
    def cross_entropy_loss(y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Adding epsilon for stability

    # Initialize neural network parameters
    def initialize_nn(input_size, hidden_layers, output_size):
        layers = [input_size] + hidden_layers + [output_size]
        weights = [np.random.randn(layers[i], layers[i+1]) * 0.01 for i in range(len(layers) - 1)]
        biases = [np.zeros((1, layers[i+1])) for i in range(len(layers) - 1)]
        return {"weights": weights, "biases": biases}

    # Forward propagation
    def forward_pass(x, params):
        a = [x.reshape(x.shape[0], -1) / 255.0]  # Flatten and normalize input
        z = []
        for w, b in zip(params["weights"], params["biases"]):
            z.append(np.dot(a[-1], w) + b)
            a.append(relu(z[-1]))
        a[-1] = softmax(z[-1])  # Apply softmax at output layer
        return a, z

    # Backward propagation
    def backward_pass(y_true, a, z, params, learning_rate):
        m = y_true.shape[0]
        y_one_hot = np.eye(10)[y_true]  # Convert to one-hot encoding
        dz = a[-1] - y_one_hot  # Softmax derivative
        
        for i in range(len(params["weights"]) - 1, -1, -1):
            dw = np.dot(a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                dz = np.dot(dz, params["weights"][i].T) * relu_derivative(z[i-1])
            
            params["weights"][i] -= learning_rate * dw
            params["biases"][i] -= learning_rate * db
    
    # Prediction function
    def predict(x, params):
        a, _ = forward_pass(x, params)
        return np.argmax(a[-1], axis=1)

    # Training function
    def train_nn(x_train, y_train, x_val, y_val, params, config):
        for epoch in range(config.epochs):
            indices = np.random.permutation(len(x_train))
            x_train, y_train = x_train[indices], y_train[indices]
            
            train_loss = 0
            for i in range(0, len(x_train), config.batch_size):
                x_batch, y_batch = x_train[i:i+config.batch_size], y_train[i:i+config.batch_size]
                a, z = forward_pass(x_batch, params)
                backward_pass(y_batch, a, z, params, config.learning_rate)

                # Compute loss for batch
                y_one_hot = np.eye(10)[y_batch]
                batch_loss = cross_entropy_loss(y_one_hot, a[-1])
                train_loss += batch_loss

            # Compute average training loss
            train_loss /= (len(x_train) // config.batch_size)

            # Compute training accuracy
            train_preds = predict(x_train, params)
            train_accuracy = np.mean(train_preds == y_train)

            # Compute validation loss & accuracy
            val_a, _ = forward_pass(x_val, params)
            y_val_one_hot = np.eye(10)[y_val]
            val_loss = cross_entropy_loss(y_val_one_hot, val_a[-1])
            val_preds = np.argmax(val_a[-1], axis=1)
            val_accuracy = np.mean(val_preds == y_val)

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "loss": train_loss,
                "accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })

            print(f"Epoch {epoch+1}/{config.epochs} - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Initialize model
    params = initialize_nn(input_size=28*28, hidden_layers=config.hidden_layers, output_size=10)

    # Train model
    train_nn(x_train, y_train, x_val, y_val, params, config)

    # Evaluate on test data
    y_pred = predict(x_test, params)
    test_acc = np.mean(y_pred == y_test)
    wandb.log({"test_accuracy": test_acc})
    print(f"Test Accuracy: {test_acc:.4f}")

wandb.agent(sweep_id, train, count=20)