# DA6401: Deep Learning - Assignment 1
## Nikshay Jain | MM21B044

## Introduction:
Multi class classification neural netwrok coded from scratch for the fashion-mnist dataset.

## Setup:
- Clone the GitHub repository by running the command:
    ```bash
    git clone https://github.com/Nikshay-Jain/DA6401-Assign-1.git
    ```

- Select the main directory as your working directory which has the following contents:

    ```
    A1_MM21B044
    ├── src/                      # source folder for all codes
    │   ├── dataset.py            # prepares dataset
    │   ├── model_arch.txt        # contains code for the model
    |   |── supporting_funcs.py   # contain functions necessary to be used
    │   ├── train.py              # trains and tests the model
    │   ├── wandb_setup.py        # Sets up wandb.ai
    ├── wandb/                    # Contains the log files for wandb
    │   ├── <folders1>/           # folders for logs
    │   ├── <folders1>/           # folders for logs
    ├── .gitignore                # Ignoring the venv files
    ├── fashion-mnist.npz         # fashion-mnist dataset
    ├── requirements.txt          # Dependencies for the project
    ├── Readme.md                 # This file
    ```

- Requirements:

    Python version: **3.9+**
    ```bash
    pip install -r requirements.txt
    ```
    It has the following libraries
    - tqdm - to help judge the time taken
    - wandb - to keep track of the experiments
    - keras - to get the dataset
    - tensorflow - as a supporting library for keras
    - numpy - for matrix operations
    - matplotlib - to plot figures
    - argparse - to parse arguments

- Setup wandb.ai:
    ```bash
    wandb login
    ```
    This would prompt you for API key. Just insert it in the command line and the server gets connected.

- Starting execution:
    ```bash
    python .\src\dataset.py
    python .\src\train.py
    ```
    Extra arguments can be passed while executing train in the command line itself as tabulated in the Assignment sheet.

    Here is the snippet mentioned:
    | Argument | Default Value | Description |
    |----------|--------------|-------------|
    | `-wp, --wandb_project` | `DL-Assign-1` | Project name used to track experiments in Weights & Biases dashboard. |
    | `-we, --wandb_entity` | `mm21b044` | W&B Entity used to track experiments in the Weights & Biases dashboard. |
    | `-d, --dataset` | `fashion_mnist` | Dataset to use. Choices: `["mnist", "fashion_mnist"]` |
    | `-e, --epochs` | `10` | Number of epochs to train the neural network. |
    | `-b, --batch_size` | `32` | Batch size used to train the neural network. |
    | `-l, --loss` | `cross_entropy` | Loss function to use. Choices: `["mean_squared_error", "cross_entropy"]` |
    | `-o, --optimizer` | `sgd` | Optimizer to use. Choices: `["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]` |
    | `-lr, --learning_rate` | `1e-4` | Learning rate used to optimize model parameters. |
    | `-m, --momentum` | `0.9` | Momentum used by Momentum and NAG optimizers. |
    | `-beta, --beta` | `0.9` | Beta used by RMSprop optimizer. |
    | `-beta1, --beta1` | `0.9` | Beta1 used by Adam and Nadam optimizers. |
    | `-beta2, --beta2` | `0.999` | Beta2 used by Adam and Nadam optimizers. |
    | `-eps, --epsilon` | `1e-10` | Epsilon used by optimizers. |
    | `-w_d, --weight_decay` | `0.0` | Weight decay used by optimizers. |
    | `-w_i, --weight_init` | `Xavier` | Weight initialization method. Choices: `["random", "Xavier"]` |
    | `-nhl, --num_layers` | `3` | Number of hidden layers used in the feedforward neural network. |
    | `-sz, --hidden_size` | `512` | Number of hidden neurons in a feedforward layer. |
    | `-a, --activation` | `relu` | Activation function to use. Choices: `["identity", "sigmoid", "tanh", "ReLU"]` |

## Files
### dataset.py
Contains the code to download and store the dataset as a .npz file for seamless future usage. It makes a wandb entry for the same too.

### supporting_funcs.py
This module contains essential functions for data preprocessing, activation functions, loss functions, and their derivatives to support neural network training.

- **Dataset Loading & Preprocessing**:
    - `load_dataset()`: Loads the Fashion-MNIST dataset.
    - `one_hot(inp)`: Converts labels to one-hot encoded format.
    - `Preprocess(X, y)`: Normalizes input data and applies one-hot encoding to labels.
    - `train_val_split(X, y, splits=0.1)`: Splits data into training and validation sets.

- **Activation Functions & Their Derivatives**:
    - `get_activation(activation)`: Returns the activation function (`sigmoid`, `softmax`, `ReLU`, `etc`).
    - `diff_activation(activation)`: Returns the derivative of the activation function.

- **Loss Functions & Their Derivatives**:
    - `get_loss(loss)`: Computes the loss function (`cross_entropy`, `mean_squared_error`).
    - `get_diff_loss(loss)`: Computes the gradient of the loss function.

This module is designed for efficient implementation of neural networks with custom activation and loss functions.

### model_arch.py
Contains the entire architecure for the neural network.

Structure of this file includes:-
Includes:-
- **layer class** - Each layer can be initialized to different sizes, activations, initializations (He/Xavier/Random)
- **Model class** - It is the complete model.
- Contains list of layers, loss metric, derivatives of loss mertic, regularization parameter lambda, batch size.
- **Opimizer class** - Contains code for the optimisers
    - wandb_logger: logs to wandb
    - iterate: the control iterates over epochs, performs forward, backprop
    - Calls updator, loss_calc methods 
    - Has model object, implements ealry stopping and returns the best model class trough Optimizer.model
    - uses a single layer size, activation, and initialization for each layer in  Optimizer.model

### train.py
**Features**
- **Supports multiple optimizers:** SGD, Momentum, NAG, RMSprop, Adam, Nadam
- **Dataset loading & preprocessing:** Supports Fashion-MNIST and MNIST
- **Hyperparameter tuning:** Uses WandB sweeps for optimization
- **Training & validation modes:** Supports both full training and train-validation split
- **Metrics tracking:** Logs loss, accuracy, confusion matrix, and visualizations in WandB

**Workflow**
1. **Argument Parsing**
   - The script accepts various hyperparameters via command-line arguments:
     ```bash
     --wandb_project <project_name>
     ```

2. **WandB Setup**
   - Initializes **Weights & Biases (WandB)** for experiment logging and tracking.

3. **Dataset Loading & Preprocessing**
   - Loads Fashion-MNIST or MNIST dataset, applies normalization, and prepares training/testing sets.

4. **Optimizer Selection**
   - Selects an optimizer dynamically from the supported types:
     - **SGD**
     - **Momentum**
     - **NAG (Nesterov Accelerated Gradient)**
     - **RMSprop**
     - **Adam**
     - **Nadam**

5. **Training & Validation**
   - The dataset is split into train:val = 90:10, while the test set is maintained seperate.
   - After completeing all epochs, the model is run on the unseen test set to log the metrics.

6. **Training Process**
   - Applies the selected optimizer iteratively.
   - Logs loss and accuracy at each step using WandB.

7. **Hyperparameter Tuning with WandB Sweeps**
   - Supports hyperparameter tuning via **grid search** using WandB sweeps.
   - Optimizes for minimum loss by exploring different configurations.

8. **Evaluation & Metrics Logging**
   - After training, logs test accuracy and additional metrics.
   - Generates **confusion matrix** and **loss comparison plots**.

### wandb_setup.py
This module provides functions to integrate Weights & Biases (W&B) for experiment tracking, logging metrics, and visualizing model performance.

### Features:
- **W&B Setup**:
  `setup_wandb(project_name, run_name, args)`: Initializes W&B for logging experiments.
  
- **Logging Metrics**:
  `log_metrics(epoch, loss, accuracy)`: Logs training loss and accuracy for each epoch.

- **Model Evaluation Logging**:
  `log_evaluation(model, X_test, Y_test)`: Logs test accuracy, confusion matrix, and loss comparison plots.

- **Finishing the W&B Session**:
  `finish_wandb()`: Ends the W&B logging session.

This module simplifies tracking and evaluating deep learning experiments using W&B.

## WandB report:
The WandB report for the assignment can be accessed through:
https://wandb.ai/mm21b044-indian-institute-of-technology-madras/DL-Assign-1/reports/DA6401-Assignment-1--VmlldzoxMTgzMDg0OQ?accessToken=z54kzkplm6ggnn7dn6rx71m2w9g0ce2v6fmtpcai4iaab0sns7ty7yhacusndfzt