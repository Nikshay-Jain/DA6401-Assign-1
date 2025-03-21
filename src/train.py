import wandb, argparse, datetime
from model_arch import *
from wandb_setup import *
from supporting_funcs import *

def log_metrics(loss, accuracy):
    wandb.log({
        "train_loss": loss,
        "train_accuracy": accuracy
    })

def train():
    setup_wandb()              # Initialize Weights & Biases using the separate setup function
    config = wandb.config

    # Load dataset
    (X_train, y_train), (X_test, y_test) = load_dataset()
    X_train, y_train = Preprocess(X_train, y_train)     # Preprocess training data
    X_test, y_test = Preprocess(X_test, y_test)         # Preprocess test data

    # Initialize optimizer
    optimizer = optimizers(
        X_size=784, Y_size=10,
        num_layers=config.num_layers,
        const_hidden_layer_size=config.hidden_size,
        const_hidden_layer_activation=config.activation.lower(),
        const_hidden_layer_initializations=config.weight_init.lower(),
        loss=config.loss,
        optimizer=config.optimizer.lower(),
        lamdba=config.weight_decay,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate
    )

    # Define updator function based on the optimizer
    if config.optimizer.lower() == 'sgd':
        def updator(t):
            optimizer.batch_gradient_descent((X_train, y_train), (X_test, y_test))

    elif config.optimizer.lower() == 'momentum':
        def updator(t):
            optimizer.momentum((X_train, y_train), (X_test, y_test), momentum=config.momentum)

    elif config.optimizer.lower() == 'nesterov':
        def updator(t):
            optimizer.NAG((X_train, y_train), (X_test, y_test), beta=config.beta)

    elif config.optimizer.lower() == 'rmsprop':
        def updator(t):
            optimizer.rmsprop((X_train, y_train), (X_test, y_test), beta=config.beta, epsilon=config.epsilon)

    elif config.optimizer.lower() == 'adam':
        def updator(t):
            optimizer.Adam((X_train, y_train), (X_test, y_test), beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon)

    elif config.optimizer.lower() == 'nadam':
        def updator(t):
            optimizer.NAdam((X_train, y_train), (X_test, y_test), beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    loss, accuracy = optimizer.iterate(updator, X_train, y_train, testdat=(X_test, y_test))
    log_metrics(loss, accuracy)
    wandb.log({"test_accuracy": optimizer.model.compute_accuracy(X_test, y_test), "_timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    wandb.finish()
    return optimizer

def main(args):
    setup_wandb(project_name=args.wandb_project, run_name=args.wandb_entity, args=args)  # Use setup_wandb() for logging

    X_size, Y_size = 784, 10
    config = wandb.config

    # Load dataset
    (X_train, y_train), (X_test, y_test) = load_dataset()
    X_test, y_test = Preprocess(X_test, y_test)             # Preprocess test data

    # Initialize optimizer
    opt = optimizers(
        X_size, Y_size,
        num_layers=config.num_layers,
        const_hidden_layer_size=config.hidden_size,
        const_hidden_layer_activation=config.activation.lower(),
        const_hidden_layer_initializations=config.weight_init.lower(),
        loss=config.loss,
        optimizer=config.optimizer.lower(),
        lamdba=config.weight_decay,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        ES=config.earlystop,
        log=config.logger
    )

    if config.mode.lower() == 'test':
        X_train, y_train = Preprocess(X_train, y_train)     # Preprocess training data
        opt.run((X_train, y_train), (X_test, y_test),
                momentum=config.momentum, beta=config.beta,
                beta1=config.beta1, beta2=config.beta2,
                epsilon=config.epsilon)
    else:
        X_train, X_val, y_train, y_val = train_val_split(X_train, y_train, splits=0.1)
        X_train, y_train = Preprocess(X_train, y_train)     # Preprocess training data
        X_val, y_val = Preprocess(X_val, y_val)             # Preprocess validation data
        opt.run((X_train, y_train), (X_val, y_val),
                momentum=config.momentum, beta=config.beta,
                beta1=config.beta1, beta2=config.beta2,
                epsilon=config.epsilon)
    
    # Log final test accuracy
    wandb.log({"test_accuracy": opt.model.compute_accuracy(X_test, y_test), "_timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    opt.model.plot_confusion_matrix(X_test, y_test)
    opt.model.compare_losses(X_test, y_test)
    return opt.model.predict(X_test, config.probab)

# Argument Parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-wp', "--wandb_project", type=str, default="DL-Assign-1")
    parser.add_argument("-we", "--wandb_entity", type=str, default="mm21b044")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", help="fashion_mnist / mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10, help='Number of epochs')
    parser.add_argument("-b", "--batch_size", type=int, default=32, help='Batch Size')
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", help="Cross-entropy loss / MSE loss")
    parser.add_argument("-o", "--optimizer", type=str, default="nadam", help="sgd/momentum/nesterov/rmsprop/adam/nadam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum used by nesterov & momentum gd")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta used by rmsprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 used by adam & nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 used by adam & nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-10)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0, help="L2 Regularizer")
    parser.add_argument("-w_i", "--weight_init", type=str, default="He", help="Xavier, He or Random Initialization")   
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers in the network")
    parser.add_argument("-sz", "--hidden_size", type=int, default=512, help="Number of neurons in each hidden layer")
    parser.add_argument("-a", "--activation", type=str, default='relu', help="Activation Function")
    parser.add_argument("-ES", "--earlystop", type=bool, default=True, help="Perform Early Stopping or not")
    parser.add_argument("-lg", "--logger", type=bool, default=True, help="Log to wandb or not")
    parser.add_argument("-md", "--mode", type=str, default="test", help="Test mode, or train+val")   
    parser.add_argument("-prb", "--probab", type=bool, default=True, help="Use probability outputs")

    args = parser.parse_args()

    # Run wandb sweep
    sweep_config = {
        'method': 'grid',
        'metric': {'name': 'loss', 'goal': 'minimize'},
        'parameters': {
            "num_layers": {"values": [3, 4, 5]},
            "hidden_size": {"values": [32, 64, 128]},
            "activation": {"values": ["relu", "tanh", "sigmoid"]},
            "weight_init": {"values": ["xavier", "random"]},
            "loss": {"values": ["cross_entropy", "mse"]},
            "optimizer": {"values": ["sgd", "nesterov", "rmsprop", "momentum", "adam", "nadam"]},
            "learning_rate": {"values": [0.001, 0.0001]},
            "batch_size": {"values": [32, 64, 128]},
            "epochs": {"values": [10, 20]},
            "weight_decay": {"values": [0, 0.0005, 0.001, 0.5]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=train, count=1)  # Run 1 experiments
    main(args)