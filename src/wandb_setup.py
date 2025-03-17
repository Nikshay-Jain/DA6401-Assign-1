import wandb

def setup_wandb(project_name="DL-Assign-1", run_name=None, args=None):
    wandb.init(project=project_name, name=run_name, config=vars(args) if args else None)
    return wandb

def log_metrics(epoch, loss, accuracy):
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})

def log_evaluation(model, X_test, Y_test):
    """ Log test accuracy, confusion matrix, and loss comparison plots. """
    test_accuracy = model.compute_accuracy(X_test, Y_test)
    wandb.log({"test_accuracy": test_accuracy})

    model.plot_confusion_matrix(X_test, Y_test)
    model.compare_losses(X_test, Y_test)

def finish_wandb():
    """ Finish the W&B logging session. """
    wandb.finish()