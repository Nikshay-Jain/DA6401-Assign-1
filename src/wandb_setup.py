import wandb

def setup_wandb(project_name="DL-Assign-1", run_name=None, args=None):
    wandb.init(project=project_name, name=run_name, config=vars(args) if args else None)
    return wandb