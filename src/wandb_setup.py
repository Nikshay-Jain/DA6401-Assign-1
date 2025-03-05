import wandb

def setup_wandb(project_name="DL-Assign-1", run_name=None):
    wandb.init(project=project_name, name=run_name)
    return wandb