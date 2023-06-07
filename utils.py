import torch
from pathlib import Path

def save_checkpoint(model: torch.nn.Module,
                    target_dir: str,
                    model_name: str):
    
    """
    Saves a model checkpoint.
    
    Args:
        model (torch.nn.Module): A Pytorch model.
        target_dir (str): Path to directory to save model.
        model_name (str): Name of model.
    
    Example:
        >>> save_checkpoint(model, 'models', 'model.pth')
        
    
        
    """
    
    # Create target directory if it doesn't exist

    target_dir_path = Path(target_dir)

    if target_dir_path.is_dir():
        print(f"[INFO] Target directory {target_dir_path} exists.")
    else:
        print(f"[INFO] Target directory {target_dir_path} does not exist. Creating it now...")
    target_dir_path.mkdir(parents=True, exist_ok=True)

    
    # Check if model name endswith .pth or .pt

    assert model_name.endswith('.pth') or model_name.endswith('.pt') 

    # Create model save path

    model_save_path = target_dir_path / model_name

    # Save the model state dict
    print(f"[INFO] Saving model to {model_save_path}")
    
    torch.save(model.state_dict(), model_save_path)


