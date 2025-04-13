# Model evaluation metrics and functions
def load_loss_history_from_checkpoints(checkpoint_dir):
    """Extract loss information from model checkpoints if available.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        
    Returns:
        dict: Dictionary with training and validation loss history
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("model_epoch*_chunk*.pkl"))
    
    # Dictionary to store loss values by epoch
    epoch_losses = {}
    
    # Extract epoch numbers and losses from checkpoint files
    for checkpoint_path in checkpoints:
        # Extract epoch number from filename
        match = re.search(r'epoch(\d+)_chunk(\d+)', checkpoint_path.name)
        if not match:
            continue
            
        epoch = int(match.group(1))
        
        # Load the checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        # Check if loss information is stored in the checkpoint
        # Note: This depends on how loss is stored in your checkpoints
        if 'val_loss' in checkpoint_data:
            if epoch not in epoch_losses:
                epoch_losses[epoch] = {}
            epoch_losses[epoch]['val_loss'] = checkpoint_data['val_loss']
            
        if 'train_loss' in checkpoint_data:
            if epoch not in epoch_losses:
                epoch_losses[epoch] = {}
            epoch_losses[epoch]['train_loss'] = checkpoint_data['train_loss']
    
    # If no loss information found in checkpoints, return empty history
    if not epoch_losses:
        print("No loss history found in checkpoints.")
        return {'train_loss': [], 'val_loss': []}
    
    # Convert to lists for plotting
    epochs = sorted(epoch_losses.keys())
    train_losses = [epoch_losses[e].get('train_loss', None) for e in epochs]
    val_losses = [epoch_losses[e].get('val_loss', None) for e in epochs]
    
    return {'train_loss': train_losses, 'val_loss': val_losses, 'epochs': epochs}