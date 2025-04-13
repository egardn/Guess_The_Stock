# Functions for saving and loading models
import os
import pickle

def save_model_checkpoint(model, pipeline, epoch, chunk_idx, model_save_dir, is_final=False, model_type="gru"):
    """
    Save model checkpoint to disk with model type information
    
    Args:
        model: The model to save
        pipeline: The preprocessing pipeline
        epoch: Current epoch number
        chunk_idx: Current chunk index
        model_save_dir: Directory to save the model
        is_final: Whether this is the final model
        model_type: Type of model ("gru" or "gb")
    """
    if model_save_dir is None:
        return
        
    os.makedirs(model_save_dir, exist_ok=True)
    
    if is_final:
        checkpoint_path = os.path.join(model_save_dir, f"final_model_{model_type}.pkl")
        data_to_save = {
            'model': model,
            'pipeline': pipeline,
            'epochs_completed': epoch,
            'model_type': model_type
        }
    else:
        checkpoint_path = os.path.join(
            model_save_dir, 
            f"model_{model_type}_epoch{epoch+1}_chunk{chunk_idx+1}.pkl"
        )
        data_to_save = {
            'model': model,
            'pipeline': pipeline,
            'epoch': epoch + 1,
            'chunk': chunk_idx + 1,
            'model_type': model_type
        }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"{model_type.upper()} model saved to {checkpoint_path}")
