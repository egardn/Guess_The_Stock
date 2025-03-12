import numpy as np
import random
import gc
import pickle
from .data import PreprocessedDataGenerator
from .preprocessing import create_order_book_pipeline
from .models import OrderBookEmbeddingModel
import os
import matplotlib.pyplot as plt
import pandas as pd

def preprocess_for_training(X_data, y_dict, pipeline, need_visualization=False):
    """
    Combined function to preprocess data and prepare it for training in one step
    
    Args:
        X_data: Raw order book data
        y_dict: Dictionary mapping observation IDs to labels
        pipeline: Preprocessing pipeline
        need_visualization: Whether to return sequences for visualization
        
    Returns:
        X_filtered: Features ready for training
        y_values: Corresponding labels
        sequences: (Optional) Sequences for visualization if need_visualization=True
    """
    # Transform the data using the pipeline
    transformed_data = pipeline.transform(X_data)
        
    # Unpack the results
    X_dict, obs_ids = transformed_data
    
    # Create a dictionary mapping observation IDs to their positions
    obs_positions = {obs_id: i for i, obs_id in enumerate(obs_ids)}
    
    # Find observations that have labels
    labeled_obs = [obs_id for obs_id in obs_ids if obs_id in y_dict]
    
    # Get positions of labeled observations
    positions = [obs_positions[obs_id] for obs_id in labeled_obs]
    
    # Create training data dictionary
    X_filtered = {
        'venue_input': X_dict['venue_input'][positions],
        'action_input': X_dict['action_input'][positions],
        'trade_input': X_dict['trade_input'][positions],
        'numeric_input': X_dict['numeric_input'][positions]
    }
    
    # Get y values for labeled observations
    y_values = np.array([y_dict[obs_id] for obs_id in labeled_obs])
    
    # Get sequences for visualization if needed
    sequences = None
    if need_visualization:
        vectorizer_pipeline = pipeline[:'vectorizer']
        sequences = vectorizer_pipeline.transform(X_data)
    
    if need_visualization:
        return X_filtered, y_values, sequences
    else:
        return X_filtered, y_values

def visualize_observations(sequences, vectorizer, output_dir=None, n_samples=1):
    """
    Visualize and save preprocessing steps for selected observations
    
    Args:
        sequences: Dictionary of sequences
        vectorizer: Fitted vectorizer
        output_dir: Directory to save visualizations
        n_samples: Number of samples to visualize
        
    Returns:
        int: Number of visualized samples
    """
    from .preprocessing import visualize_preprocessing
    
    if not sequences:
        return 0
    
    # Select random observations to visualize
    obs_to_visualize = random.sample(
        list(sequences.keys()), 
        min(n_samples, len(sequences))
    )
    
    visualized_count = 0
    
    for obs_id in obs_to_visualize:
        print(f"Visualizing observation: {obs_id}")
        fig = visualize_preprocessing(sequences, vectorizer, obs_id)
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"observation_{obs_id}.png")
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Visualization saved to {output_path}")
        
        visualized_count += 1
        
    return visualized_count

def load_all_data(X_path, y_path):
    """
    Load all data at once from X and y files
    
    Args:
        X_path: Path to X data (parquet file or CSV)
        y_path: Path to y data (CSV file)
        
    Returns:
        tuple: (X_data, y_data)
    """
    # Determine file type and load accordingly
    if X_path.endswith('.parquet'):
        X_data = pd.read_parquet(X_path)
    elif X_path.endswith('.csv'):
        X_data = pd.read_csv(X_path)
    else:
        raise ValueError(f"Unsupported file format for X_data: {X_path}")
    
    # Load y data (assuming CSV format)
    if y_path.endswith('.parquet'):
        y_data = pd.read_parquet(y_path)
    elif y_path.endswith('.csv'):
        y_data = pd.read_csv(y_path)
    else:
        raise ValueError(f"Unsupported file format for y_data: {y_path}")
    
    print(f"Loaded all data: X shape={X_data.shape}, y shape={y_data.shape}")
    return X_data, y_data

def save_model_checkpoint(model, pipeline, epoch, chunk_idx, model_save_dir, is_final=False):
    """
    Save model checkpoint to disk
    
    Args:
        model: The model to save
        pipeline: The preprocessing pipeline
        epoch: Current epoch number
        chunk_idx: Current chunk index
        model_save_dir: Directory to save the model
        is_final: Whether this is the final model
    """
    if model_save_dir is None:
        return
        
    os.makedirs(model_save_dir, exist_ok=True)
    
    if is_final:
        checkpoint_path = os.path.join(model_save_dir, "final_model.pkl")
        data_to_save = {
            'model': model,
            'pipeline': pipeline,
            'epochs_completed': epoch
        }
    else:
        checkpoint_path = os.path.join(
            model_save_dir, 
            f"model_epoch{epoch+1}_chunk{chunk_idx+1}.pkl"
        )
        data_to_save = {
            'model': model,
            'pipeline': pipeline,
            'epoch': epoch + 1,
            'chunk': chunk_idx + 1
        }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"Model saved to {checkpoint_path}")


def setup_pipeline_and_model(X_path, y_path, batch_size, 
                            val_split=0.2, preprocessed_dir=None):
    """
    Set up preprocessing pipeline, preprocess all data once, and initialize the model
    
    Args:
        X_path: Path to X data
        y_path: Path to y data
        batch_size: Batch size for model training
        chunk_size: Chunk size for initial data loading
        val_split: Fraction of data to use for validation
        preprocessed_dir: Directory to save preprocessed data
        
    Returns:
        pipeline: Fitted preprocessing pipeline
        model: Initialized model
        train_path: Path to preprocessed training data file
        val_path: Path to preprocessed validation data file
    """
    # Initialize the pipeline
    pipeline = create_order_book_pipeline()
    
    # Load all data for preprocessing - directly, without OrderBookDataGenerator
    print("Loading all data for preprocessing...")
    all_X_data, all_y_data = load_all_data(X_path, y_path)
    print(f"Loaded all data: {len(all_X_data)} timesteps")
    
    # Fit the preprocessing pipeline
    print("Fitting preprocessing pipeline...")
    pipeline.fit(all_X_data)
    
    # Create model with mappings
    n_venues = len(pipeline.named_steps['vectorizer'].venue_mapping)
    n_actions = len(pipeline.named_steps['vectorizer'].action_mapping)
    
    print(f"Creating model with {n_venues} venues and {n_actions} actions")
    model = OrderBookEmbeddingModel(
        n_venues=n_venues,
        n_actions=n_actions,
        n_categories=24,
        epochs=1,
        batch_size=batch_size,
        learning_rate=3e-3,
    )
    
    # If no preprocessed directory specified, return without preprocessing
    if preprocessed_dir is None:
        # Free up memory
        del all_X_data
        gc.collect()
        return pipeline, model, None, None
    
    # Create preprocessed data directory
    os.makedirs(preprocessed_dir, exist_ok=True)
    train_path = os.path.join(preprocessed_dir, 'train.parquet')
    val_path = os.path.join(preprocessed_dir, 'val.parquet')
    
    # Create labels dictionary
    y_dict = dict(zip(all_y_data['obs_id'], all_y_data['eqt_code_cat']))
    
    # Transform the data using the pipeline
    print("Preprocessing all data...")
    X_dict, obs_ids = pipeline.transform(all_X_data)
    print("Done.")

    print("Finding observations that have labels...")
    # Create a dictionary mapping observation IDs to their positions
    obs_positions = {obs_id: i for i, obs_id in enumerate(obs_ids)}
    
    # Find observations that have labels
    labeled_obs = [obs_id for obs_id in obs_ids if obs_id in y_dict]
    print("Done")

    print("Shuffling and splitting observation IDs in train and validation...")
    # Shuffle and split observation IDs into train and validation
    np.random.shuffle(labeled_obs)
    split_idx = int(len(labeled_obs) * (1 - val_split))
    train_obs_ids = labeled_obs[:split_idx]
    val_obs_ids = labeled_obs[split_idx:]
    
    print(f"Split data into {len(train_obs_ids)} training and {len(val_obs_ids)} validation samples")
    
    # Create preprocessed dataframes
    def create_preprocessed_df(obs_list):
        # Get positions for these observations
        positions = [obs_positions[obs_id] for obs_id in obs_list]
        
        # Create a DataFrame with all necessary columns
        df = pd.DataFrame({
            'obs_id': obs_list,
            'label': [y_dict[obs_id] for obs_id in obs_list]
        })
        
        # Add the preprocessed features
        for i, pos in enumerate(positions):
            # For each input type, extract the features for this position
            venue_input = X_dict['venue_input'][pos]
            action_input = X_dict['action_input'][pos]
            trade_input = X_dict['trade_input'][pos]
            numeric_input = X_dict['numeric_input'][pos]
            
            # Store as serialized numpy arrays
            df.at[i, 'venue_input'] = pickle.dumps(venue_input)
            df.at[i, 'action_input'] = pickle.dumps(action_input)
            df.at[i, 'trade_input'] = pickle.dumps(trade_input)
            df.at[i, 'numeric_input'] = pickle.dumps(numeric_input)
            
        return df
    
    # Create and save train dataframe
    print("Creating and saving training data...")
    train_df = create_preprocessed_df(train_obs_ids)
    train_df.to_parquet(train_path)
    print(f"Training data saved to {train_path}")
    
    # Create and save validation dataframe
    print("Creating and saving validation data...")
    val_df = create_preprocessed_df(val_obs_ids)
    val_df.to_parquet(val_path)
    print(f"Validation data saved to {val_path}")
    
    # Free up memory
    del all_X_data, all_y_data, X_dict
    gc.collect()
    
    return pipeline, model, train_path, val_path

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

def plot_loss_history(train_losses, val_losses, title="Model Training History"):
    """
    Plot training and validation loss history.
    
    Args:
        train_losses: List of training loss values per epoch
        val_losses: List of validation loss values per epoch
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def main(X_path, y_path, batch_size, chunk_size, n_epochs, val_split,
         visualize, n_samples_to_visualize, output_dir, 
         model_save_dir, preprocessed_dir, use_existing_preprocessed=False):
    """
    Main function to run the training with preprocessed data
    
    Args:
        X_path: Path to X data (parquet file or CSV)
        y_path: Path to y data (CSV file)
        batch_size: Batch size for model training
        chunk_size: Number of observations to load during training
        n_epochs: Number of passes through the entire dataset
        val_split: Fraction of data to use for validation
        visualize: Whether to visualize some examples
        n_samples_to_visualize: Number of sample observations to visualize
        output_dir: Directory to save visualizations
        model_save_dir: Directory to save model checkpoints
        preprocessed_dir: Directory to save/load preprocessed data
    """
    print("Starting training...")
    
    # Create directory for model checkpoints if specified
    if model_save_dir is not None:
        os.makedirs(model_save_dir, exist_ok=True)
        print(f"Model checkpoints will be saved to {model_save_dir}")
    
    # Setup pipeline, model, and preprocess data if needed
    if preprocessed_dir is not None:
        os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Check if preprocessed files already exist
    train_path = os.path.join(preprocessed_dir, 'train.parquet') if preprocessed_dir else None
    val_path = os.path.join(preprocessed_dir, 'val.parquet') if preprocessed_dir else None
    
    
    # Check if we can use existing preprocessed files
    if (use_existing_preprocessed and preprocessed_dir and 
        os.path.exists(train_path) and os.path.exists(val_path)):
        print("Using existing preprocessed files found at:")
        print(f"  - Training data: {train_path}")
        print(f"  - Validation data: {val_path}")
        
        # Create and initialize the pipeline
        pipeline = create_order_book_pipeline()
        
        # We need to fit the pipeline on a small sample to get feature mappings
        print("Loading a small sample of data to fit the pipeline...")
        sample_X = pd.read_parquet(X_path).head(100000) if X_path.endswith('.parquet') else pd.read_csv(X_path, nrows=1000)
        pipeline.fit(sample_X)
        
        # Create model with mappings from pipeline
        n_venues = len(pipeline.named_steps['vectorizer'].venue_mapping)
        n_actions = len(pipeline.named_steps['vectorizer'].action_mapping)
        
        print(f"Creating model with {n_venues} venues and {n_actions} actions")
        model = OrderBookEmbeddingModel(
            n_venues=n_venues,
            n_actions=n_actions,
            n_categories=24,
            epochs=1,
            batch_size=batch_size,
            learning_rate=3e-3,
        )
    else:
        # Standard process: set up pipeline, model, and preprocess data
        print("Preprocessing data...")
        pipeline, model, train_path, val_path = setup_pipeline_and_model(
            X_path, y_path, batch_size, val_split, preprocessed_dir
        )
    
    # Create data generators for train and validation
    print("Creating data generators from preprocessed files...")
    train_gen = PreprocessedDataGenerator(train_path, chunk_size=chunk_size)
    val_gen = PreprocessedDataGenerator(val_path, chunk_size=chunk_size)
    
    best_val_loss = float('inf')
    epoch_train_losses = []  # Track average training loss per epoch
    epoch_val_losses = []    # Track validation loss per epoch
    
    # Training for multiple epochs
    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")
        
        # Training phase
        print("Training phase:")
        train_losses = []
        for chunk_idx, (X_filtered, y_values) in enumerate(train_gen.generate_chunks()):
            print(f"Training on chunk {chunk_idx+1} with {len(y_values)} observations")
            
            # Train on this chunk
            model, loss_value = model.fit(X_filtered, y_values)
            train_losses.append(loss_value)
            
            # Save model checkpoint periodically
            if (chunk_idx + 1) % 10 == 0:
                save_model_checkpoint(model, pipeline, epoch, chunk_idx, model_save_dir)
                print(f"Average training loss: {np.mean(train_losses):.4f}")
                print(f"Processed {chunk_idx+1} chunks")
            # Clean up memory
            gc.collect()
        
        # Calculate average training loss for the epoch
        avg_train_loss = np.mean(train_losses)
        epoch_train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        print("Validation phase:")
        val_losses = []
        for chunk_idx, (X_filtered, y_values) in enumerate(val_gen.generate_chunks(shuffle=False)):
            print(f"Validating on chunk {chunk_idx+1}")
            
            # Evaluate on validation data
            val_loss = model.evaluate(X_filtered, y_values)
            val_losses.append(val_loss)
        
        # Calculate average validation loss
        epoch_val_loss = np.mean(val_losses)
        epoch_val_losses.append(epoch_val_loss)
        print(f"Epoch {epoch+1} validation loss: {epoch_val_loss:.4f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_model_checkpoint(model, pipeline, epoch, 0, model_save_dir, is_final=False)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    # Save final model
    save_model_checkpoint(model, pipeline, n_epochs, 0, model_save_dir, is_final=True)
    print(f"Training completed. Final validation loss: {epoch_val_loss:.4f}")

    history = {
        'train_loss': epoch_train_losses,
        'val_loss': epoch_val_losses
    }

    return model, pipeline, history
