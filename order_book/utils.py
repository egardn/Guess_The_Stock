import numpy as np
import random
import gc
import pickle
from .data import OrderBookDataGenerator
from .preprocessing import create_order_book_pipeline
from .models import OrderBookEmbeddingModel
import os
import matplotlib.pyplot as plt

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


def setup_pipeline_and_model(X_path, y_path, batch_size, chunk_size=1000000):
    """
    Set up and fit the preprocessing pipeline and initialize the model
    
    Args:
        X_path: Path to X data
        chunk_size: Chunk size for initial data loading
    
    Returns:
        pipeline: Fitted preprocessing pipeline
        model: Initialized model
    """
    # Initialize the pipeline (only the processing steps)
    pipeline = create_order_book_pipeline()
    
    # Load all data at once for preprocessing
    print("Loading all data for preprocessing...")
    data_gen = OrderBookDataGenerator(
        X_path=X_path,
        y_path=y_path,
        chunk_size=chunk_size
    )
    
    # Get all X data and ignore y data
    all_X_data, _, _ = next(data_gen.generate_chunks())
    print(f"Loaded all data: {len(all_X_data)} timesteps")
    
    # Fit the entire pipeline at once (vectorizer then reshaper)
    print("Fitting preprocessing pipeline...")
    pipeline.fit(all_X_data)
    
    # Now create the model with complete mappings
    n_venues = len(pipeline.named_steps['vectorizer'].venue_mapping)
    n_actions = len(pipeline.named_steps['vectorizer'].action_mapping)
    
    print(f"Creating model with {n_venues} venues and {n_actions} actions")
    model = OrderBookEmbeddingModel(
        n_venues=n_venues,
        n_actions=n_actions,
        n_categories=24,
        epochs=1,
        batch_size=batch_size,
        learning_rate= 3e-3,
    )
    
    # Free up memory
    del all_X_data
    gc.collect()
    
    return pipeline, model


def main(X_path, y_path, batch_size, chunk_size, n_epochs,
         visualize=False, n_samples_to_visualize=1, output_dir=None, model_save_dir=None):
    """
    Main function to run the training with progressive data loading from parquet/CSV files
    
    Args:
        X_path: Path to X data (parquet file or CSV)
        y_path: Path to y data (CSV file)
        chunk_size: Number of observations to load at once
        n_epochs: Number of passes through the entire dataset
        visualize: Whether to visualize some examples
        n_samples_to_visualize: Number of sample observations to visualize
        output_dir: Directory to save visualizations
        model_save_dir: Directory to save model checkpoints every 10 chunks
    """
    print("Starting training with progressive parquet/CSV data loading...")
    
    # Create directory for model checkpoints if specified
    if model_save_dir is not None:
        os.makedirs(model_save_dir, exist_ok=True)
        print(f"Model checkpoints will be saved to {model_save_dir}")
    
    # Setup pipeline and model
    pipeline, model = setup_pipeline_and_model(X_path, y_path, batch_size)
    
    # Create a data generator with the original chunk size for training
    print("Setting up training with chunked data...")
    data_gen = OrderBookDataGenerator(
        X_path=X_path,
        y_path=y_path,
        chunk_size=chunk_size
    )
    
    visualized_count = 0  # Track how many samples we've visualized
    
    # Training for multiple epochs
    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")
        
        # Process each chunk
        for chunk_idx, (X_chunk, y_chunk, chunk_obs_ids) in enumerate(data_gen.generate_chunks()):
            print(f"Processing chunk {chunk_idx+1} with {len(chunk_obs_ids)} observations")
            
            # Map observation IDs to labels
            y_dict = dict(zip(y_chunk['obs_id'], y_chunk['eqt_code_cat']))
            
            # Determine if we need visualization data for this chunk
            need_visualization = visualize and visualized_count < n_samples_to_visualize
            
            # Process data in a single step
            if need_visualization:
                X_filtered, y_values, sequences = preprocess_for_training(
                    X_chunk, y_dict, pipeline, need_visualization=True
                )
            else:
                X_filtered, y_values = preprocess_for_training(
                    X_chunk, y_dict, pipeline, need_visualization=False
                )
                sequences = None
            
            # Visualization logic
            if need_visualization and sequences is not None:
                remaining_to_vis = n_samples_to_visualize - visualized_count
                vis_count = visualize_observations(
                    sequences, 
                    pipeline.named_steps['vectorizer'],
                    output_dir=output_dir,
                    n_samples=remaining_to_vis
                )
                visualized_count += vis_count
            
            # Train on this chunk
            print(f"Training on {len(y_values)} sequences from chunk {chunk_idx+1}")
            model.fit(X_filtered, y_values)
            
            # Save model checkpoint every 10 chunks
            if (chunk_idx + 1) % 10 == 0:
                save_model_checkpoint(model, pipeline, epoch, chunk_idx, model_save_dir)
            
            # Clean up memory
            gc.collect()

    # Save final model
    save_model_checkpoint(model, pipeline, n_epochs, 0, model_save_dir, is_final=True)

    return model, pipeline