# Complete workflow for GRU models
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
import time
from gts_challenge.order_book.models.gru.pipeline import GRUPipeline

def preprocess_data(X_path, y_path, val_split, preprocessed_dir, chunk_size):
    """GRU-specific preprocessing implementation
    
    Args:
        X_path: Path to X data file (parquet or csv)
        y_path: Path to y data file (parquet or csv)
        val_split: Ratio of validation data
        preprocessed_dir: Directory to save/load preprocessed data
        
    Returns:
        Tuple of (train_data, val_data) for model training
    """
    from gts_challenge.order_book.data.loaders import load_all_data
    from gts_challenge.order_book.data.data_utils import save_preprocessed_train_data, check_preprocessed_data, train_val_split

    # Setup preprocessed directory
    preprocessed_dir = Path(preprocessed_dir)
    
    preprocessed_dir.mkdir(exist_ok=True)
    
    # Define paths
    X_train_path = preprocessed_dir / "gru_train_features.h5"
    X_val_path = preprocessed_dir / "gru_val_features.h5"
    y_train_path = preprocessed_dir / "gru_y_train.parquet"
    y_val_path = preprocessed_dir / "gru_y_val.parquet"
    pipeline_path = preprocessed_dir / "gru_pipeline.pkl"

    # Check if preprocessed data already exists
    if X_train_path.exists() and X_val_path.exists():
        print(f"Preprocessed data found. It will be loaded from {preprocessed_dir}")
    else:
        # Load raw data using existing function
        print(f"Loading raw data from {X_path} and {y_path}")
        X, y = load_all_data(X_path, y_path, convert_to_parquet=True)
        
        # Split data before preprocessing
        X_train, X_val, y_train, y_val = train_val_split(X, y, val_split, random_state=42)
        
        # Process with pipeline
        pipeline = create_and_fit_pipeline(X_train, y_train)

        X_train_transformed, X_val_transformed, y_train_transformed, y_val_transformed = transform_data(
            pipeline, X_train, y_train, X_val, y_val)

        del X_train, X_val, y_train, y_val
    
        # Save processed data
        save_preprocessed_train_data(X_train_transformed, X_val_transformed, 
            y_train_transformed, y_val_transformed, 
            pipeline, 
            X_train_path, X_val_path, 
            y_train_path, y_val_path, 
            pipeline_path)
    
    return X_train_path, y_train_path, X_val_path, y_val_path

def create_and_fit_pipeline(X_train, y_train):
    """Create and fit the GRU pipeline"""
    pipeline = GRUPipeline()
    print("Fitting pipeline...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Pipeline fitted in {time.time() - start_time:.2f} seconds")
    return pipeline

def transform_data(pipeline, X_train, y_train, X_val, y_val):
    """Transform both train and validation data"""
    print("Transforming training data...")
    start_time = time.time()
    X_train_transformed, y_train_transformed = pipeline.transform(X_train, y_train)
    print(f"Training data transformed in {time.time() - start_time:.2f} seconds")
    
    print("Transforming validation data...")
    start_time = time.time()
    X_val_transformed, y_val_transformed = pipeline.transform(X_val, y_val)
    print(f"Validation data transformed in {time.time() - start_time:.2f} seconds")
    
    return X_train_transformed, X_val_transformed, y_train_transformed, y_val_transformed

def train_model(model, pipeline, epochs, X_train_path, y_train_path, X_val_path, y_val_path, checkpoint_dir, **params):
    """GRU-specific training loop that handles chunks
    
    Args:
        model_params: Parameters for the GRU model
        pipeline: The preprocessing pipeline
        train_data: Training data generator or initial chunk
        val_data: Validation data generator or initial chunk
        checkpoint_dir: Directory to save model checkpoints
        
    Returns:
        Tuple of (model, history) after training
    """
    from gts_challenge.order_book.data.generators import PreprocessedDataGenerator
    import os
    from pathlib import Path

    # Setup checkpoint directory
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

    # Initialize history to track metrics across chunks
    history = {
        'train_loss': [], 
        'val_loss': []
    }

    
    # Train on chunks
    print(f"Training on data chunks of size {params.get('chunk_size')}...")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_train_loss = []

        train_generator = PreprocessedDataGenerator(X_train_path, y_train_path, chunk_size=params.get('chunk_size'))
        val_generator = PreprocessedDataGenerator(X_val_path, y_val_path, chunk_size=params.get('chunk_size'))
    
        # Train on chunks
        chunk_num = 0
        for X_train, y_train in train_generator.generate_chunks(shuffle=True):
            chunk_num += 1
            print(f"Training on chunk {chunk_num}...")

            # Train on this chunk
            model, loss_value = model.fit(X_train, y_train[:, 1])
            epoch_train_loss.append(loss_value)

        # Compute average training loss for the epoch
        avg_train_loss = np.mean(epoch_train_loss)
        history['train_loss'].append(avg_train_loss)
        print(f"Average training loss for epoch {epoch + 1}: {avg_train_loss}")

        # Evaluate on validation data at the end of the epoch
        val_losses = []
        for X_val, y_val in val_generator.generate_chunks(shuffle=False):
            val_loss = model.evaluate(X_val, y_val[:, 1])
            val_losses.append(val_loss)

        # Compute average validation loss for the epoch
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        print(f"Average validation loss for epoch {epoch + 1}: {avg_val_loss}")
        
        # Save model checkpoint after each epoch
        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"gru_model_epoch_{epoch}.pkl"
            print(f"Saving model checkpoint to {checkpoint_path}")
            model.save(checkpoint_path)

    # Save final model
    if checkpoint_dir is not None:
        final_model_path = checkpoint_dir / "final_model_gru.pkl"
        print(f"Saving final model to {final_model_path}")
        model.save(final_model_path)

    return model, history
    
def evaluate_model(model, pipeline, data, **params):
    """GRU-specific evaluation code"""
    X_test, y_test = data
    return model.evaluate(X_test, y_test)

def predict(model, pipeline, X_path, **params):
    from gts_challenge.order_book.data.loaders import load_all_data

    X, _ = load_all_data(X_path)

    obs_ids = X['obs_id'].unique()

    # Transform data
    X, _ = pipeline.transform(X)

    # Make predictions
    y_pred = model.predict(X)

    result = pd.DataFrame({
            'obs_id': obs_ids,
            'prediction': y_pred
        })
    
    return result

def explain(model, pipeline, X_path, **params):
    """Explain function for GRU model, returns feature importances"""
    from gts_challenge.order_book.data.loaders import load_all_data

    X, _ = load_all_data(X_path)

    # Transform data
    X, _ = pipeline.transform(X)

    # Get feature importances using intra-observation permutation
    importance_scores = model.intra_observation_permutation_feature_importance(X)
    model.visualize_feature_importance(importance_scores)

    return {'feature_importances': importance_scores}


def api_predict(model, pipeline, X_df, **params):
    """Prediction function for GRU model used by the API.

    Args:
        model: Trained GRU model.
        pipeline: Fitted pipeline.
        X_df: DataFrame containing the input data.

    Returns:
        DataFrame with predictions.
    """
    X = X_df.copy()

    obs_ids = X['obs_id'].unique()

    # Transform data
    X, _ = pipeline.transform(X)

    # Make predictions
    y_pred = model.predict(X)

    result = pd.DataFrame({
        'obs_id': obs_ids,
        'prediction': y_pred
    })
    return result


def api_explain(model, pipeline, X_df, **params):
    """Explanation function for GRU model used by the API."""
    X = X_df.copy()

    # Transform data
    X, _ = pipeline.transform(X)

    # Get feature importances using intra-observation permutation
    importance_scores = model.intra_observation_permutation_feature_importance(X)
    model.visualize_feature_importance(importance_scores)

    return {'feature_importances': importance_scores}