# Complete workflow for GB models
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
import time
from order_book.models.gb.pipeline import GBPipeline

def preprocess_data(X_path, y_path, val_split, preprocessed_dir, chunk_size):
    """GB-specific preprocessing implementation
    
    Args:
        X_path: Path to X data file (parquet or csv)
        y_path: Path to y data file (parquet or csv)
        val_split: Ratio of validation data
        preprocessed_dir: Directory to save/load preprocessed data
        
    Returns:
        Tuple of (train_data, val_data) for model training
    """
    from order_book.data.loaders import load_all_data
    from order_book.data.data_utils import save_preprocessed_train_data, check_preprocessed_data, train_val_split

    # Setup preprocessed directory
    if preprocessed_dir is None:
        preprocessed_dir = Path("preprocessed_data")
    else:
        preprocessed_dir = Path(preprocessed_dir)
    
    preprocessed_dir.mkdir(exist_ok=True)
    
    # Define paths
    X_train_path = preprocessed_dir / "gb_train_features.h5"
    X_val_path = preprocessed_dir / "gb_val_features.h5"
    y_train_path = preprocessed_dir / "gb_y_train.parquet"
    y_val_path = preprocessed_dir / "gb_y_val.parquet"
    pipeline_path = preprocessed_dir / "gb_pipeline.pkl"

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
    """Create and fit the GB pipeline"""
    pipeline = GBPipeline()
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

def train_model(model, pipeline, X_train_path, y_train_path, X_val_path, y_val_path, **params):
    """GB-specific training loop that handles chunks
    
    Args:
        model_params: Parameters for the GB model
        pipeline: The preprocessing pipeline
        train_data: Training data generator or initial chunk
        val_data: Validation data generator or initial chunk
        checkpoint_dir: Directory to save model checkpoints
        
    Returns:
        Tuple of (model, history) after training
    """
    from order_book.data.generators import PreprocessedDataGenerator
    import os
    from pathlib import Path

    checkpoint_dir = params.get('checkpoint_dir', None)
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
    

    train_generator = PreprocessedDataGenerator(X_train_path, y_train_path, chunk_size=params.get('chunk_size'))
    val_generator = PreprocessedDataGenerator(X_val_path, y_val_path, chunk_size=params.get('chunk_size'))

    X_train, y_train = next(train_generator.generate_chunks(shuffle=True))
    X_val, y_val = next(val_generator.generate_chunks(shuffle=True))


    model, evals_result = model.fit(X_train, y_train[:, 1], X_val, y_val[:, 1])
    history['train_loss'] = evals_result['train']['multi_logloss']
    history['val_loss'] = evals_result['valid']['multi_logloss']

    # Save final model
    if checkpoint_dir is not None:
        final_model_path = checkpoint_dir / "final_model_gb.pkl"
        print(f"Saving final model to {final_model_path}")
        model.save(final_model_path)

    return model, history
    
def evaluate_model(model, pipeline, data, **params):
    """GRU-specific evaluation code"""
    X_test, y_test = data
    return model.evaluate(X_test, y_test)


def predict(X_path, model, pipeline, **params):
    from order_book.data.loaders import load_all_data

    X, _ = load_all_data(X_path)
    
    
    obs_ids = X['obs_id'].unique()

    # Transform data
    X, _ = pipeline.transform(X)

    # Make predictions
    y_pred = model.predict(X)

    importance_scores = model.get_feature_importance(X)
    del X    
    model.visualize_feature_importance(importance_scores)

    result = pd.DataFrame({
            'obs_id': obs_ids,
            'prediction': y_pred
        })
    
    return result