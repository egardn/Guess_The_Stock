# Complete workflow for GB models
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
import time
from gts_challenge.order_book.models.gb.pipeline import GBPipeline
import logging

logger = logging.getLogger(__name__)

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
    from gts_challenge.order_book.data.loaders import load_all_data
    from gts_challenge.order_book.data.data_utils import save_preprocessed_train_data, check_preprocessed_data, train_val_split

    # Setup preprocessed directory
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
    from gts_challenge.order_book.data.generators import PreprocessedDataGenerator
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


def predict(model, pipeline, X_path, **params):
    from gts_challenge.order_book.data.loaders import load_all_data

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

def explain(model, pipeline, X_path, **params):
    """Explain function for GB model, returns feature importances"""
    from gts_challenge.order_book.data.loaders import load_all_data

    X, _ = load_all_data(X_path)

    # Transform data
    X, _ = pipeline.transform(X)

    importance_scores = model.get_feature_importance(X)
    model.visualize_feature_importance(importance_scores)

    return {'feature_importances': importance_scores}

def api_predict(model, pipeline, X_df, **params):
    """Prediction function for GB model used by the API.

    Args:
        model: Trained GB model.
        pipeline: Fitted pipeline.
        X_df: DataFrame containing the input data for a single sequence.

    Returns:
        DataFrame with a single prediction row.
    """
    logger.info(f"API Predict GB - Input DataFrame shape: {X_df.shape}")
    if 'obs_id' not in X_df.columns or X_df['obs_id'].nunique() != 1:
         logger.warning("API Predict GB - Input DataFrame does not seem to contain a single obs_id.")
         # Proceeding, but this might indicate an issue upstream

    obs_id = X_df['obs_id'].iloc[0] # Get obs_id from the original input

    try:
        # Transform data - Assuming pipeline.transform handles single sequence input
        # and potentially returns a dict, DataFrame, or NumPy array for the aggregated features.
        X_transformed, _ = pipeline.transform(X_df) # Pass the original DataFrame
        logger.info(f"API Predict GB - Transformed data type: {type(X_transformed)}")

        # --- FIX: Reshape data for LightGBM ---
        if isinstance(X_transformed, dict):
            # Extract scalar values from the arrays within the dictionary
            try:
                # Ensure consistent feature order if possible, though dicts are ordered in Python 3.7+
                # It's safer if the pipeline guarantees order or returns a structure with order (like DataFrame)
                feature_values = [arr.item() if isinstance(arr, np.ndarray) and arr.size == 1 else arr for arr in X_transformed.values()]
                X_reshaped = np.array(feature_values).reshape(1, -1) # Reshape to (1, n_features)
                logger.info(f"API Predict GB - Reshaped data from dict to shape: {X_reshaped.shape}")
            except Exception as reshape_err:
                 logger.error(f"API Predict GB - Error reshaping dict data: {reshape_err}", exc_info=True)
                 raise ValueError("Failed to reshape transformed dictionary data.") from reshape_err
        elif isinstance(X_transformed, pd.DataFrame):
            X_reshaped = X_transformed.values # Use .values to get NumPy array
            if X_reshaped.ndim == 1: # Ensure 2D if DataFrame had only one row
                X_reshaped = X_reshaped.reshape(1, -1)
            logger.info(f"API Predict GB - Using DataFrame values, shape: {X_reshaped.shape}")
        elif isinstance(X_transformed, np.ndarray):
            if X_transformed.ndim == 1:
                X_reshaped = X_transformed.reshape(1, -1)
                logger.info(f"API Predict GB - Reshaped 1D NumPy array to shape: {X_reshaped.shape}")
            elif X_transformed.ndim == 2:
                 X_reshaped = X_transformed # Already 2D
                 logger.info(f"API Predict GB - Using 2D NumPy array, shape: {X_reshaped.shape}")
            else:
                 logger.error(f"API Predict GB - Transformed NumPy array has unexpected dimensions: {X_transformed.ndim}")
                 raise ValueError(f"Transformed NumPy array has unexpected dimensions: {X_transformed.ndim}")
        else:
            logger.error(f"API Predict GB - Unexpected data type after transform: {type(X_transformed)}")
            raise TypeError(f"Unexpected data type after pipeline transform: {type(X_transformed)}")
        # --- END FIX ---

        # Ensure X_reshaped is now a 2D array before prediction
        if not isinstance(X_reshaped, np.ndarray) or X_reshaped.ndim != 2:
             logger.error(f"API Predict GB - Data is not a 2D NumPy array before prediction. Type: {type(X_reshaped)}, Shape: {getattr(X_reshaped, 'shape', 'N/A')}")
             raise ValueError("Data could not be prepared as a 2D NumPy array for prediction.")

        # Make prediction using the reshaped data
        # Use predict_proba to get probabilities, then argmax for the final class
        y_pred_proba = model.predict_proba(X_reshaped)
        y_pred = np.argmax(y_pred_proba, axis=1) # Get the class index

        # Create a result DataFrame matching the expected output format for a single prediction
        result_df = pd.DataFrame({'obs_id': [obs_id], 'prediction': y_pred})

        logger.info(f"API Predict GB - Prediction successful for obs_id {obs_id}. Result shape: {result_df.shape}")
        return result_df

    except Exception as e:
        logger.error(f"Error in GB api_predict for obs_id {obs_id}: {e}", exc_info=True)
        # Re-raise the exception to be caught by the factory/endpoint
        raise


def api_explain(model, pipeline, X_df, **params):
    """Simplified explanation function for GB model used by the API."""
    logger.info(f"API Explain GB - Input DataFrame shape: {X_df.shape}")

    # Ensure the input DataFrame contains a single obs_id
    if 'obs_id' not in X_df.columns or X_df['obs_id'].nunique() != 1:
        logger.warning("API Explain GB - Input DataFrame does not seem to contain a single obs_id.")

    obs_id = X_df['obs_id'].iloc[0]

    try:
        # Transform data using the pipeline
        X_transformed, _ = pipeline.transform(X_df)
        logger.info(f"API Explain GB - Transformed data type: {type(X_transformed)}")

        # Convert transformed data to a 2D NumPy array
        X_reshaped = np.array(list(X_transformed.values())).reshape(1, -1)
        logger.info(f"API Explain GB - Reshaped data to shape: {X_reshaped.shape}")

        # Get feature importances from the model
        feature_importances = model.feature_importances_

        # Ensure feature names are available
        feature_names = list(X_transformed.keys())
        if len(feature_names) != len(feature_importances):
            logger.warning("API Explain GB - Mismatch between feature names and importances.")
            raise ValueError("Feature names and importances length mismatch.")

        # Create a dictionary of feature importances
        importance_scores = dict(zip(feature_names, feature_importances))
        logger.info(f"API Explain GB - Explanation successful for obs_id {obs_id}.")

        return {"feature_importances": importance_scores}

    except Exception as e:
        logger.error(f"Error in GB api_explain for obs_id {obs_id}: {e}", exc_info=True)
        raise