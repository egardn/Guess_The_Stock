import numpy as np
import pandas as pd
import gc
import os
from pathlib import Path
import pickle

class PreprocessedDataGenerator:
    """
    Generator for loading preprocessed data in chunks
    """
    def __init__(self, data_path, chunk_size=1000):
        """
        Initialize the preprocessed data generator
        
        Args:
            data_path: Path to preprocessed parquet file
            chunk_size: Number of observations to load per chunk
        """
        self.data_path = data_path
        self.chunk_size = chunk_size
        
        # Get total number of observations
        self.total_obs = pd.read_parquet(data_path, columns=['obs_id']).shape[0]
        print(f"Preprocessed data contains {self.total_obs} observations")
        
    def generate_chunks(self, shuffle=True):
        """
        Generate chunks of preprocessed data
        
        Args:
            shuffle: Whether to shuffle the row indices
            
        Yields:
            Dictionary with preprocessed features and labels
        """
        # Create row indices
        indices = np.arange(self.total_obs)
        
        # Shuffle if requested
        if shuffle:
            np.random.shuffle(indices)
            
        # Process in chunks
        for i in range(0, self.total_obs, self.chunk_size):
            chunk_indices = indices[i:i + self.chunk_size]
            
            # Read the entire file first, then select rows
            df_full = pd.read_parquet(self.data_path)
            df_chunk = df_full.iloc[chunk_indices]
            
            # Extract features and labels
            X_filtered = {
                'venue_input': np.stack([pickle.loads(x) for x in df_chunk['venue_input']]),
                'action_input': np.stack([pickle.loads(x) for x in df_chunk['action_input']]),
                'trade_input': np.stack([pickle.loads(x) for x in df_chunk['trade_input']]),
                'numeric_input': np.stack([pickle.loads(x) for x in df_chunk['numeric_input']])
            }
            y_values = np.array(df_chunk['label'])
            
            yield X_filtered, y_values


def convert_x_to_parquet(csv_path, parquet_path=None, chunksize=100000, dtype=None):
    """
    Convert X data CSV file to Parquet format.
    
    Args:
        csv_path: Path to the source CSV file
        parquet_path: Path where the Parquet file will be saved (default: same name with .parquet extension)
        chunksize: Number of rows to process at once
        dtype: Data types for columns (optional)
        
    Returns:
        Path to the created Parquet file
    """
    csv_path = Path(csv_path)
    
    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet')
    else:
        parquet_path = Path(parquet_path)
        
    print(f"Converting {csv_path} to Parquet format...")
    
    # Determine dtypes if not provided
    if dtype is None:
        # Read a small sample to infer dtypes
        sample = pd.read_csv(csv_path, nrows=1000)
        
        # Apply appropriate dtypes
        dtype = {
            'obs_id': 'int32',
            # Add any other columns with specific dtypes
        }
        
        # For float columns, use float32 instead of float64 to save memory
        for col in sample.select_dtypes(include=['float64']).columns:
            dtype[col] = 'float32'
    
    # Instead of using append, concatenate all chunks and write once
    # or use a different approach depending on your data size
    
    # Option 1: For smaller files, read all at once
    df = pd.read_csv(csv_path, dtype=dtype)
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
        
    print(f"Conversion complete. Saved to {parquet_path}")
    return parquet_path

def convert_y_to_parquet(csv_path, parquet_path=None):
    """
    Convert y data CSV file to Parquet format.
    
    Args:
        csv_path: Path to the source CSV file
        parquet_path: Path where the Parquet file will be saved (default: same name with .parquet extension)
        
    Returns:
        Path to the created Parquet file
    """
    csv_path = Path(csv_path)
    
    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet')
    else:
        parquet_path = Path(parquet_path)
        
    print(f"Converting {csv_path} to Parquet format...")
    
    # y data is typically smaller, so we can load it all at once
    df = pd.read_csv(csv_path, dtype={'obs_id': 'int32', 'y': 'int8'})
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    print(f"Conversion complete. Saved to {parquet_path}")
    return parquet_path

def convert_dataset_to_parquet(x_path, y_path, output_dir=None):
    """
    Convert both X and y data CSV files to Parquet format.
    
    Args:
        x_path: Path to the X data CSV file
        y_path: Path to the y data CSV file
        output_dir: Directory where the Parquet files will be saved
            (default: same directories as input files)
            
    Returns:
        Tuple of (x_parquet_path, y_parquet_path)
    """
    x_path = Path(x_path)
    y_path = Path(y_path)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        x_parquet_path = output_dir / x_path.with_suffix('.parquet').name
        y_parquet_path = output_dir / y_path.with_suffix('.parquet').name
    else:
        x_parquet_path = x_path.with_suffix('.parquet')
        y_parquet_path = y_path.with_suffix('.parquet')
    
    # Convert X data
    x_parquet_path = convert_x_to_parquet(x_path, x_parquet_path)
    
    # Convert y data
    y_parquet_path = convert_y_to_parquet(y_path, y_parquet_path)
    
    return x_parquet_path, y_parquet_path