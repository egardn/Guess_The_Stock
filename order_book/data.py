import numpy as np
import pandas as pd
import gc
import os
from pathlib import Path

class OrderBookDataGenerator:
    """
    Optimized data generator for order book data with 100-row observations.
    Only handles Parquet files for X data.
    """
    
    def __init__(self, X_path, y_path, chunk_size=100):
        """
        Initialize the data generator.
        
        Args:
            X_path: Path to X data (parquet file only)
            y_path: Path to y data (CSV file)
            chunk_size: Number of observations to load at once
        """
        self.X_path = Path(X_path)
        self.y_path = Path(y_path)
        self.chunk_size = chunk_size
        
        # Ensure X_path is a parquet file
        if self.X_path.suffix != '.parquet':
            raise ValueError("X_path must be a parquet file")
        
        # Load y data completely
        self.y_data = pd.read_csv(self.y_path)
        
        # Get observation information
        self.events_per_obs = 100  # Known from problem description
        
        # Load observation IDs from parquet
        self.all_obs_ids = pd.read_parquet(
            self.X_path, 
            columns=['obs_id']
        )['obs_id'].unique()
        
        # Current position in the observation IDs list
        self.current_pos = 0
        
    def generate_chunks(self, shuffle=True):
        """
        Generator that yields chunks of data for training.
        
        Args:
            shuffle: Whether to shuffle the observation IDs
            
        Yields:
            Tuple of (X_chunk, y_chunk, chunk_obs_ids)
        """
        # Reset position
        self.current_pos = 0
        
        # Get all observation IDs
        obs_ids = np.array(self.all_obs_ids)
        
        # Shuffle observations if requested
        if shuffle:
            np.random.shuffle(obs_ids)
        
        # Process in chunks of observations
        while self.current_pos < len(obs_ids):
            # Get obs_ids for this chunk
            chunk_obs_ids = obs_ids[self.current_pos:self.current_pos + self.chunk_size]
            self.current_pos += self.chunk_size
            
            if len(chunk_obs_ids) == 0:
                break
            
            # Read X data for these observation IDs from parquet
            X_chunk = pd.read_parquet(
                self.X_path, 
                filters=[('obs_id', 'in', chunk_obs_ids.tolist())]
            )
            
            # Get corresponding y data
            y_chunk = self.y_data[self.y_data['obs_id'].isin(chunk_obs_ids)]
            
            # Clean up memory
            gc.collect()
            
            yield X_chunk, y_chunk, chunk_obs_ids


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