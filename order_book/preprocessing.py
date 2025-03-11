import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Define sequence length as a global constant
SEQ_LENGTH = 100


class NonPositiveBidSizeFilter(BaseEstimator, TransformerMixin):
    """
    Filters out observations where any bid_size value is non-positive (≤ 0).
    This filter must be applied before log transformation in DataVectorizer.
    """
    def __init__(self):
        self.filtered_obs_count = 0
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
        
    def transform(self, X):
        # Find obs_ids with any non-positive bid_size
        obs_ids_with_nonpositive_bids = X[X['bid_size'] <= 0]['obs_id'].unique()
        
        # Count filtered observations
        self.filtered_obs_count = len(obs_ids_with_nonpositive_bids)
        if self.filtered_obs_count > 0:
            print(f"Filtered out {self.filtered_obs_count} observations with non-positive bid_size values")
        
        # Return data without the invalid obs_ids
        return X[~X['obs_id'].isin(obs_ids_with_nonpositive_bids)]

class DataVectorizer(BaseEstimator, TransformerMixin):
    """
    Transforms raw order book data into vectorized features before creating sequences.
    """
    def __init__(self):
        self.venue_mapping = None
        self.action_mapping = None
        
    def fit(self, X, y=None):
        # Extract unique values for categorical features
        self.venue_mapping = {v: i for i, v in enumerate(sorted(X['venue'].unique()))}
        self.action_mapping = {a: i for i, a in enumerate(sorted(X['action'].unique()))}
        self.n_features_in_ = X.shape[1]
        return self
        
    def transform(self, X):
        # Create a copy of the dataframe to avoid modifying the original
        X_transformed = X.copy()
        
        # Transform categorical features to indices
        X_transformed['venue_idx'] = X_transformed['venue'].map(self.venue_mapping)
        X_transformed['action_idx'] = X_transformed['action'].map(self.action_mapping)
        X_transformed['trade_idx'] = X_transformed['trade'].astype(np.int32)
        
        # Transform numeric features
        X_transformed['log_bid_size'] = np.log1p(X_transformed['bid_size'])
        X_transformed['log_ask_size'] = np.log1p(X_transformed['ask_size'])
        
        # Transform flux with sign preservation
        flux_values = X_transformed['flux'].values
        signs = np.sign(flux_values)
        log_abs_flux = np.log1p(np.abs(flux_values))
        X_transformed['log_flux'] = signs * log_abs_flux
        
        return X_transformed

class VectorizedSequenceReshaper(BaseEstimator, TransformerMixin):
    """
    Reshapes vectorized order book data into sequence tensors ready for the model.
    """
    def __init__(self, seq_length=SEQ_LENGTH):
        self.seq_length = seq_length
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        # Group by observation ID
        grouped = X.groupby('obs_id')
        n_sequences = len(grouped)
        
        # Pre-allocate arrays
        venue_data = np.zeros((n_sequences, self.seq_length), dtype=np.int32)
        action_data = np.zeros((n_sequences, self.seq_length), dtype=np.int32)
        trade_data = np.zeros((n_sequences, self.seq_length), dtype=np.int32)
        numeric_data = np.zeros((n_sequences, self.seq_length, 6), dtype=np.float32)
        
        obs_ids = []
        
        for idx, (obs_id, group) in enumerate(grouped):
            obs_ids.append(obs_id)
            seq = group.reset_index(drop=True)
            
            # Fill in categorical indices
            venue_data[idx, :len(seq)] = seq['venue_idx'].values[:self.seq_length]
            action_data[idx, :len(seq)] = seq['action_idx'].values[:self.seq_length]
            trade_data[idx, :len(seq)] = seq['trade_idx'].values[:self.seq_length]
            
            # Fill in numeric features
            numeric_data[idx, :len(seq), 0] = seq['bid'].values[:self.seq_length]
            numeric_data[idx, :len(seq), 1] = seq['ask'].values[:self.seq_length]
            numeric_data[idx, :len(seq), 2] = seq['price'].values[:self.seq_length]
            numeric_data[idx, :len(seq), 3] = seq['log_bid_size'].values[:self.seq_length]
            numeric_data[idx, :len(seq), 4] = seq['log_ask_size'].values[:self.seq_length]
            numeric_data[idx, :len(seq), 5] = seq['log_flux'].values[:self.seq_length]
        
        return {
            'venue_input': venue_data,
            'action_input': action_data,
            'trade_input': trade_data,
            'numeric_input': numeric_data
        }, obs_ids

# Define the full pipeline
def create_order_book_pipeline():
    pipeline = Pipeline([
        ('bid_size_filter', NonPositiveBidSizeFilter()),  # Add the filter BEFORE vectorizer
        ('vectorizer', DataVectorizer()),
        ('reshaper', VectorizedSequenceReshaper()),
        # The model would be added after preprocessing in the training script
    ])
    return pipeline

def visualize_preprocessing(raw_data, processed_data, obs_ids, obs_id=None):
    """
    Visualize raw data and processed tensors for a single observation
    
    Parameters:
    -----------
    raw_data : DataFrame
        Original dataframe with raw data
    processed_data : dict
        Dictionary of processed tensors
    obs_ids : list
        List of observation IDs corresponding to tensor indices
    obs_id : str or int, optional
        Specific observation ID to visualize. If None, takes the first one.
    """
    if obs_id is None:
        # Take the first observation ID
        obs_id = obs_ids[0]
    
    # Get the raw sequence for this obs_id
    raw_sequence = raw_data[raw_data['obs_id'] == obs_id].reset_index(drop=True)
    
    # Find the index of this obs_id in the processed data
    idx = obs_ids.index(obs_id)
    
    # Extract the processed tensors for this observation
    venue_indices = processed_data['venue_input'][idx]
    action_indices = processed_data['action_input'][idx]
    trade_indices = processed_data['trade_input'][idx]
    numeric_features = processed_data['numeric_input'][idx]
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    # 1. Visualize raw data (first 10 rows for readability)
    ax1.set_title(f"Raw Data for Observation {obs_id} (first 10 rows)")
    raw_data_sample = raw_sequence.head(10).copy()
    
    # Create a table-like visualization
    col_labels = raw_data_sample.columns
    row_labels = raw_data_sample.index
    
    # Convert categorical data to strings for better display
    for col in ['venue', 'action', 'side', 'trade']:
        if col in raw_data_sample.columns:
            raw_data_sample[col] = raw_data_sample[col].astype(str)
    
    ax1.axis('tight')
    ax1.axis('off')
    table = ax1.table(cellText=raw_data_sample.values, 
                     colLabels=col_labels,
                     rowLabels=row_labels, 
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    
    # 2. Visualize processed features as indices (first 10 time steps)
    ax2.set_title(f"Processed Features for Observation {obs_id} (first 10 time steps)")
    ax2.axis('tight')
    ax2.axis('off')
    
    # Prepare data for display
    display_data = np.zeros((10, 9))  # First 10 rows, 9 columns (3 categorical + 6 numeric)
    
    # Add the index data for the first 10 time steps
    for i in range(min(10, len(venue_indices))):
        display_data[i, 0] = venue_indices[i]
        display_data[i, 1] = action_indices[i]
        display_data[i, 2] = trade_indices[i]
        display_data[i, 3:] = numeric_features[i, :6]  # 6 numeric features
    
    # Define column names
    feature_names = [
        "venue_idx", "action_idx", "trade_idx", 
        "bid", "ask", "price", "log_bid_size", "log_ask_size", "log_flux",
        "side_binary", "order_id"
    ]
    
    # Round the numeric values
    display_data = np.round(display_data, 4)
    
    # Create the table
    tensor_table = ax2.table(cellText=display_data, 
                           colLabels=feature_names,
                           rowLabels=range(10), 
                           cellLoc='center',
                           loc='center')
    tensor_table.auto_set_font_size(False)
    tensor_table.set_fontsize(9)
    tensor_table.scale(1.2, 1.2)
    
    # Add note about embeddings
    ax2.text(0.5, -0.1, 
             "Note: Categorical features are represented as indices.\n"
             "Actual embeddings will be learned during model training.", 
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=12, color='darkred', bbox=dict(boxstyle="round,pad=0.5", 
                                                  fc="lightyellow", ec="orange", alpha=0.8))
    
    plt.tight_layout(pad=4.0)
    plt.show()
    
    return fig