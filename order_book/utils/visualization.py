# Visualization functions
import matplotlib.pyplot as plt


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