# GRU model class definition

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Concatenate, Dropout, Multiply
from tensorflow.keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # Add SHAP import
import lightgbm as lgbm  # or import xgboost as xgb
import pandas as pd
from gts_challenge.order_book.base.model_interface import ModelInterface
# Importez les dépendances nécessaires


# Define embedding dimensions as constants
VENUE_EMBED_DIM = 8
ACTION_EMBED_DIM = 8
TRADE_EMBED_DIM = 8
SEQ_LENGTH = 100

class OrderBookGRUModel(ModelInterface, BaseEstimator, TransformerMixin):
    """
    Neural network model for order book classification with proper embeddings
    """
    def __init__(self, n_venues, n_actions, n_categories, batch_size, learning_rate, gru_units, dense_units, gru_dropout):
        self.n_venues = n_venues
        self.n_actions = n_actions
        self.n_categories = n_categories
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gru_units = gru_units
        self.dense_units = dense_units  
        self.gru_dropout = gru_dropout
        self.model = None

    def create_model(self):
        # Create input layers for each feature type
        venue_input = Input(shape=(SEQ_LENGTH,), name='venue_input', dtype='int32')
        action_input = Input(shape=(SEQ_LENGTH,), name='action_input', dtype='int32')
        trade_input = Input(shape=(SEQ_LENGTH,), name='trade_input', dtype='int32')
        numeric_input = Input(shape=(SEQ_LENGTH, 6), name='numeric_input')  # Now 6 features
        
        # Create embedding layers
        venue_embedding = Embedding(self.n_venues, VENUE_EMBED_DIM, name='venue_embedding')(venue_input)
        action_embedding = Embedding(self.n_actions, ACTION_EMBED_DIM, name='action_embedding')(action_input)
        trade_embedding = Embedding(2, TRADE_EMBED_DIM, name='trade_embedding')(trade_input)
        
        # Concatenate embeddings with numeric features
        concatenated = Concatenate(axis=2)([
            venue_embedding,        # (batch, seq_len, 8)
            action_embedding,       # (batch, seq_len, 8)
            trade_embedding,        # (batch, seq_len, 8)
            numeric_input           # (batch, seq_len, 6)
        ])  # Result: (batch, seq_len, 30)
        
        # GRU layers - forward and backward
        forward_gru = GRU(self.gru_units,
                          return_sequences=False,
                          dropout=self.gru_dropout
                          )(concatenated)
        backward_gru = GRU(self.gru_units,
                           return_sequences=False,
                           go_backwards=True,
                           dropout=self.gru_dropout
                           )(concatenated)

        # Concatenate GRU outputs
        concat = Concatenate()([forward_gru, backward_gru])

        # Dense layers
        dense1 = Dense(self.dense_units, activation='selu')(concat)
        output_layer = Dense(self.n_categories, activation='softmax')(dense1)

        # Create and compile model
        model = Model(
            inputs=[
                venue_input, 
                action_input, 
                trade_input, 
                numeric_input
            ],
            outputs=output_layer
        )
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )

        return model
    
    def fit(self, X, y):
        """Fit the model to the data."""
        # Initialize model if not already done
        if self.model is None:
            self.model = self.create_model()
            self.model.summary()
        
        # Create model inputs - with only 4 required inputs
        model_inputs = [
            X['venue_input'], 
            X['action_input'], 
            X['trade_input'],
            X['numeric_input']
        ]
        
        # Train model
        history = self.model.fit(
            model_inputs, y,
            batch_size=self.batch_size,
            epochs=1,
            verbose=1
        )

        loss_value = history.history['loss'][0] if history.history['loss'] else None
        
        return self, loss_value

    def transform(self, X):
        # Convert inputs to correct format with same keys as fit() method
        model_inputs = [
            X['venue_input'], 
            X['action_input'], 
            X['trade_input'],
            X['numeric_input']
        ]
        
        # Return predictions
        return self.model.predict(model_inputs, batch_size=self.batch_size)

    def predict(self, X, explain=False, X_background=None, visualize=False, top_k=20, verbose=1):
        """
        Predict class labels with optional SHAP explanation.
        
        Args:
            X: Dictionary of input features
            explain: If True, return SHAP explanation along with predictions
            X_background: Background data for SHAP (subset of training data), 
                        required if explain=True
            visualize: If True, also generate visualization (requires explain=True)
            top_k: Number of top features to show in visualization
            
        Returns:
            If explain=False: Class predictions array
            If explain=True: Tuple of (predictions, shap_explanation)
            If explain=True and visualize=True: Tuple of (predictions, shap_explanation, figure)
        """
        # Convert inputs to correct format
        model_inputs = [
            X['venue_input'], 
            X['action_input'], 
            X['trade_input'],
            X['numeric_input']
        ]
        
        # Get raw predictions and class predictions
        raw_preds = self.model.predict(model_inputs, batch_size=self.batch_size, verbose=verbose)
        predictions = np.argmax(raw_preds, axis=1)

        
        # Return just the predictions if we don't need explanations
        if not explain:
            return predictions
        
        # Get SHAP explanations
        shap_explanation = self.explain_with_shap(X, X_background)
        
        if not visualize:
            return predictions, shap_explanation
        
        # Generate visualization
        fig = self.visualize_shap(shap_explanation, top_k=top_k)
        return predictions, shap_explanation, fig

    def evaluate(self, X, y):
        """Evaluate the model on validation data."""
        # Convert inputs to correct format
        model_inputs = [
            X['venue_input'], 
            X['action_input'], 
            X['trade_input'],
            X['numeric_input']
        ]
        
        # Evaluate and return loss
        eval_results = self.model.evaluate(model_inputs, y, 
                                        batch_size=self.batch_size,
                                        verbose=0)
        
        # Return loss value (first metric)
        return eval_results[0]

    def save(self, path: str) -> None:
        """Save the model to a file."""
        import pickle
        
        # Pickle the entire instance
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        return

    @classmethod
    def load(cls, path: str) -> 'ModelInterface':
        """Load the model from a file."""
        import pickle
        import os
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the pickled instance
        with open(path, 'rb') as f:
            model_instance = pickle.load(f)
        
        return model_instance
    
    def get_feature_importance(self, X, option="classical", n_repeats=5, k=None):
        """
        Calculate feature importance using the specified method.

        Args:
            X: Dict with model inputs
            option: 'classical' or 'intra' for different permutation methods
            n_repeats: Number of times to repeat permutation for each feature
            k: Number of observations to use (if None, use all observations)

        Returns:
            Flat dictionary with importance scores (Python floats) for each feature.
        """
        print(f"Calculating feature importance using method: {option}")
        if option == 'classical':
            return self.get_classical_permutation_feature_importance(X, n_repeats, k)
        elif option == 'intra':
            return self.intra_observation_permutation_feature_importance(X, n_repeats, k)
        else:
            # Raise an error for invalid option
            raise ValueError(f"Invalid feature importance option: {option}. Choose 'classical' or 'intra'.")

    def get_classical_permutation_feature_importance(self, X, n_repeats=5, k=None):
        """
        Calculate classical permutation feature importance by shuffling features across observations.
        
        Args:
            X: Dict with model inputs
            n_repeats: Number of times to repeat permutation for each feature
            k: Number of observations to use (if None, use all observations)
            
        Returns:
            Dictionary with importance scores for each feature group and feature
        """
        # Limit to first k observations if specified
        if k is not None:
            X_subset = {key: X[key][:k] for key in X}
        else:
            X_subset = X
        
        # Get baseline prediction
        model_inputs = [
            X_subset['venue_input'], 
            X_subset['action_input'], 
            X_subset['trade_input'],
            X_subset['numeric_input']
        ]
        
        baseline_preds = self.model.predict(model_inputs, verbose=0)
        batch_size = baseline_preds.shape[0]
        
        # Initialize importance scores dictionary
        importance_scores = {
            'venue': {},
            'action': {},
            'trade': {},
            'numeric': {}
        }
        
        # Define feature groups configuration
        feature_groups = [
            # For sequence features (shuffle entire sequences across observations)
            {'name': 'venue', 'input_idx': 0, 'feature_key': 'venue_all'},
            {'name': 'action', 'input_idx': 1, 'feature_key': 'action_all'},
            {'name': 'trade', 'input_idx': 2, 'feature_key': 'trade_all'},
            # For numeric features (shuffle each feature individually across observations)
            {'name': 'numeric', 'input_idx': 3, 'features': [
                "bid", "ask", "price", "log_bid_size", "log_ask_size", "log_flux"
            ]}
        ]
        
        # Process all feature groups
        for group in feature_groups:
            group_name = group['name']
            input_idx = group['input_idx']
            
            if 'feature_key' in group:
                # Handle sequence features (venue, action, trade)
                feature_key = group['feature_key']
                
                for _ in range(n_repeats):
                    permuted_inputs = [arr.copy() for arr in model_inputs]
                    
                    # Get the array to shuffle
                    values = permuted_inputs[input_idx].copy()
                    
                    # Shuffle entire sequences between observations
                    obs_indices = np.random.permutation(batch_size)
                    permuted_inputs[input_idx] = values[obs_indices]
                    
                    # Get predictions with shuffled data
                    permuted_preds = self.model.predict(permuted_inputs, verbose=0)
                    
                    # Calculate importance as mean absolute difference in predictions
                    importance = np.mean(np.abs(baseline_preds - permuted_preds))
                    importance_scores[group_name][feature_key] = importance_scores[group_name].get(feature_key, 0) + importance
            else:
                # Handle numeric features (shuffle each feature separately)
                for feat_idx, feat_name in enumerate(group['features']):
                    for _ in range(n_repeats):
                        permuted_inputs = [arr.copy() for arr in model_inputs]
                        
                        # Create shuffled version of this specific feature across observations
                        # Extract the feature values at this index for all observations
                        for time_step in range(permuted_inputs[input_idx].shape[1]):  # For each time step
                            # Get values for this feature at this time step across all observations
                            feature_vals = permuted_inputs[input_idx][:, time_step, feat_idx].copy()
                            
                            # Shuffle these values across observations
                            np.random.shuffle(feature_vals)
                            
                            # Put shuffled values back
                            permuted_inputs[input_idx][:, time_step, feat_idx] = feature_vals
                        
                        # Get predictions with shuffled data
                        permuted_preds = self.model.predict(permuted_inputs, verbose=0)
                        
                        # Calculate importance
                        importance = np.mean(np.abs(baseline_preds - permuted_preds))
                        importance_scores[group_name][feat_name] = importance_scores[group_name].get(feat_name, 0) + importance
        
        # Average importance scores across repeats
        for group in importance_scores:
            for feature in importance_scores[group]:
                importance_scores[group][feature] /= n_repeats

        # Flatten the dictionary and convert scores to float
        flat_importance_scores = {}
        if 'venue' in importance_scores and 'venue_all' in importance_scores['venue']:
            flat_importance_scores['venue_sequence'] = float(importance_scores['venue']['venue_all'])
        if 'action' in importance_scores and 'action_all' in importance_scores['action']:
            flat_importance_scores['action_sequence'] = float(importance_scores['action']['action_all'])
        if 'trade' in importance_scores and 'trade_all' in importance_scores['trade']:
            flat_importance_scores['trade_sequence'] = float(importance_scores['trade']['trade_all'])
        if 'numeric' in importance_scores:
            for feat, score in importance_scores['numeric'].items():
                flat_importance_scores[f'numeric_{feat}'] = float(score)

        return flat_importance_scores

    def intra_observation_permutation_feature_importance(self, X, n_repeats=5, k=None):
        """
        Calculate intra-observation permutation feature importance.
        Returns a flat dictionary with importance scores as percentages of the total importance.
        """
        # Limit to first k observations if specified
        if k is not None:
            X_subset = {key: X[key][:k] for key in X}
        else:
            X_subset = X

        # Get baseline prediction
        model_inputs = [
            X_subset['venue_input'],
            X_subset['action_input'],
            X_subset['trade_input'],
            X_subset['numeric_input']
        ]
        baseline_preds = self.model.predict(model_inputs, verbose=0)
        batch_size = baseline_preds.shape[0]

        # Initialize importance scores dictionary (for accumulating absolute scores)
        absolute_importance_scores = {
            'venue': {},
            'action': {},
            'trade': {},
            'numeric': {}
        }

        # Define feature groups configuration
        feature_groups = [
            # For sequence features (entire array permutation)
            {'name': 'venue', 'input_idx': 0, 'feature_key': 'venue_all'},
            {'name': 'action', 'input_idx': 1, 'feature_key': 'action_all'},
            {'name': 'trade', 'input_idx': 2, 'feature_key': 'trade_all'},
            # For numeric features (individual feature permutation)
            {'name': 'numeric', 'input_idx': 3, 'feature_key': 'bid', 'numeric_idx': 0},
            {'name': 'numeric', 'input_idx': 3, 'feature_key': 'ask', 'numeric_idx': 1},
            {'name': 'numeric', 'input_idx': 3, 'feature_key': 'price', 'numeric_idx': 2},
            {'name': 'numeric', 'input_idx': 3, 'feature_key': 'log_bid_size', 'numeric_idx': 3},
            {'name': 'numeric', 'input_idx': 3, 'feature_key': 'log_ask_size', 'numeric_idx': 4},
            {'name': 'numeric', 'input_idx': 3, 'feature_key': 'log_flux', 'numeric_idx': 5},
        ]

        # Calculate importance for each feature group
        for group_config in feature_groups:
            group_name = group_config['name']
            input_idx = group_config['input_idx']
            feat_name = group_config['feature_key']
            numeric_idx = group_config.get('numeric_idx') # None for sequence features

            accumulated_importance = 0.0
            for _ in range(n_repeats):
                permuted_inputs = [inp.copy() for inp in model_inputs] # Create a fresh copy for permutation

                # Permute within each observation
                permuted_feature = permuted_inputs[input_idx].copy()
                for i in range(batch_size):
                    if numeric_idx is None: # Permute entire sequence feature (venue, action, trade)
                        np.random.shuffle(permuted_feature[i])
                    else: # Permute specific numeric feature across time steps
                        np.random.shuffle(permuted_feature[i, :, numeric_idx])

                permuted_inputs[input_idx] = permuted_feature

                # Predict with permuted data
                permuted_preds = self.model.predict(permuted_inputs, verbose=0)

                # Calculate importance (mean absolute difference in probability distributions)
                # Using KL divergence or other metrics is also possible
                importance = np.mean(np.sum(np.abs(baseline_preds - permuted_preds), axis=-1))
                accumulated_importance += importance

            # Store the averaged absolute importance
            absolute_importance_scores[group_name][feat_name] = accumulated_importance / n_repeats


        # --- START CHANGE: Convert absolute scores to percentages ---
        # Flatten the dictionary first
        flat_absolute_scores = {}
        if 'venue' in absolute_importance_scores and 'venue_all' in absolute_importance_scores['venue']:
            flat_absolute_scores['venue_sequence'] = absolute_importance_scores['venue']['venue_all']
        if 'action' in absolute_importance_scores and 'action_all' in absolute_importance_scores['action']:
            flat_absolute_scores['action_sequence'] = absolute_importance_scores['action']['action_all']
        if 'trade' in absolute_importance_scores and 'trade_all' in absolute_importance_scores['trade']:
            flat_absolute_scores['trade_sequence'] = absolute_importance_scores['trade']['trade_all']
        if 'numeric' in absolute_importance_scores:
            for feat, score in absolute_importance_scores['numeric'].items():
                flat_absolute_scores[f'numeric_{feat}'] = score # Keep original name structure for clarity

        # Calculate total absolute importance
        total_importance = sum(flat_absolute_scores.values())

        # Calculate percentages
        percentage_importance_scores = {}
        if total_importance > 1e-10: # Avoid division by zero or near-zero
            for feature, score in flat_absolute_scores.items():
                proportion = score / total_importance
                percentage_importance_scores[feature] = float(round(proportion, 4))
        else:
            # If total importance is zero, assign zero percentage to all
            for feature in flat_absolute_scores:
                percentage_importance_scores[feature] = 0.0

        return percentage_importance_scores

    # Update the visualization method

    def visualize_feature_importance(self, importance_scores, top_k=10):
        """
        Visualize permutation feature importance results (handles flat percentage dictionary).

        Args:
            importance_scores: Flat dictionary of feature importance percentages.
            top_k: Number of top features to show.

        Returns:
            matplotlib figure
        """
        # Check if the input is the flat percentage dictionary
        if not isinstance(importance_scores, dict) or not all(isinstance(v, (float, np.floating)) for v in importance_scores.values()):
             print("Warning: visualize_feature_importance expected a flat dictionary of importance percentages.")
             # Attempt to handle the old nested structure for backward compatibility (optional)
             # Or simply return an empty figure or raise an error
             # For now, let's assume the flat structure is passed
             all_features = {} # Or handle old structure if needed
        else:
             # Directly use the flat dictionary
             all_features = importance_scores.copy()


        # Sort features by importance
        # Ensure values are numeric before sorting
        valid_features = {k: v for k, v in all_features.items() if isinstance(v, (int, float, np.number))}
        sorted_features = dict(sorted(valid_features.items(), key=lambda item: item[1], reverse=True))

        # Take top K features
        top_features = dict(list(sorted_features.items())[:top_k])

        if not top_features:
             print("No features to plot.")
             return plt.figure() # Return empty figure

        # Create the visualization
        fig = plt.figure(figsize=(12, max(6, len(top_features) * 0.5))) # Adjust height based on number of features
        ax = fig.add_subplot(111)

        # Plot horizontal bars
        y_pos = np.arange(len(top_features))
        feature_names = list(top_features.keys())
        scores = list(top_features.values())

        bars = ax.barh(y_pos, scores, align='center')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance Percentage') # Updated label
        ax.set_title(f'Top {len(top_features)} Feature Importances (Intra-Observation Permutation)') # Updated title

        # Add values on bars
        for i, v in enumerate(scores):
            ax.text(v + (max(scores) * 0.01), i, f" {v:.3f}", va='center', color='grey')


        # Color bars by feature group (heuristic based on name)
        for i, feature_name in enumerate(feature_names):
            if 'venue' in feature_name:
                bars[i].set_color('skyblue')
            elif 'action' in feature_name:
                bars[i].set_color('lightgreen')
            elif 'trade' in feature_name:
                bars[i].set_color('salmon')
            elif 'numeric' in feature_name:
                bars[i].set_color('lightcoral') # Example color for numeric
            else:
                bars[i].set_color('lightgrey') # Default

        plt.tight_layout()
        return fig
            
        