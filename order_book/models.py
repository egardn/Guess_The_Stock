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

# Define embedding dimensions as constants
VENUE_EMBED_DIM = 8
ACTION_EMBED_DIM = 8
TRADE_EMBED_DIM = 8
SEQ_LENGTH = 100

class OrderBookEmbeddingModel(BaseEstimator, TransformerMixin):
    """
    Neural network model for order book classification with proper embeddings
    """
    def __init__(self, n_venues, n_actions, n_categories, epochs, batch_size, learning_rate):
        self.n_venues = n_venues
        self.n_actions = n_actions
        self.n_categories = n_categories
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs  # Add this parameter to store epochs
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
        forward_gru = GRU(64, return_sequences=False)(concatenated)
        backward_gru = GRU(64, return_sequences=False, go_backwards=True)(concatenated)

        # Concatenate GRU outputs
        concat = Concatenate()([forward_gru, backward_gru])

        # Dense layers
        dense1 = Dense(64, activation='selu')(concat)
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
            epochs=self.epochs,
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

    def predict(self, X, explain=False, X_background=None, visualize=False, top_k=20):
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
        raw_preds = self.model.predict(model_inputs, batch_size=self.batch_size)
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


    # Add these methods to the OrderBookEmbeddingModel class

    # Replace the permutation_feature_importance method with this improved version

    # Updated method for batches

    def permutation_feature_importance(self, X, y=None, n_repeats=5):
        """
        Calculate permutation feature importance for each input feature.
        
        Args:
            X: Dict with model inputs
            y: True labels (optional, used for accuracy calculation)
            n_repeats: Number of times to repeat permutation for each feature
            
        Returns:
            Dictionary with importance scores for each feature group and feature
        """
        # Get baseline prediction
        model_inputs = [
            X['venue_input'], 
            X['action_input'], 
            X['trade_input'],
            X['numeric_input']
        ]
        
        baseline_preds = self.model.predict(model_inputs)
        batch_size = baseline_preds.shape[0]
        
        # Define feature groups to permute
        importance_scores = {
            'venue': {},
            'action': {},
            'trade': {},
            'numeric': {}
        }
        
        # Loop through each feature group
        
        # 1. Venue features
        for _ in range(n_repeats):
            # Create permuted copy of inputs
            permuted_inputs = [arr.copy() for arr in model_inputs]
            
            # Shuffle venue sequence for each item in the batch
            permuted_values = permuted_inputs[0].copy()
            
            # For each observation in batch, shuffle along sequence dimension
            for b in range(batch_size):
                seq_indices = np.random.permutation(permuted_values.shape[1])
                permuted_inputs[0][b] = permuted_values[b, seq_indices]
            
            # Get predictions with permuted feature
            permuted_preds = self.model.predict(permuted_inputs)
            
            # Calculate importance based on probability distribution change
            # Mean absolute difference in probabilities across batch
            importance = np.mean(np.abs(baseline_preds - permuted_preds))
            importance_scores['venue']['venue_all'] = importance_scores['venue'].get('venue_all', 0) + importance
        
        # 2. Action features
        for _ in range(n_repeats):
            permuted_inputs = [arr.copy() for arr in model_inputs]
            permuted_values = permuted_inputs[1].copy()
            
            for b in range(batch_size):
                seq_indices = np.random.permutation(permuted_values.shape[1])
                permuted_inputs[1][b] = permuted_values[b, seq_indices]
            
            permuted_preds = self.model.predict(permuted_inputs)
            importance = np.mean(np.abs(baseline_preds - permuted_preds))
            importance_scores['action']['action_all'] = importance_scores['action'].get('action_all', 0) + importance
            
        # 3. Trade features
        for _ in range(n_repeats):
            permuted_inputs = [arr.copy() for arr in model_inputs]
            permuted_values = permuted_inputs[2].copy()
            
            for b in range(batch_size):
                seq_indices = np.random.permutation(permuted_values.shape[1])
                permuted_inputs[2][b] = permuted_values[b, seq_indices]
            
            permuted_preds = self.model.predict(permuted_inputs)
            importance = np.mean(np.abs(baseline_preds - permuted_preds))
            importance_scores['trade']['trade_all'] = importance_scores['trade'].get('trade_all', 0) + importance
            
        # 4. Numeric features (permuting each feature separately)
        feature_names = ["bid", "ask", "price", "log_bid_size", "log_ask_size", "log_flux"]
        for feat_idx, feat_name in enumerate(feature_names):
            for _ in range(n_repeats):
                permuted_inputs = [arr.copy() for arr in model_inputs]
                
                # Shuffle this numeric feature for each observation in batch
                for b in range(batch_size):
                    feature_vals = permuted_inputs[3][b, :, feat_idx].copy()
                    seq_indices = np.random.permutation(feature_vals.shape[0])
                    permuted_inputs[3][b, :, feat_idx] = feature_vals[seq_indices]
                
                permuted_preds = self.model.predict(permuted_inputs)
                importance = np.mean(np.abs(baseline_preds - permuted_preds))
                importance_scores['numeric'][feat_name] = importance_scores['numeric'].get(feat_name, 0) + importance
        
        # Average importance scores across repeats
        for group in importance_scores:
            for feature in importance_scores[group]:
                importance_scores[group][feature] /= n_repeats
                
        return importance_scores

    # Update the visualization method

    def visualize_feature_importance(self, importance_scores, top_k=10, consolidated=False):
        """
        Visualize permutation feature importance results
        
        Args:
            importance_scores: Result from permutation_feature_importance()
            top_k: Number of top features to show for each group
            consolidated: If True, displays all features in a single plot
            
        Returns:
            matplotlib figure
        """
        if consolidated:
            # Create consolidated view with all features in a single plot
            # Collect all features into a single dictionary
            all_features = {}
            
            # Add venue feature
            if 'venue' in importance_scores and 'venue_all' in importance_scores['venue']:
                all_features['venue_sequence'] = importance_scores['venue']['venue_all']
            
            # Add action feature
            if 'action' in importance_scores and 'action_all' in importance_scores['action']:
                all_features['action_sequence'] = importance_scores['action']['action_all']
            
            # Add trade feature
            if 'trade' in importance_scores and 'trade_all' in importance_scores['trade']:
                all_features['trade_sequence'] = importance_scores['trade']['trade_all']
            
            # Add numeric features
            if 'numeric' in importance_scores:
                for feat, score in importance_scores['numeric'].items():
                    all_features[f'numeric_{feat}'] = score
            
            # Sort features by importance
            sorted_features = dict(sorted(all_features.items(), key=lambda x: x[1], reverse=True))
            
            # Create the visualization
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            
            # Plot horizontal bars
            y_pos = range(len(sorted_features))
            bars = ax.barh(y_pos, list(sorted_features.values()), align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(sorted_features.keys()))
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Mean Change in Probability Distribution')
            ax.set_title('Consolidated Feature Importance')
            
            # Color bars by feature group
            for i, feature_name in enumerate(sorted_features.keys()):
                if 'venue' in feature_name:
                    bars[i].set_color('skyblue')
                elif 'action' in feature_name:
                    bars[i].set_color('lightgreen')
                elif 'trade' in feature_name:
                    bars[i].set_color('salmon')
                else:
                    bars[i].set_color('lightsalmon')
            
            plt.tight_layout()
            return fig
            
        else:
            # Original visualization with 2x2 grid of subplots
            # Create figure with 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            
            # 1. Venue importance (top left)
            venue_scores = importance_scores.get('venue', {})
            if venue_scores:
                venue_names = ["Venue Sequence"]
                venue_values = [venue_scores.get('venue_all', 0)]
                
                axes[0, 0].barh(venue_names, venue_values, color='skyblue')
                axes[0, 0].set_title('Venue Feature Importance')
                axes[0, 0].set_xlabel('Mean Change in Probability Distribution')
            
            # 2. Action importance (top right)
            action_scores = importance_scores.get('action', {})
            if action_scores:
                action_names = ["Action Sequence"]
                action_values = [action_scores.get('action_all', 0)]
                
                axes[0, 1].barh(action_names, action_values, color='lightgreen')
                axes[0, 1].set_title('Action Feature Importance')
                axes[0, 1].set_xlabel('Mean Change in Probability Distribution')
            
            # 3. Trade importance (bottom left)
            trade_scores = importance_scores.get('trade', {})
            if trade_scores:
                trade_names = ["Trade Sequence"]
                trade_values = [trade_scores.get('trade_all', 0)]
                
                axes[1, 0].barh(trade_names, trade_values, color='salmon')
                axes[1, 0].set_title('Trade Feature Importance')
                axes[1, 0].set_xlabel('Mean Change in Probability Distribution')
            
            # 4. Numeric feature importance (bottom right)
            numeric_scores = importance_scores.get('numeric', {})
            if numeric_scores:
                # Sort by importance
                sorted_items = sorted(numeric_scores.items(), key=lambda x: x[1], reverse=True)
                numeric_names = [name for name, _ in sorted_items]
                numeric_values = [value for _, value in sorted_items]
                
                axes[1, 1].barh(numeric_names, numeric_values, color='lightsalmon')
                axes[1, 1].set_title('Numeric Feature Importance')
                axes[1, 1].set_xlabel('Mean Change in Probability Distribution')
            
            plt.tight_layout()
            return fig