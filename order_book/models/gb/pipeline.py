# Pipeline specific to Gradient Boosting preprocessing
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from order_book.base.pipeline_interface import PipelineInterface
from order_book.data.preprocessors import NonPositiveBidSizeFilter, CategoricalEncoder, DataVectorizer, VectorizedSequenceReshaper

class GBPipeline(PipelineInterface):
    def __init__(self):
        # Create a single pipeline that handles all transformations
        self.filter = NonPositiveBidSizeFilter()
        
        self.pipeline = Pipeline([
            ('cat_encoder', CategoricalEncoder()),
            ('vectorizer', DataVectorizer()),
            ('reshaper', VectorizedSequenceReshaper()),
            ('feature_extractor', GBFeatureExtractor())
            ])

    def fit(self, X, y=None):
        """Fits the pipeline to the data"""
        self.filter.fit(X)
        self.pipeline.fit(X)
        return self
        
    def transform(self, X=None, y=None):
        """Transforms data through the unified pipeline"""
        X_filtered, y_filtered = self.filter.transform(X, y)
        
        return self.pipeline.transform(X_filtered), y_filtered
    
    def fit_transform(self, X, y=None):
        """Fits and transforms the data through the unified pipeline"""
        self.filter.fit(X)
        X_filtered, y_filtered = self.filter.transform(X, y)
        
        return self.pipeline.fit_transform(X_filtered), y_filtered
        
    def save(self, path: str) -> None:
        """Saves the pipeline to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'GRUPipeline':
        """Loads a pipeline from disk"""
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"Pipeline loaded from {path}")
        return pipeline

class GBFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transforms sequence data into 15 scale-independent financial time series features
    suitable for gradient boosting models.
    """
    def __init__(self):
        self.n_features_in_ = None
        self.numeric_feature_names = ['bid', 'ask', 'price', 'log_bid_size', 'log_ask_size', 'log_flux']
        self.scaler = None
        self.feature_names = None
        
    def fit(self, X, y=None):
        if not isinstance(X, dict):
            raise ValueError("X should be a dictionary of input sequences")
            
        self.n_features_in_ = len(X)
        
        # Create feature dict from X to learn scaling parameters
        features = self._extract_features(X)
        
        # Store feature names
        self.feature_names = list(features.keys())
        
        # Fit scaler on training data
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        
        # Convert features to array for scaling
        feature_array = np.column_stack([features[k] for k in self.feature_names])
        self.scaler.fit(feature_array)
        
        return self
    
    def _extract_features(self, X):
        # Get sequence data
        action_data = X['action_input']    # (n_sequences, seq_length)
        numeric_data = X['numeric_input']  # (n_sequences, seq_length, 6)
        
        features = {}
        
        # --- 1. Price trend features ---
        price_data = numeric_data[:, :, 2]  # price is at index 2
        
        # Calculate returns (percentage changes) instead of raw prices
        returns = np.zeros_like(price_data)
        returns[:, 1:] = (price_data[:, 1:] - price_data[:, :-1]) / (price_data[:, :-1] + 1e-10)
        
        # Price trend ratio (direction of price movement)
        up_moves = np.sum(returns > 0, axis=1)
        down_moves = np.sum(returns < 0, axis=1)
        total_moves = up_moves + down_moves + 1e-10
        features['price_trend_ratio'] = (up_moves - down_moves) / total_moves
        
        # Return volatility (normalized)
        features['return_volatility'] = np.std(returns, axis=1)
        
        # Normalized price range
        mean_price = np.mean(price_data, axis=1)
        price_range = np.max(price_data, axis=1) - np.min(price_data, axis=1)
        features['norm_price_range'] = price_range / (mean_price + 1e-10)
        
        # Relative price momentum (percentage change from start to end)
        features['rel_price_momentum'] = (price_data[:, -1] - price_data[:, 0]) / (price_data[:, 0] + 1e-10)
        
        # Return autocorrelation (lag-1)
        n_samples = returns.shape[0]
        autocorr = np.zeros(n_samples)
        for i in range(n_samples):
            if np.std(returns[i, 1:]) > 0 and np.std(returns[i, :-1]) > 0:
                autocorr[i] = np.corrcoef(returns[i, 1:], returns[i, :-1])[0, 1]
        features['return_autocorr'] = autocorr
        
        # --- 2. Bid-ask spread features ---
        bid_data = numeric_data[:, :, 0]  # bid is at index 0
        ask_data = numeric_data[:, :, 1]  # ask is at index 1
        spread = ask_data - bid_data
        mid_price = (bid_data + ask_data) / 2
        
        # Relative spread
        rel_spread = spread / (mid_price + 1e-10)
        features['mean_rel_spread'] = np.mean(rel_spread, axis=1)
        
        # Spread volatility (normalized)
        features['rel_spread_volatility'] = np.std(rel_spread, axis=1) / (np.mean(rel_spread, axis=1) + 1e-10)
        
        # Spread trend (is spread widening or narrowing?)
        spread_changes = np.diff(rel_spread, axis=1)
        features['spread_trend'] = np.sum(spread_changes > 0, axis=1) / (np.sum(np.abs(spread_changes) > 0, axis=1) + 1e-10)
        
        # --- 3. Volume features ---
        log_bid_size = numeric_data[:, :, 3]  # log_bid_size at index 3
        log_ask_size = numeric_data[:, :, 4]  # log_ask_size at index 4
        
        # Volume imbalance and its stability
        volume_imbalance = log_bid_size - log_ask_size
        features['log_volume_imbalance'] = np.mean(volume_imbalance, axis=1)
        features['volume_imbalance_stability'] = np.std(volume_imbalance, axis=1)
        
        # Total volume trend
        total_volume = log_bid_size + log_ask_size
        volume_changes = np.diff(total_volume, axis=1)
        features['volume_trend'] = np.sum(volume_changes > 0, axis=1) / (np.sum(np.abs(volume_changes) > 0, axis=1) + 1e-10)
        
        # --- 4. Market activity features ---
        # Action transition rate (normalized)
        transitions = np.sum(action_data[:, 1:] != action_data[:, :-1], axis=1)
        seq_length = action_data.shape[1]
        features['action_transition_rate'] = transitions / (seq_length - 1 + 1e-10)
        
        # Action diversity (unique actions / sequence length)
        unique_actions = np.zeros(action_data.shape[0])
        for i in range(action_data.shape[0]):
            unique_actions[i] = len(np.unique(action_data[i]))
        features['action_diversity'] = unique_actions / seq_length
        
        # Flux features (already log-transformed)
        flux = numeric_data[:, :, 5]  # log_flux at index 5
        features['mean_log_flux'] = np.mean(flux, axis=1)
        
        # Normalized flux volatility
        features['norm_flux_volatility'] = np.std(flux, axis=1) / (np.abs(np.mean(flux, axis=1)) + 1e-10)
        
        return features
        
    def transform(self, X):
        # Extract features
        features = self._extract_features(X)
        
        # Handle missing features
        for name in self.feature_names:
            if name not in features:
                features[name] = np.zeros(next(iter(features.values())).shape[0])
        
        # Only keep features that were seen during training
        features = {k: features[k] for k in self.feature_names if k in features}
        
        # Convert to array for scaling
        feature_array = np.column_stack([features[k] for k in self.feature_names])
        
        # Apply scaling
        scaled_features = self.scaler.transform(feature_array)
        
        # Convert back to dictionary
        result = {}
        for i, name in enumerate(self.feature_names):
            result[name] = scaled_features[:, i]
            
        return result