# Pipeline specific to Gradient Boosting preprocessing
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from gts_challenge.order_book.base.pipeline_interface import PipelineInterface
from gts_challenge.order_book.data.preprocessors import NonPositiveBidSizeFilter, CategoricalEncoder, DataVectorizer, VectorizedSequenceReshaper

class GBPipeline(PipelineInterface):
    def __init__(self, gb_features='all'):
        # Create a single pipeline that handles all transformations
        self.filter = NonPositiveBidSizeFilter()
        
        self.pipeline = Pipeline([
            ('cat_encoder', CategoricalEncoder()),
            ('vectorizer', DataVectorizer()),
            ('reshaper', VectorizedSequenceReshaper()),
            ('feature_extractor', GBFeatureExtractor(features_to_extract=gb_features))
            ])

    def fit(self, X, y=None):
        """Fits the pipeline to the data"""
        X_filterd, _ = self.filter.fit_transform(X)
        self.pipeline.fit(X_filtered, y)
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
        
        # --- Compatibility Check ---
        # Check if the loaded feature extractor has the new attributes.
        # If not, assume it's an old version and set defaults.
        feature_extractor_step = pipeline.pipeline.named_steps.get('feature_extractor')
        if feature_extractor_step and isinstance(feature_extractor_step, GBFeatureExtractor):
            if not hasattr(feature_extractor_step, 'features_to_extract'):
                 print("Warning: Loaded pipeline uses an older GBFeatureExtractor version. Setting 'features_to_extract' to 'all'.")
                 feature_extractor_step.features_to_extract = 'all'
            if not hasattr(feature_extractor_step, '_all_feature_calculators'):
                 # If the calculator map is missing, it's harder to recover.
                 # Might need to re-initialize parts or raise an error.
                 # For now, just warn. The fit method logic might handle it.
                 print("Warning: Loaded GBFeatureExtractor missing '_all_feature_calculators'. May cause issues.")
            # Ensure _selected_feature_names is initialized based on features_to_extract
            if not hasattr(feature_extractor_step, '_selected_feature_names') or not feature_extractor_step._selected_feature_names:
                 if feature_extractor_step.features_to_extract == 'all' and hasattr(feature_extractor_step, '_all_feature_calculators'):
                     feature_extractor_step._selected_feature_names = sorted(list(feature_extractor_step._all_feature_calculators.keys()))
                 elif isinstance(feature_extractor_step.features_to_extract, list) and hasattr(feature_extractor_step, '_all_feature_calculators'):
                      feature_extractor_step._selected_feature_names = sorted([f for f in feature_extractor_step.features_to_extract if f in feature_extractor_step._all_feature_calculators])
                 else:
                      feature_extractor_step._selected_feature_names = [] # Default to empty if cannot determine

        # Also update the __init__ parameter if the outer GBPipeline is old
        if not hasattr(pipeline, 'gb_features'):
             # Infer from the step if possible, otherwise default
             if feature_extractor_step and hasattr(feature_extractor_step, 'features_to_extract'):
                 pipeline.gb_features = feature_extractor_step.features_to_extract
             else:
                 print("Warning: Loaded GBPipeline missing 'gb_features' attribute. Defaulting to 'all'.")
                 pipeline.gb_features = 'all'


        print(f"Pipeline loaded from {path}")
        return pipeline

class GBFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transforms sequence data into a selection of scale-independent financial time series features
    suitable for gradient boosting models.
    """
    def __init__(self, features_to_extract='all'):
        """
        Initializes the feature extractor.

        Args:
            features_to_extract (list or 'all'): A list of feature names to extract,
                                                 or 'all' to extract all available features.
                                                 Defaults to 'all'.
        """
        self.n_features_in_ = None
        # self.numeric_feature_names = ['bid', 'ask', 'price', 'log_bid_size', 'log_ask_size', 'log_flux'] # Kept for reference, but not directly used in logic below
        self.scaler = None
        self.feature_names = None # Will store the names of features actually extracted and scaled
        self.features_to_extract = features_to_extract
        self._selected_feature_names = [] # Stores the validated list of features to compute

        # Map feature names to their calculation methods
        self._all_feature_calculators = {
            'price_trend_ratio': self._calculate_price_trend_ratio,
            'return_volatility': self._calculate_return_volatility,
            'norm_price_range': self._calculate_norm_price_range,
            'rel_price_momentum': self._calculate_rel_price_momentum,
            'return_autocorr': self._calculate_return_autocorr,
            'mean_rel_spread': self._calculate_mean_rel_spread,
            'rel_spread_volatility': self._calculate_rel_spread_volatility,
            'spread_trend': self._calculate_spread_trend,
            'log_volume_imbalance': self._calculate_log_volume_imbalance,
            'volume_imbalance_stability': self._calculate_volume_imbalance_stability,
            'volume_trend': self._calculate_volume_trend,
            'action_transition_rate': self._calculate_action_transition_rate,
            'action_diversity': self._calculate_action_diversity,
            'mean_log_flux': self._calculate_mean_log_flux,
            'norm_flux_volatility': self._calculate_norm_flux_volatility,
        }

    def fit(self, X, y=None):
        """
        Fits the feature extractor and scaler to the data. Determines which features
        to extract based on `features_to_extract` and learns scaling parameters.
        """
        if not isinstance(X, dict):
            raise ValueError("X should be a dictionary of input sequences")

        # Assuming keys like 'action_input', 'numeric_input' exist after previous steps
        if 'numeric_input' not in X or 'action_input' not in X:
             raise ValueError("X dictionary must contain 'numeric_input' and 'action_input'")

        self.n_features_in_ = X['numeric_input'].shape[2] + X.get('cat_input_shape', 1) # Rough estimate

        # Determine which features to actually extract based on init parameter
        if self.features_to_extract == 'all':
            self._selected_feature_names = sorted(list(self._all_feature_calculators.keys())) # Sort for consistent order
        elif isinstance(self.features_to_extract, list):
            self._selected_feature_names = sorted([f for f in self.features_to_extract if f in self._all_feature_calculators])
            if not self._selected_feature_names:
                raise ValueError("No valid features selected in features_to_extract or list is empty.")
            # Check for duplicates
            if len(self._selected_feature_names) != len(set(self.features_to_extract)):
                 print("Warning: Duplicate feature names found in features_to_extract. Using unique set.")
                 self._selected_feature_names = sorted(list(set(self._selected_feature_names)))

        else:
            raise ValueError("features_to_extract must be 'all' or a list of valid feature names.")

        # Extract only the selected features to learn scaling parameters
        features = self._extract_features(X) # This will now only extract selected features

        # Store the names of the features that were actually extracted and will be scaled
        self.feature_names = sorted(list(features.keys())) # Should match _selected_feature_names

        # Fit scaler on the extracted training data
        self.scaler = StandardScaler()

        # Convert features to array for scaling, ensuring consistent order
        # Check if features were extracted before stacking
        if not self.feature_names:
             print("Warning: No features were extracted during fit. Scaler will not be fitted.")
             # Handle case with no features: maybe return self or raise error?
             # Depending on desired behavior, you might want an empty scaler or error.
             # Let's assume an empty scaler is acceptable for now.
             self.scaler.fit(np.empty((X['numeric_input'].shape[0], 0))) # Fit on empty array
        else:
            try:
                feature_array = np.column_stack([features[k] for k in self.feature_names])
                self.scaler.fit(feature_array)
            except ValueError as e:
                # This might happen if a feature calculation returns inconsistent shapes
                raise ValueError(f"Error fitting scaler. Check feature calculation outputs. Details: {e}") from e


        return self

    # --- Helper methods for feature calculation ---
    # (Refactored calculations - using intermediate values passed as arguments)

    def _calculate_price_trend_ratio(self, returns):
        up_moves = np.sum(returns > 1e-10, axis=1) # Use tolerance
        down_moves = np.sum(returns < -1e-10, axis=1) # Use tolerance
        total_moves = up_moves + down_moves
        # Avoid division by zero if no moves occurred
        safe_denominator = np.where(total_moves == 0, 1e-10, total_moves)
        return (up_moves - down_moves) / safe_denominator


    def _calculate_return_volatility(self, returns):
        return np.std(returns, axis=1)

    def _calculate_norm_price_range(self, price_data):
        mean_price = np.mean(price_data, axis=1)
        price_range = np.max(price_data, axis=1) - np.min(price_data, axis=1)
        # Avoid division by zero if mean price is zero or negative
        safe_denominator = np.where(mean_price <= 1e-10, 1e-10, mean_price)
        return price_range / safe_denominator


    def _calculate_rel_price_momentum(self, price_data):
         # Avoid division by zero if initial price is zero or negative
         safe_denominator = np.where(price_data[:, 0] <= 1e-10, 1e-10, price_data[:, 0])
         return (price_data[:, -1] - price_data[:, 0]) / safe_denominator


    def _calculate_return_autocorr(self, returns):
        n_samples = returns.shape[0]
        autocorr = np.zeros(n_samples)
        # Calculate correlation only for sequences with variation
        valid_mask = (np.std(returns[:, 1:], axis=1) > 1e-10) & (np.std(returns[:, :-1], axis=1) > 1e-10)

        for i in np.where(valid_mask)[0]:
             # Check again for constant slices which can cause issues
             if not np.all(returns[i, 1:] == returns[i, 1]) and not np.all(returns[i, :-1] == returns[i, 0]):
                 try:
                     corr_matrix = np.corrcoef(returns[i, 1:], returns[i, :-1])
                     # Ensure result is not NaN (can happen with insufficient data points or edge cases)
                     if not np.isnan(corr_matrix[0, 1]):
                         autocorr[i] = corr_matrix[0, 1]
                 except Exception: # Catch potential errors during correlation calculation
                     pass # Keep autocorr[i] as 0
        return autocorr


    def _calculate_mean_rel_spread(self, rel_spread):
        return np.mean(rel_spread, axis=1)

    def _calculate_rel_spread_volatility(self, rel_spread):
         mean_spread = np.mean(rel_spread, axis=1)
         # Avoid division by zero or near-zero mean spread
         safe_denominator = np.where(np.abs(mean_spread) < 1e-10, 1e-10, mean_spread)
         # Ensure std dev is non-negative before division
         std_dev = np.std(rel_spread, axis=1)
         return std_dev / safe_denominator


    def _calculate_spread_trend(self, rel_spread):
        # Calculate differences between consecutive relative spreads
        spread_changes = np.diff(rel_spread, axis=1)
        # Count significant positive changes (spread widening)
        up_changes = np.sum(spread_changes > 1e-10, axis=1)
        # Count significant absolute changes (any widening or narrowing)
        abs_changes_count = np.sum(np.abs(spread_changes) > 1e-10, axis=1)
        # Avoid division by zero if no significant changes occurred
        safe_denominator = np.where(abs_changes_count == 0, 1e-10, abs_changes_count)
        # Trend is the proportion of significant changes that were positive
        return up_changes / safe_denominator


    def _calculate_log_volume_imbalance(self, volume_imbalance):
        return np.mean(volume_imbalance, axis=1)

    def _calculate_volume_imbalance_stability(self, volume_imbalance):
        return np.std(volume_imbalance, axis=1)

    def _calculate_volume_trend(self, total_volume):
        volume_changes = np.diff(total_volume, axis=1)
        # Count significant positive changes
        up_changes = np.sum(volume_changes > 1e-10, axis=1)
        # Count significant absolute changes
        abs_changes_count = np.sum(np.abs(volume_changes) > 1e-10, axis=1)
        # Avoid division by zero
        safe_denominator = np.where(abs_changes_count == 0, 1e-10, abs_changes_count)
        return up_changes / safe_denominator


    def _calculate_action_transition_rate(self, action_data):
        seq_length = action_data.shape[1]
        if seq_length <= 1:
             return np.zeros(action_data.shape[0]) # No transitions possible
        transitions = np.sum(action_data[:, 1:] != action_data[:, :-1], axis=1)
        return transitions / (seq_length - 1)


    def _calculate_action_diversity(self, action_data):
        seq_length = action_data.shape[1]
        if seq_length == 0:
            return np.zeros(action_data.shape[0]) # No diversity if no actions
        unique_actions_count = np.zeros(action_data.shape[0])
        for i in range(action_data.shape[0]):
            unique_actions_count[i] = len(np.unique(action_data[i]))
        return unique_actions_count / seq_length


    def _calculate_mean_log_flux(self, flux):
        return np.mean(flux, axis=1)

    def _calculate_norm_flux_volatility(self, flux):
        mean_flux = np.mean(flux, axis=1)
        # Avoid division by zero or near-zero mean flux
        safe_denominator = np.where(np.abs(mean_flux) < 1e-10, 1e-10, np.abs(mean_flux))
        return np.std(flux, axis=1) / safe_denominator


    def _extract_features(self, X):
        """
        Internal method to calculate the features specified in self._selected_feature_names.
        """
        # Check if essential inputs exist
        if 'numeric_input' not in X or 'action_input' not in X:
             raise ValueError("_extract_features requires 'numeric_input' and 'action_input' in X")

        numeric_data = X['numeric_input']  # Shape: (n_sequences, seq_length, 6)
        action_data = X['action_input']    # Shape: (n_sequences, seq_length)

        # --- Pre-calculate intermediate values needed by multiple features ---
        # Ensure numeric_data has the expected third dimension size
        if numeric_data.shape[2] < 6:
             raise ValueError(f"numeric_input expected to have at least 6 features, but got {numeric_data.shape[2]}")

        price_data = numeric_data[:, :, 2]
        returns = np.zeros_like(price_data)
        # Calculate returns safely
        prev_price = price_data[:, :-1]
        safe_denominator_price = np.where(prev_price <= 1e-10, 1e-10, prev_price)
        returns[:, 1:] = (price_data[:, 1:] - prev_price) / safe_denominator_price

        bid_data = numeric_data[:, :, 0]
        ask_data = numeric_data[:, :, 1]
        spread = ask_data - bid_data
        mid_price = (bid_data + ask_data) / 2
        # Calculate relative spread safely
        safe_mid_price = np.where(mid_price <= 1e-10, 1e-10, mid_price)
        rel_spread = spread / safe_mid_price

        log_bid_size = numeric_data[:, :, 3]
        log_ask_size = numeric_data[:, :, 4]
        volume_imbalance = log_bid_size - log_ask_size
        total_volume = log_bid_size + log_ask_size

        flux = numeric_data[:, :, 5] # log_flux

        # --- Store intermediate values for potential use by calculators ---
        intermediate_data = {
            'price_data': price_data,
            'returns': returns,
            'rel_spread': rel_spread,
            'volume_imbalance': volume_imbalance,
            'total_volume': total_volume,
            'action_data': action_data,
            'flux': flux
        }

        features = {}
        # --- Calculate only the selected features ---
        for feature_name in self._selected_feature_names:
            calculator = self._all_feature_calculators[feature_name]
            # Determine which intermediate data the calculator needs
            # This mapping could be more sophisticated if needed
            if 'price' in feature_name or 'return' in feature_name:
                 if feature_name == 'norm_price_range' or feature_name == 'rel_price_momentum':
                     features[feature_name] = calculator(intermediate_data['price_data'])
                 else:
                     features[feature_name] = calculator(intermediate_data['returns'])
            elif 'spread' in feature_name:
                features[feature_name] = calculator(intermediate_data['rel_spread'])
            elif 'volume' in feature_name:
                if 'imbalance' in feature_name:
                    features[feature_name] = calculator(intermediate_data['volume_imbalance'])
                else: # volume_trend
                    features[feature_name] = calculator(intermediate_data['total_volume'])
            elif 'action' in feature_name:
                features[feature_name] = calculator(intermediate_data['action_data'])
            elif 'flux' in feature_name:
                features[feature_name] = calculator(intermediate_data['flux'])
            else:
                 print(f"Warning: Don't know how to calculate feature '{feature_name}'. Skipping.")


        return features

    def transform(self, X):
        """
        Transforms the input data by extracting the selected features and scaling them.
        """
        if self.scaler is None or self.feature_names is None:
             raise RuntimeError("The feature extractor has not been fitted yet.")

        # Extract features (will use _selected_feature_names determined during fit)
        features = self._extract_features(X)

        # Check if any features were extracted
        if not self.feature_names:
             if features: # If transform extracted features but fit didn't expect any
                 print("Warning: Features extracted during transform, but none expected from fit. Returning empty.")
             # Return empty array with correct number of samples
             num_samples = X['numeric_input'].shape[0] if 'numeric_input' in X else 0
             return np.empty((num_samples, 0))


        # Ensure the extracted features match the ones from fitting, handle discrepancies
        current_feature_keys = sorted(list(features.keys()))
        if current_feature_keys != self.feature_names:
            # Align features: Use features expected from fit, fill missing with 0 before scaling
            aligned_features = {}
            num_samples = X['numeric_input'].shape[0] # Get number of samples
            for name in self.feature_names:
                if name in features:
                    aligned_features[name] = features[name]
                else:
                    print(f"Warning: Feature '{name}' expected from fit was not found during transform. Filling with zeros.")
                    aligned_features[name] = np.zeros(num_samples)
            features = aligned_features # Replace potentially incomplete features dict

        # Convert to array for scaling, ensuring correct order based on self.feature_names
        try:
            feature_array = np.column_stack([features[k] for k in self.feature_names])
        except ValueError as e:
             raise RuntimeError(f"ValueError during column stacking in transform. Check feature dimensions. Features: {{k: v.shape for k, v in features.items()}}") from e
        except KeyError as e:
             # This should ideally be caught by the alignment logic above, but as a safeguard:
             raise RuntimeError(f"Feature '{e}' expected from fit was not found during transform even after alignment.") from e


        # Apply scaling
        scaled_features_array = self.scaler.transform(feature_array)

        # Convert back to dictionary using the feature names from fit
        result = {}
        for i, name in enumerate(self.feature_names):
            result[name] = scaled_features_array[:, i]

        return result

    def get_feature_names_out(self, input_features=None):
        """Returns the names of the features produced by the transform method."""
        if self.feature_names is None:
             # Option 1: Return empty list if not fitted
             # return []
             # Option 2: Raise error
             raise RuntimeError("Cannot get feature names before fitting.")
        return self.feature_names