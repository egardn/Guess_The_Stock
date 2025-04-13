import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgbm
import pandas as pd
import pickle
import os
from gts_challenge.order_book.base.model_interface import ModelInterface
from typing import Dict, Any, Optional


class OrderBookGBModel(ModelInterface):
    """
    Gradient boosting model for order book classification using pipeline preprocessing
    """
    def __init__(self, params=None):
        # Default parameters for LightGBM
        self.params = params or {
            'objective': 'multiclass',
            'num_class': 24,  # 24 stocks to classify
            'boosting_type': 'gbdt',
            'metric': 'multi_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'n_estimators': 100,
            'verbose': -1
        }
        self.model = None
        self.feature_names = None
        
    def create_model(self, **kwargs) -> 'ModelInterface':
        """Create and initialize the LightGBM model"""
        # Update parameters if provided
        if kwargs:
            self.params.update(kwargs)
            
        self.model = lgbm.LGBMClassifier(**self.params)
        return self
    
    def fit(self, X_train, y_train, X_val, y_val) -> 'ModelInterface':
        """
        Fit the model to the data
        
        Parameters:
        -----------
        X : dict
            Features extracted from order book data as dictionary with feature names as keys
        y : array-like
            Target labels (stock identifiers 0-23)
        validation_data : tuple, optional
            (X_val, y_val) for validation during training
        early_stopping_rounds : int, optional
            Stop training when validation score doesn't improve
        """
        if self.model is None:
            self.create_model()
        
        # Convert dictionary to DataFrame
        X_train_df = pd.DataFrame(X_train)
        
        # Store feature names
        self.feature_names = list(X_train.keys())
            
        
        X_val_df = pd.DataFrame(X_val)

        # Create dictionary to store evaluation results
        evals_result = {}
        
        # Define evaluation sets
        eval_set = [(X_train_df, y_train), (X_val_df, y_val)]
        eval_names = ['train', 'valid']
        
        # Fit the model
        self.model.fit(
            X_train_df, y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric='multi_logloss',
            callbacks=[lgbm.early_stopping(5, verbose=True, min_delta=0.001)],
        )

        # Compute validation accuracy after training

        from sklearn.metrics import accuracy_score, classification_report
        # Get predictions on validation set
        y_val_pred = self.model.predict(X_val_df)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        # Print training information
        print(f"Model training completed:")
        print(f"Best iteration: {self.model.best_iteration_}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print("\nValidation Classification Report:")
        print(classification_report(y_val, y_val_pred))

        # Store validation accuracy as an attribute
        self.val_accuracy_ = val_accuracy
        
        # Return both model and loss
        return self, self.model.evals_result_
    
    def transform(self, X) -> Any:
        """
        Transform input data for prediction/evaluation
        
        For gradient boosting models, typically no transformation is needed at model level
        This is implemented to comply with the ModelInterface
        """
        # Convert dictionary to DataFrame
        return pd.DataFrame(X)
    
    def predict(self, X, explain=False) -> np.ndarray:
        """
        Generate class predictions
        
        Parameters:
        -----------
        X : dict
            Features to predict on as dictionary
            
        Returns:
        --------
        np.ndarray : Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Convert dictionary to DataFrame
        X_df = pd.DataFrame(X)
        return self.model.predict(X_df)
    
    def predict_proba(self, X) -> np.ndarray:
        """Generate class probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
            
        # Convert dictionary to DataFrame
        X_df = pd.DataFrame(X)
        return self.model.predict_proba(X_df)
    
    def evaluate(self, X, y_true) -> float:
        """
        Evaluate the model performance
        
        Parameters:
        -----------
        X : dict
            Features to evaluate on as dictionary
        y_true : array-like
            True labels
            
        Returns:
        --------
        float : Accuracy score
        """
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        
        y_pred = self.predict(X)
        
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        print(f"Evaluation accuracy: {results['accuracy']:.4f}")
        #print("Classification Report:")
        #print(results['classification_report'])
        
        return results['accuracy']
    
    def save(self, path: str) -> None:
        """
        Save the model to the specified location
        
        Parameters:
        -----------
        path : str
            Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model using pickle
        with open(path, 'wb') as f:
            model_data = {
                'model': self.model,
                'params': self.params,
                'feature_names': self.feature_names
            }
            pickle.dump(model_data, f)
            
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelInterface':
        """
        Load a model from the specified location
        
        Parameters:
        -----------
        path : str
            Path from where to load the model
            
        Returns:
        --------
        ModelInterface : Loaded model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance and set attributes
        instance = cls(params=model_data['params'])
        instance.model = model_data['model']
        instance.feature_names = model_data.get('feature_names')
        
        return instance
    
    def get_feature_importance(self, X) -> Dict[str, float]:
        """
        Calculate feature importance
        
        Parameters:
        -----------
        X : dict
            Features to use for importance calculation
            
        Returns:
        --------
        Dict[str, float] : Dictionary of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Get feature importance from the model
        importance = self.model.feature_importances_
        
        # Get feature names
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = list(X.keys())
        
        # Create dictionary of feature importance
        importance_dict = {name: score for name, score in zip(feature_names, importance)}
        
        return importance_dict
    
    def visualize_feature_importance(self, importance_scores: Dict[str, Any], **kwargs) -> Any:
        """
        Visualize feature importance
        
        Parameters:
        -----------
        importance_scores : Dict[str, Any]
            Feature importance scores as returned by get_feature_importance
        **kwargs : Additional keyword arguments
            top_n : int, optional
                Number of top features to display
                
        Returns:
        --------
        pd.DataFrame : DataFrame with feature importance scores
        """
        top_n = kwargs.get('top_n', 20)
        
        # Convert to DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': list(importance_scores.keys()),
            'Importance': list(importance_scores.values())
        }).sort_values(by='Importance', ascending=False)
        
        # Take top N features
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
        
        # Return the importance DataFrame
        return importance_df