from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin


class ModelInterface(ABC, BaseEstimator, TransformerMixin):
    """Interface commune pour tous les modèles d'order book"""
    
    @abstractmethod
    def create_model(self, **kwargs) -> 'ModelInterface':
        """Crée une nouvelle instance du modèle"""
        pass
    
    @abstractmethod
    def fit(self, X: Any, y: np.ndarray) -> 'ModelInterface':
        """Entraine le modèle sur les données"""
        pass
    
    @abstractmethod
    def transform(self, X: Any) -> Any:
        """Transform input data for prediction/evaluation"""
        pass

    @abstractmethod
    def predict(self, X: Any, explain: bool = False) -> np.ndarray:
        """Génère des prédictions pour les données d'entrée"""
        pass
    
    @abstractmethod
    def evaluate(self, X: Any, y: np.ndarray) -> float:
        """Évalue le modèle sur des données de validation"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Sauvegarde le modèle à l'emplacement spécifié"""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'ModelInterface':
        """Charge le modèle depuis l'emplacement spécifié"""
        pass
    
    @abstractmethod
    def get_feature_importance(self, X: Any, n_repeats: int = 5) -> Dict[str, float]:
        """Calcule l'importance des features"""
        pass
    
    
    @abstractmethod
    def visualize_feature_importance(self, importance_scores: Dict[str, Any], **kwargs) -> Any:
        """Visualize feature importance"""
        pass