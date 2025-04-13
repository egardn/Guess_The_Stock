from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PipelineInterface(ABC, BaseEstimator, TransformerMixin):
    """Interface commune pour tous les pipelines de preprocessing"""
    
    @abstractmethod
    def fit(self, X: Any, y=None) -> 'PipelineInterface':
        """Ajuste le pipeline aux données"""
        pass
    
    @abstractmethod
    def transform(self, X: Any) -> Any:
        """Transforme les données et retourne un format standardisé"""
        pass
    
    @abstractmethod
    def fit_transform(self, X: Any, y=None) -> Any:
        """Ajuste puis transforme les données"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Sauvegarde le pipeline"""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'PipelineInterface':
        """Charge le pipeline"""
        pass