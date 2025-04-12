from dataclasses import dataclass
from typing import Dict, List, Union, Any, Optional
import numpy as np
import pandas as pd

@dataclass
class DataContainer:
    """Format de données standard utilisé entre les modules"""
    
    # Données principales (peut être dict, DataFrame, ou autre selon le besoin)
    data: Union[Dict[str, np.ndarray], pd.DataFrame]
    
    # IDs des observations pour le mapping
    obs_ids: List
    
    # Métadonnées optionnelles
    metadata: Optional[Dict[str, Any]] = None
    
    def to_gru_format(self) -> Dict[str, np.ndarray]:
        """Convertit les données au format attendu par les modèles GRU"""
        if isinstance(self.data, dict):
            # Déjà au format GRU
            return self.data
        
        # Conversion d'un DataFrame au format GRU
        # Implémentation spécifique selon les besoins
        # ...
        
        raise NotImplementedError("Conversion non implémentée pour ce type de données")
    
    def to_gb_format(self) -> pd.DataFrame:
        """Convertit les données au format attendu par les modèles GB"""
        if isinstance(self.data, pd.DataFrame):
            # Déjà au format GB
            return self.data
        
        # Conversion d'un dict au format GB
        # Implémentation spécifique selon les besoins
        # ...
        
        raise NotImplementedError("Conversion non implémentée pour ce type de données")