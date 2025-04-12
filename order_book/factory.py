from typing import Dict, Any, Tuple
from order_book.base.model_interface import ModelInterface
from order_book.base.pipeline_interface import PipelineInterface


def create_workflow(model_type: str) -> Dict[str, Any]:
    """
    Crée un workflow complet pour le type de modèle spécifié
    
    Args:
        model_type: 'gru' ou 'gb'
        
    Returns:
        Un dictionnaire contenant toutes les composantes du workflow
    """
    if model_type.lower() == 'gru':
        # Import only when needed
        from order_book.models.gru.model import OrderBookGRUModel
        from order_book.models.gru.pipeline import GRUPipeline
        from order_book.workflows import gru_workflow
        
        return {
            'pipeline': GRUPipeline(),
            'model_creator': lambda **kwargs: OrderBookGRUModel(**kwargs),
            'train_fn': gru_workflow.train_model,
            'evaluate_fn': gru_workflow.evaluate_model,
            'predict_fn': gru_workflow.predict,
            'preprocess_fn': gru_workflow.preprocess_data,
        }
    elif model_type.lower() == 'gb':
        # Import only when needed
        from order_book.models.gb.model import OrderBookGBModel
        from order_book.models.gb.pipeline import GBPipeline
        from order_book.workflows import gb_workflow
        
        return {
            'pipeline': GBPipeline(),
            'model_creator': lambda **kwargs: OrderBookGBModel(params=kwargs),
            'train_fn': gb_workflow.train_model,
            'evaluate_fn': gb_workflow.evaluate_model,
            'predict_fn': gb_workflow.predict,
            'preprocess_fn': gb_workflow.preprocess_data,
        }
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}. Utilisez 'gru' ou 'gb'.")


def load_workflow(model_type: str, model_path: str, pipeline_path: str) -> Dict[str, Any]:
    """
    Loads a previously trained workflow from saved model and pipeline files
    
    Args:
        model_type: 'gru' or 'gb' 
        model_path: Path to the saved model file
        pipeline_path: Path to the saved pipeline file
        
    Returns:
        A dictionary containing the loaded workflow components
    """
    if model_type.lower() == 'gru':
        # Import only when needed
        from order_book.models.gru.model import OrderBookGRUModel
        from order_book.models.gru.pipeline import GRUPipeline
        from order_book.workflows import gru_workflow
        
        # Load model and pipeline
        model = OrderBookGRUModel.load(model_path)
        pipeline = GRUPipeline.load(pipeline_path)
        
        return {
            'pipeline': pipeline,
            'model': model,
            'evaluate_fn': gru_workflow.evaluate_model,
            'predict_fn': gru_workflow.predict
        }
    elif model_type.lower() == 'gb':
        # Import only when needed
        from order_book.models.gb.model import OrderBookGBModel
        from order_book.models.gb.pipeline import GBPipeline
        from order_book.workflows import gb_workflow
        
        # Load model and pipeline
        model = OrderBookGBModel.load(model_path)
        pipeline = GBPipeline.load(pipeline_path)
        
        return {
            'pipeline': pipeline,
            'model': model,
            'evaluate_fn': gb_workflow.evaluate_model,
            'predict_fn': gb_workflow.predict
        }
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}. Utilisez 'gru' ou 'gb'.")


def train_workflow(model_type: str, X_path: str, y_path: str, **params) -> Tuple[ModelInterface, PipelineInterface, Dict[str, Any]]:
    """
    Exécute le workflow complet pour un type de modèle donné
    
    Args:
        model_type: 'gru' ou 'gb'
        X_path: Chemin vers les données X
        y_path: Chemin vers les données y
        **params: Paramètres additionnels (batch_size, etc.)
        
    Returns:
        Modèle entraîné, pipeline et historique d'entraînement
    """
    workflow = create_workflow(model_type)
    
    # Prétraitement des données
    X_train_path, y_train_path, X_val_path, y_val_path = workflow['preprocess_fn'](
        X_path=X_path,
        y_path=y_path,
        val_split=params.get('val_split'),
        preprocessed_dir=params.get('preprocessed_dir'),
        chunk_size=params.get('chunk_size')
    )

    # Création et entraînement du modèle
    model = workflow['model_creator'](**params.get('model_params', {}))
    model, history = workflow['train_fn'](
        model=model,
        pipeline=workflow['pipeline'],
        X_train_path=X_train_path,
        y_train_path=y_train_path,
        X_val_path=X_val_path,
        y_val_path=y_val_path,
        **params
    )
    
    return model, workflow['pipeline'], history

def predict_workflow(model_type: str, model_path: str, pipeline_path: str, 
                X_test_path: str, **params) -> Dict[str, Any]:
    """
    Tests a previously trained model loaded from disk
    
    Args:
        model_type: 'gru' or 'gb'
        model_path: Path to the saved model file
        pipeline_path: Path to the saved pipeline file
        X_test_path: Path to test features data
        y_test_path: Path to test labels data (optional, if None will only generate predictions)
        **params: Additional parameters
        
    Returns:
        Dictionary containing evaluation metrics and/or predictions
    """
    # Load workflow components
    workflow = load_workflow(model_type, model_path, pipeline_path)
    
    # Prediction mode - only generate predictions
    predictions = workflow['predict_fn'](
        X_path=X_test_path,
        model=workflow['model'],
        pipeline=workflow['pipeline'],
        **params
    )
    
    return predictions