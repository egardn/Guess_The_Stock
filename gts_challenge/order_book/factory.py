import logging
from typing import Dict, Any, Tuple
import pandas as pd
from pathlib import Path
from gts_challenge.order_book.base.model_interface import ModelInterface
from gts_challenge.order_book.base.pipeline_interface import PipelineInterface

# Configure logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Uncomment if needed


def create_workflow(model_type: str) -> Dict[str, Any]:
    """
    Crée un workflow complet pour le type de modèle spécifié, incluant les fonctions API.

    Args:
        model_type: 'gru' ou 'gb'

    Returns:
        Un dictionnaire contenant toutes les composantes du workflow
    """
    model_type = model_type.lower()
    if model_type == 'gru':
        # Import only when needed
        try:
            from gts_challenge.order_book.models.gru.model import OrderBookGRUModel
            from gts_challenge.order_book.models.gru.pipeline import GRUPipeline
            # Import workflow functions from the correct location
            from gts_challenge.order_book.workflows import gru_workflow
        except ImportError as ie:
            logger.error(f"Failed to import GRU components/workflow: {ie}", exc_info=True)
            raise

        # Check for required functions
        required_funcs = ['train_model', 'evaluate_model', 'predict', 'explain', 'api_predict', 'api_explain', 'preprocess_data']
        missing_funcs = [f for f in required_funcs if not hasattr(gru_workflow, f)]
        if missing_funcs:
            logger.error(f"GRU workflow module ('gru_workflow.py') is missing required functions: {missing_funcs}")
            raise AttributeError(f"GRU workflow module is missing required functions: {missing_funcs}")

        return {
            'pipeline': GRUPipeline(),
            'model_creator': lambda **kwargs: OrderBookGRUModel(**kwargs),
            'train_fn': gru_workflow.train_model,
            'evaluate_fn': gru_workflow.evaluate_model,
            'predict_fn': gru_workflow.predict,
            'explain_fn': gru_workflow.explain, # Added explain_fn
            'api_predict_fn': gru_workflow.api_predict, # Added api_predict_fn
            'api_explain_fn': gru_workflow.api_explain, # Added api_explain_fn
            'preprocess_fn': gru_workflow.preprocess_data,
        }
    elif model_type == 'gb':
        # Import only when needed
        try:
            from gts_challenge.order_book.models.gb.model import OrderBookGBModel
            from gts_challenge.order_book.models.gb.pipeline import GBPipeline
            # Import workflow functions from the correct location
            from gts_challenge.order_book.workflows import gb_workflow
        except ImportError as ie:
            logger.error(f"Failed to import GB components/workflow: {ie}", exc_info=True)
            raise

        # Check for required functions
        required_funcs = ['train_model', 'evaluate_model', 'predict', 'explain', 'api_predict', 'api_explain', 'preprocess_data']
        missing_funcs = [f for f in required_funcs if not hasattr(gb_workflow, f)]
        if missing_funcs:
            logger.error(f"GB workflow module ('gb_workflow.py') is missing required functions: {missing_funcs}")
            raise AttributeError(f"GB workflow module is missing required functions: {missing_funcs}")

        return {
            'pipeline': GBPipeline(),
            # Adjusted model creator based on original template
            'model_creator': lambda **kwargs: OrderBookGBModel(params=kwargs),
            'train_fn': gb_workflow.train_model,
            'evaluate_fn': gb_workflow.evaluate_model,
            'predict_fn': gb_workflow.predict,
            'explain_fn': gb_workflow.explain, # Added explain_fn
            'api_predict_fn': gb_workflow.api_predict, # Added api_predict_fn
            'api_explain_fn': gb_workflow.api_explain, # Added api_explain_fn
            'preprocess_fn': gb_workflow.preprocess_data,
        }
    else:
        logger.error(f"Attempted to create workflow for unsupported model type: {model_type}")
        raise ValueError(f"Type de modèle non supporté: {model_type}. Utilisez 'gru' ou 'gb'.")


def load_workflow(model_type: str, model_path: str, pipeline_path: str) -> Dict[str, Any]:
    """
    Loads a previously trained workflow from saved model and pipeline files.
    Note: This version doesn't load API-specific functions by default.

    Args:
        model_type: 'gru' or 'gb'
        model_path: Path to the saved model file
        pipeline_path: Path to the saved pipeline file

    Returns:
        A dictionary containing the loaded workflow components
    """
    model_type = model_type.lower()
    if model_type == 'gru':
        # Import only when needed
        try:
            from gts_challenge.order_book.models.gru.model import OrderBookGRUModel
            from gts_challenge.order_book.models.gru.pipeline import GRUPipeline
            from gts_challenge.order_book.workflows import gru_workflow # Needed for evaluate/predict
        except ImportError as ie:
            logger.error(f"Failed to import GRU components/workflow for loading: {ie}", exc_info=True)
            raise

        # Load model and pipeline (Assuming .load methods exist)
        try:
            model = OrderBookGRUModel.load(model_path)
            pipeline = GRUPipeline.load(pipeline_path)
        except Exception as e:
            logger.error(f"Failed to load GRU model/pipeline from paths: {model_path}, {pipeline_path}. Error: {e}", exc_info=True)
            raise

        return {
            'pipeline': pipeline,
            'model': model,
            'evaluate_fn': gru_workflow.evaluate_model,
            'predict_fn': gru_workflow.predict
            # Note: API functions not included here, use create_workflow if needed
        }
    elif model_type == 'gb':
        # Import only when needed
        try:
            from gts_challenge.order_book.models.gb.model import OrderBookGBModel
            from gts_challenge.order_book.models.gb.pipeline import GBPipeline
            from gts_challenge.order_book.workflows import gb_workflow # Needed for evaluate/predict
        except ImportError as ie:
            logger.error(f"Failed to import GB components/workflow for loading: {ie}", exc_info=True)
            raise

        # Load model and pipeline (Assuming .load methods exist)
        try:
            model = OrderBookGBModel.load(model_path)
            pipeline = GBPipeline.load(pipeline_path)
        except Exception as e:
            logger.error(f"Failed to load GB model/pipeline from paths: {model_path}, {pipeline_path}. Error: {e}", exc_info=True)
            raise

        return {
            'pipeline': pipeline,
            'model': model,
            'evaluate_fn': gb_workflow.evaluate_model,
            'predict_fn': gb_workflow.predict
            # Note: API functions not included here, use create_workflow if needed
        }
    else:
        logger.error(f"Attempted to load workflow for unsupported model type: {model_type}")
        raise ValueError(f"Type de modèle non supporté: {model_type}. Utilisez 'gru' ou 'gb'.")


# --- API Specific Workflow Execution ---

def api_predict_workflow(model_type: str, model: Any, pipeline: Any, X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the prediction workflow for a given model type using API functions.
    Retrieves the necessary API function dynamically via create_workflow.
    """
    try:
        # Get the workflow dictionary containing the API function
        workflow_config = create_workflow(model_type)
        api_predict_fn = workflow_config.get('api_predict_fn')

        if not callable(api_predict_fn):
             logger.error(f"API predict function ('api_predict_fn') not found or not callable in workflow config for model type: {model_type}")
             raise ValueError(f"API predict function not configured correctly for {model_type}")

        # Call the specific API prediction function
        predictions = api_predict_fn(
            model=model,
            pipeline=pipeline,
            X_df=X_df
        )
        return predictions
    except Exception as e:
        logger.error(f"Error during api_predict_workflow for {model_type}: {e}", exc_info=True)
        # Re-raise or handle as appropriate for the API layer
        raise ValueError(f"Failed to execute API prediction workflow for {model_type}") from e


def api_explain_workflow(model_type: str, model: Any, pipeline: Any, X_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes the explanation workflow for a given model type using API functions.
    Retrieves the necessary API function dynamically via create_workflow.
    """
    try:
        # Get the workflow dictionary containing the API function
        workflow_config = create_workflow(model_type)
        api_explain_fn = workflow_config.get('api_explain_fn')

        if not callable(api_explain_fn):
             logger.error(f"API explain function ('api_explain_fn') not found or not callable in workflow config for model type: {model_type}")
             raise ValueError(f"API explain function not configured correctly for {model_type}")

        # Call the specific API explanation function
        explanation_results = api_explain_fn(
            model=model,
            pipeline=pipeline,
            X_df=X_df
        )
        return explanation_results
    except Exception as e:
        logger.error(f"Error during api_explain_workflow for {model_type}: {e}", exc_info=True)
        # Re-raise or handle as appropriate for the API layer
        raise ValueError(f"Failed to execute API explanation workflow for {model_type}") from e


# --- Training and Standard Prediction Workflows (from template) ---

def train_workflow(model_type: str, X_path: str, y_path: str, **params) -> Tuple[ModelInterface, PipelineInterface, Dict[str, Any]]:
    """
    Orchestrates the training process including preprocessing, model creation,
    training, and saving.

    Args:
        model_type (str): Type of model ('gru' or 'gb').
        X_path (str): Path to the input features data file.
        y_path (str): Path to the target labels data file.
        **params: Additional parameters including model_params, chunk_size,
                  val_split, preprocessed_dir, checkpoint_dir, epochs (for GRU),
                  gb_features (for GB).

    Returns:
        Tuple[ModelInterface, PipelineInterface, Dict[str, Any]]:
            The trained model, the fitted pipeline, and the training history.
    """
    logger.info(f"Starting training workflow for model type: {model_type}")
    workflow = create_workflow(model_type)
    preprocess_fn = workflow['preprocess_fn']

    # Preprocessing
    logger.info("Starting preprocessing...")
    preprocessed_dir = Path(params.get('preprocessed_dir', './preprocessed_data'))
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # Extract gb_features specifically for the preprocessing step if model_type is 'gb'
    preprocess_args = {
        'X_path': X_path,
        'y_path': y_path,
        'val_split': params.get('val_split'),
        'preprocessed_dir': preprocessed_dir,
        'chunk_size': params.get('chunk_size')
    }
    if model_type == 'gb':
        preprocess_args['gb_features'] = params.get('gb_features', 'all') # Extract gb_features

    X_train_path, y_train_path, X_val_path, y_val_path, pipeline_path = preprocess_fn(**preprocess_args) # Pass gb_features if applicable
    logger.info(f"Preprocessing complete. Train data: {X_train_path}, {y_train_path}. Val data: {X_val_path}, {y_val_path}. Pipeline: {pipeline_path}")

    # Load the fitted pipeline
    try:
        pipeline_instance = workflow['pipeline'].load(pipeline_path) # Use the load method from the correct class
        logger.info(f"Loaded fitted pipeline from {pipeline_path}")
    except Exception as e:
        logger.error(f"Failed to load pipeline from {pipeline_path}: {e}", exc_info=True)
        raise

    # Création et entraînement du modèle
    logger.info("Creating and training model...")
    model_creator = workflow['model_creator']
    train_fn = workflow['train_fn']

    model = model_creator(**params.get('model_params', {}))
    model, history = train_fn(
        model=model,
        pipeline=pipeline_instance, # Pass the loaded pipeline instance
        X_train_path=X_train_path,
        y_train_path=y_train_path,
        X_val_path=X_val_path,
        y_val_path=y_val_path,
        **params # Pass remaining params like checkpoint_dir, chunk_size etc.
    )
    logger.info("Model training finished.")

    return model, pipeline_instance, history


def predict_workflow(model_type: str, model_path: str, pipeline_path: str,
                X_test_path: str, **params) -> Dict[str, Any]:
    """
    Loads a trained model and generates predictions on test data.

    Args:
        model_type: 'gru' or 'gb'
        model_path: Path to the saved model file
        pipeline_path: Path to the saved pipeline file
        X_test_path: Path to test features data
        **params: Additional parameters for the predict function (e.g., batch_size)

    Returns:
        Dictionary or DataFrame containing predictions
    """
    logger.info(f"Starting prediction workflow for model type: {model_type}")
    # Load workflow components (model, pipeline, predict_fn)
    workflow = load_workflow(model_type, model_path, pipeline_path)
    logger.info(f"Loaded model from {model_path} and pipeline from {pipeline_path}")

    # Generate predictions
    predict_fn = workflow['predict_fn']
    logger.info(f"Generating predictions for data at: {X_test_path}")
    predictions = predict_fn(
        X_path=X_test_path,
        model=workflow['model'],
        pipeline=workflow['pipeline'],
        **params # Pass prediction-specific params
    )
    logger.info("Prediction generation complete.")

    # Assuming predict_fn returns the desired format (e.g., DataFrame)
    return predictions