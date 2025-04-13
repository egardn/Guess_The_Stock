import sys
import os
from fastapi import HTTPException
from datetime import datetime, timedelta
import logging
import pickle
from typing import Dict, Optional, Any, List
from pathlib import Path
# Keep necessary model/pipeline class imports if needed for type hinting, but factory handles creation/loading logic
# from gts_challenge.order_book.models.gru.model import OrderBookGRUModel
# from gts_challenge.order_book.models.gb.model import OrderBookGBModel
# from gts_challenge.order_book.models.gru.pipeline import GRUPipeline
# from gts_challenge.order_book.models.gb.pipeline import GBPipeline
# REMOVE factory import if only used for create_workflow/load_workflow previously
# from gts_challenge.order_book.factory import load_workflow # Or create_workflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define get_path_from_env BEFORE using it ---
def get_path_from_env(env_var: str, default: Optional[str] = None) -> Optional[Path]:
    """Gets a path from environment variable and returns it as a Path object."""
    path_str = os.getenv(env_var, default)
    if path_str:
        # Resolve the path to make it absolute and handle potential relative paths correctly
        # Use Path(__file__).parent.parent to get the project root if paths are relative to project root
        # Assuming paths set in .bat are absolute or relative to the .bat file location (%~dp0)
        # If they are relative to the project root where api/ is, adjust base path accordingly.
        # Since %~dp0 makes them absolute, direct Path conversion should be fine.
        # Consider adding error handling if the path doesn't resolve correctly
        try:
            resolved_path = Path(path_str).resolve(strict=True) # strict=True checks existence
            return resolved_path
        except FileNotFoundError:
             logger.warning(f"Path specified by env var '{env_var}' or default '{default}' ('{path_str}') does not exist.")
             return None
        except Exception as e:
             logger.error(f"Error resolving path '{path_str}' from env var '{env_var}': {e}")
             return None
    elif default:
         # Handle default path resolution if env var is not set
         try:
             # Assuming default path is relative to this file's directory or project root
             # Adjust base path as needed, e.g., Path(__file__).parent.parent / default
             base_path = Path(__file__).parent.parent # Example: project root
             resolved_path = (base_path / default).resolve(strict=True)
             return resolved_path
         except FileNotFoundError:
             logger.warning(f"Default path '{default}' for env var '{env_var}' does not exist relative to {base_path}.")
             return None
         except Exception as e:
             logger.error(f"Error resolving default path '{default}' for env var '{env_var}': {e}")
             return None
    return None


# --- Paths to model and pipeline files from Environment Variables ---
MODEL_PATHS = {
    'gru': get_path_from_env("GRU_MODEL_PATH", "../models/gru/final_model_gru.pkl"), # Adjusted default path example
    'gb': get_path_from_env("GB_MODEL_PATH", "../models/gb/final_model_gb.pkl")    # Adjusted default path example
}
PIPELINE_PATHS = {
    'gru': get_path_from_env("GRU_PIPELINE_PATH", "../data/preprocessed_data/gru_pipeline.pkl"), # Adjusted default path example
    'gb': get_path_from_env("GB_PIPELINE_PATH", "../data/preprocessed_data/gb_pipeline.pkl")    # Adjusted default path example
}

# Track API start time
app_start_time = datetime.now()

print(f"Uvicorn using Python executable: {sys.executable}")
# Log the resolved paths for debugging
logger.info(f"Resolved GRU Model Path: {MODEL_PATHS.get('gru')}")
logger.info(f"Resolved GRU Pipeline Path: {PIPELINE_PATHS.get('gru')}")
logger.info(f"Resolved GB Model Path: {MODEL_PATHS.get('gb')}")
logger.info(f"Resolved GB Pipeline Path: {PIPELINE_PATHS.get('gb')}")


class ModelLoader:
    # ... (rest of the ModelLoader class and utils.py file as previously corrected) ...
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._pipelines: Dict[str, Any] = {}
        self._load_times: Dict[str, Optional[datetime]] = {'gru': None, 'gb': None}
        self._pipeline_load_times: Dict[str, Optional[datetime]] = {'gru': None, 'gb': None}
        # REMOVE: self._workflows: Dict[str, Any] = {}

    def _load_component(self, component_type: str, model_type: str):
        # ... (this method remains largely the same, loading from MODEL_PATHS/PIPELINE_PATHS) ...
        is_model = component_type == "model"
        paths = MODEL_PATHS if is_model else PIPELINE_PATHS
        storage = self._models if is_model else self._pipelines
        load_time_storage = self._load_times if is_model else self._pipeline_load_times
        path_obj: Optional[Path] = paths.get(model_type) # Get the Path object

        # Use Path object's exists() method
        if not path_obj: # Path resolution now handles existence check in get_path_from_env
            logger.error(f"{component_type.capitalize()} path for type '{model_type}' could not be resolved or does not exist.")
            raise FileNotFoundError(f"{component_type.capitalize()} path for type '{model_type}' not found or invalid.")

        # Check if already loaded (no change needed here)
        if model_type in storage:
             logger.debug(f"{component_type.capitalize()} for type '{model_type}' already loaded.")
             return True

        try:
            logger.info(f"Loading {component_type} for type '{model_type}' from {path_obj}...")
            # Use Path object's open() method
            with path_obj.open("rb") as f:
                # Load the actual model/pipeline object
                component_object = pickle.load(f)

                # If the pickle contains a dict like {'model': actual_model}, extract it
                if is_model and isinstance(component_object, dict) and 'model' in component_object:
                     logger.debug(f"Extracting actual model object from loaded dictionary for type '{model_type}'.")
                     storage[model_type] = component_object['model']
                else:
                     storage[model_type] = component_object

            load_time_storage[model_type] = datetime.now()
            logger.info(f"{component_type.capitalize()} for type '{model_type}' loaded successfully.")
            return True
        except FileNotFoundError: # Should be caught by path_obj check now
             logger.error(f"{component_type.capitalize()} file not found for type '{model_type}' at {path_obj}")
             raise # Re-raise the exception to potentially stop startup
        except pickle.UnpicklingError as pe:
             logger.error(f"Failed to unpickle {component_type} for type '{model_type}' from {path_obj}: {pe}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Failed to load corrupted {component_type} file for {model_type}")
        except Exception as e:
            logger.error(f"Failed to load {component_type} for type '{model_type}' from {path_obj}: {str(e)}", exc_info=True)
            storage.pop(model_type, None) # Ensure it's removed if loading failed
            load_time_storage[model_type] = None
            # Depending on requirements, maybe raise error here too
            raise HTTPException(status_code=500, detail=f"Failed to load {component_type} for {model_type}")


    def load_model(self, model_type: str) -> bool:
        """Loads a specific model file."""
        return self._load_component("model", model_type)

    def load_pipeline(self, model_type: str) -> bool:
        """Loads a specific pipeline file."""
        return self._load_component("pipeline", model_type)

    def get_model(self, model_type: str) -> Any:
        """Gets a loaded model object, loading it if necessary."""
        if model_type not in self._models:
            self.load_model(model_type) # load_model now raises exceptions on failure
        return self._models[model_type]

    def get_pipeline(self, model_type: str) -> Any:
        """Gets a loaded pipeline object, loading it if necessary."""
        if model_type not in self._pipelines:
            self.load_pipeline(model_type) # load_pipeline now raises exceptions on failure
        return self._pipelines[model_type]

    # REMOVE: load_workflow_components method
    # REMOVE: get_workflow method

    @property
    def loaded_models(self) -> List[str]:
        # Return keys of successfully loaded models
        return list(self._models.keys())

    @property
    def loaded_pipelines(self) -> List[str]:
         # Return keys of successfully loaded pipelines
         return list(self._pipelines.keys())

    @property
    def load_times(self) -> Dict[str, Optional[str]]:
         # ... (this method remains the same) ...
         times = {}
         for mt in ['gru', 'gb']:
              model_time = self._load_times.get(mt)
              pipe_time = self._pipeline_load_times.get(mt)
              # Report the latest load time if both are loaded
              latest_time = max(t for t in [model_time, pipe_time] if t is not None) if any(t is not None for t in [model_time, pipe_time]) else None
              times[mt] = latest_time.isoformat() if latest_time else None
         return times


# Initialize model loader
model_loader = ModelLoader()

# --- Load models and pipelines on startup ---
# It's often better to load essential resources when the API starts
# to catch errors early and improve response time for the first request.
try:
    logger.info("Pre-loading models and pipelines...")
    model_loader.load_model('gru')
    model_loader.load_pipeline('gru')
    model_loader.load_model('gb')
    model_loader.load_pipeline('gb')
    logger.info("Models and pipelines pre-loaded successfully.")
except Exception as e:
    # Log the error and potentially exit if loading fails,
    # as the API might be unusable.
    logger.critical(f"FATAL: Failed to load essential models/pipelines on startup: {e}", exc_info=True)
    # Depending on deployment, you might want to sys.exit(1) here
    # Or rely on the fact that subsequent get_model/get_pipeline calls will fail


def get_uptime():
    """Calculates API uptime."""
    delta = datetime.now() - app_start_time
    return str(delta).split('.')[0] # Format as H:MM:SS