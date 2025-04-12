# scripts/tune.py

import optuna
import argparse
import logging
import json
from pathlib import Path
import sys
import time

# --- Add project root to Python path ---
# This allows importing from the 'order_book' package when running from the 'scripts' directory
# Note: When run via subprocess with cwd set, this might not be strictly necessary,
# but it's good practice for local execution.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
         sys.path.append(str(PROJECT_ROOT))
except NameError:
     # __file__ is not defined in interactive environments like basic Python shell
     # Assume a reasonable structure if running interactively for testing
     PROJECT_ROOT = Path('.').resolve().parent # Adjust if needed
     if str(PROJECT_ROOT) not in sys.path:
          sys.path.append(str(PROJECT_ROOT))


# --- Import project components ---
try:
    from order_book.factory import train_workflow
    # from optuna.integration import TFKerasPruningCallback, LightGBMPruningCallback # Uncomment if used directly
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"PROJECT_ROOT used for sys.path: {PROJECT_ROOT}")
    print("Ensure the script is run with correct permissions and environment.")
    sys.exit(1)

# --- Default Constants (mainly for local execution fallback) ---
# These will typically be overridden by command-line args, especially on Kaggle
DEFAULT_X_TRAIN_PATH = PROJECT_ROOT / "data" / "X_train.parquet"
DEFAULT_Y_TRAIN_PATH = PROJECT_ROOT / "data" / "y_train.parquet"
DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_data"
DEFAULT_CHECKPOINT_DIR_BASE = PROJECT_ROOT / "outputs" / "checkpoints" # Default output subdir
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" # Default general output subdir

# Model/Data Specifics (These might need to be args too if they vary)
N_VENUES = 15       # Example: Replace with actual count
N_ACTIONS = 3       # Example: 'A', 'D', 'U'
N_CATEGORIES = 24   # Example: Number of stock classes

# Tuning Settings Defaults
DEFAULT_N_TRIALS = 50
DEFAULT_STUDY_NAME = "orderbook_tuning"
DEFAULT_STORAGE_FILENAME = "tuning_study.db" # Default filename for DB

# --- Objective Function ---
# MODIFIED Signature: Added path arguments
def objective(trial: optuna.trial.Trial, model_type: str,
              x_path: Path, y_path: Path,
              preprocessed_dir: Path, checkpoint_dir_base: Path):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object.
        model_type: 'gru' or 'gb'.
        x_path (Path): Path to input training features (X).
        y_path (Path): Path to input training labels (y).
        preprocessed_dir (Path): Directory for preprocessed data (read/write).
        checkpoint_dir_base (Path): Base directory for trial checkpoints (write).

    Returns:
        float: The metric value to minimize/maximize.
    """
    # Get logger instance (configured in __main__)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Trial {trial.number} for model_type='{model_type}'")
    start_time = time.time()

    # Use the passed checkpoint_dir_base argument for trial-specific checkpoints
    trial_checkpoint_dir = checkpoint_dir_base / f"trial_{trial.number}"
    try:
        trial_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Trial {trial.number}: Could not create checkpoint directory {trial_checkpoint_dir}. Error: {e}", exc_info=True)
        return float('inf') # Or appropriate failure value for the direction

    # --- Setup parameters for train_workflow ---
    # Use the passed path arguments
    common_params = {
        'X_path': str(x_path), # Use passed arg
        'y_path': str(y_path), # Use passed arg
        'val_split': 0.2,      # Fixed validation split for tuning consistency
        'preprocessed_dir': str(preprocessed_dir), # Use passed arg (for saving/loading preprocessed data)
        'checkpoint_dir': str(trial_checkpoint_dir), # Use trial-specific checkpoint dir
        'trial': trial         # Pass trial object for pruning callbacks
    }

    # --- Define Hyperparameter Search Space (per model type) ---
    if model_type == 'gru':
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [500, 1000, 1500, 2000])
        gru_units = trial.suggest_categorical('gru_units', [32, 64, 128])
        dense_units = trial.suggest_categorical('dense_units', [32, 64, 128])
        gru_dropout_rate = trial.suggest_float('gru_dropout', 0.0, 0.5)

        params = {
            **common_params,
            'model_type': 'gru',
            'chunk_size': 10000, # Adjust based on memory/performance
            'epochs': 15,        # Fixed epochs; pruning callback handles early stop
            'model_params': {
                'learning_rate': lr,
                'batch_size': batch_size,
                'n_venues': N_VENUES,      # Consider making these args if they change
                'n_actions': N_ACTIONS,    # Consider making these args if they change
                'n_categories': N_CATEGORIES, # Consider making these args if they change
                'gru_units': gru_units,
                'dense_units': dense_units,
                'gru_dropout': gru_dropout_rate
            }
        }
        metric_key = 'val_loss' # Keras history key

    elif model_type == 'gb':
        early_stopping_rounds = 50
        n_estimators = trial.suggest_int('n_estimators', 200, 2000)

        params = {
            **common_params,
            'model_type': 'gb',
            'chunk_size': 100000, # GB might handle larger chunks; adjust
            'model_params': {
                 'objective': 'multi_logloss',
                 'metric': 'multi_logloss',
                 'num_class': N_CATEGORIES, # Consider making arg
                 'n_jobs': -1,
                 'random_state': 42,
                 'n_estimators': n_estimators,
                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                 'max_depth': trial.suggest_int('max_depth', 3, 12),
                 'num_leaves': trial.suggest_int('num_leaves', 8, 2**8),
                 'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                 'early_stopping_rounds': early_stopping_rounds, # Crucial for pruning callback
             },
            'early_stopping_rounds': early_stopping_rounds, # Pass explicitly if train_fn needs it
        }
        metric_key = 'valid' # LightGBM evals_result structure key
        metric_name = 'multi_logloss' # Specific metric within 'valid'

    else:
        logger.error(f"Trial {trial.number}: Unsupported model_type: {model_type}")
        raise ValueError(f"Unsupported model_type: {model_type}")

    # --- Execute Training Workflow ---
    try:
        # Assumes train_workflow correctly uses passed 'preprocessed_dir', 'checkpoint_dir', 'trial'
        logger.info(f"Trial {trial.number}: Running train_workflow with params: {params['model_params']}")
        model, pipeline, history = train_workflow(**params)

        # --- Extract Final Metric ---
        # Note: Intermediate reporting for pruning is handled by callbacks inside train_workflow/train_model
        final_metric = None
        if model_type == 'gru':
            if not history or metric_key not in history or not history[metric_key]:
                 logger.warning(f"Trial {trial.number}: Metric '{metric_key}' not found in GRU history.")
                 return float('inf') # Return high value if metric missing
            final_metric = history[metric_key][-1]
        elif model_type == 'gb':
            if not history or metric_key not in history or metric_name not in history[metric_key] or not history[metric_key][metric_name]:
                logger.warning(f"Trial {trial.number}: Metric '{metric_name}' not found in GB history['{metric_key}'].")
                return float('inf')
            # Get metric from the *best* iteration found by early stopping
            try:
                best_iter = getattr(model.model, 'best_iteration_', -1) # Check common attribute names
                if best_iter <= 0: # model.best_iteration_ is 1-based
                    logger.warning(f"Trial {trial.number}: Invalid best_iteration ({best_iter}) from GB model. Using last metric.")
                    final_metric = history[metric_key][metric_name][-1]
                else:
                    best_iteration_index = best_iter - 1
                    if best_iteration_index >= len(history[metric_key][metric_name]):
                        logger.warning(f"Trial {trial.number}: Best iteration index {best_iteration_index} out of range. Using last metric.")
                        final_metric = history[metric_key][metric_name][-1]
                    else:
                        final_metric = history[metric_key][metric_name][best_iteration_index]
            except AttributeError:
                 logger.warning(f"Trial {trial.number}: Could not determine best_iteration from GB model. Using last metric.")
                 final_metric = history[metric_key][metric_name][-1] # Fallback to last value

        if final_metric is None:
             logger.error(f"Trial {trial.number}: Failed to extract final metric.")
             return float('inf') # Indicate failure

        logger.info(f"Trial {trial.number} finished. Duration: {time.time() - start_time:.2f}s. Final Metric: {final_metric:.5f}")

        # Final report (intermediate reports handled by callbacks)
        trial.report(final_metric, step=params.get('epochs', n_estimators if model_type=='gb' else -1))

        # Check if trial should be pruned (often redundant if callbacks work, but safe)
        if trial.should_prune():
             logger.info(f"Trial {trial.number} pruned.")
             raise optuna.TrialPruned()

        return final_metric

    except optuna.TrialPruned:
        raise # Propagate prune signal
    except Exception as e:
        logger.error(f"Trial {trial.number} failed during execution: {e}", exc_info=True)
        return float('inf') # Signal failure to Optuna

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hyperparameter Optimization for Order Book Models")

    # --- Essential Arguments ---
    parser.add_argument('--model_type', type=str, required=True, choices=['gru', 'gb'],
                        help="Type of model to tune ('gru' or 'gb').")
    parser.add_argument('--n_trials', type=int, default=DEFAULT_N_TRIALS,
                        help="Number of optimization trials to run.")

    # --- Path Arguments (CRUCIAL for Kaggle/flexibility) ---
    parser.add_argument('--X_path', type=str, default=str(DEFAULT_X_TRAIN_PATH),
                        help="Path to input training features (X). (Default: %(default)s)")
    parser.add_argument('--y_path', type=str, default=str(DEFAULT_Y_TRAIN_PATH),
                        help="Path to input training labels (y). (Default: %(default)s)")
    parser.add_argument('--preprocessed_dir', type=str, default=str(DEFAULT_PREPROCESSED_DIR),
                        help="Directory to save/load preprocessed data. MUST be writeable. (Default: %(default)s)")
    parser.add_argument('--checkpoint_dir_base', type=str, default=str(DEFAULT_CHECKPOINT_DIR_BASE),
                        help="Base directory to save trial checkpoints. MUST be writeable. (Default: %(default)s)")
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for general outputs (logs, best params). MUST be writeable. (Default: %(default)s)")

    # --- Tuning Configuration Arguments ---
    parser.add_argument('--study_name', type=str, default=DEFAULT_STUDY_NAME,
                        help="Name for the Optuna study. (Default: %(default)s)")
    parser.add_argument('--storage', type=str, default=None, # Default to None, construct path later
                        help=f"Optuna storage URL (e.g., 'sqlite:///study.db'). If None, defaults to '<output_dir>/{DEFAULT_STORAGE_FILENAME}'.")
    parser.add_argument('--metric_direction', type=str, default='minimize', choices=['minimize', 'maximize'],
                        help="Direction to optimize the metric ('minimize' or 'maximize'). (Default: %(default)s)")

    args = parser.parse_args()

    # --- Process and Validate Paths ---
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir_base_path = Path(args.checkpoint_dir_base).resolve()
    preprocessed_dir_path = Path(args.preprocessed_dir).resolve()
    x_train_path = Path(args.X_path).resolve()
    y_train_path = Path(args.y_path).resolve()

    # Construct default storage path if not provided
    storage_url = args.storage
    if storage_url is None:
        storage_url = f"sqlite:///{output_dir / DEFAULT_STORAGE_FILENAME}"

    # --- Create Output Directories ---
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir_base_path.mkdir(parents=True, exist_ok=True)
        preprocessed_dir_path.mkdir(parents=True, exist_ok=True) # Ensure it exists for saving
        # Ensure parent dir for SQLite DB exists
        if storage_url.startswith("sqlite:///"):
            db_path = Path(storage_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directories: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Setup Logging (pointing to the correct output dir) ---
    log_file_path = output_dir / "tuning.log"
    log_file_path.touch() # Ensure log file exists

    # Remove default handlers and configure properly
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s](%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file_path, mode='a'), # Append mode
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__) # Get logger for this script

    logger.info("="*60)
    logger.info(" Starting Hyperparameter Tuning Script ".center(60, "="))
    logger.info("="*60)
    logger.info(f"Command Line Args: {vars(args)}")
    logger.info(f"Using Output Dir:          {output_dir}")
    logger.info(f"Using Checkpoint Base Dir: {checkpoint_dir_base_path}")
    logger.info(f"Using Preprocessed Dir:    {preprocessed_dir_path}")
    logger.info(f"Using Input X Path:        {x_train_path}")
    logger.info(f"Using Input y Path:        {y_train_path}")
    logger.info(f"Using Optuna Storage:      {storage_url}")
    logger.info("="*60)

    # --- Setup and Run Optuna Study ---
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=3) # Add min trials

    try:
        study = optuna.create_study(
            study_name=f"{args.study_name}_{args.model_type}", # Append model type
            storage=storage_url,
            direction=args.metric_direction,
            load_if_exists=True, # Resume study if it exists
            pruner=pruner
        )
    except Exception as e:
        logger.error(f"Failed to create or load Optuna study from storage '{storage_url}'. Error: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Optuna study '{study.study_name}' created/loaded successfully.")
    logger.info(f"Sampler: {study.sampler.__class__.__name__}, Pruner: {study.pruner.__class__.__name__}")
    logger.info(f"Number of trials already in study: {len(study.trials)}")
    logger.info(f"Target total trials for this run: {args.n_trials}")

    # --- Prepare Objective Function with Arguments ---
    # Pass the *validated paths* from args down to the objective function
    objective_func = lambda trial: objective(
        trial,
        model_type=args.model_type,
        x_path=x_train_path,
        y_path=y_train_path,
        preprocessed_dir=preprocessed_dir_path,
        checkpoint_dir_base=checkpoint_dir_base_path
    )

    # --- Run Optimization ---
    try:
        study.optimize(objective_func, n_trials=args.n_trials, timeout=None, gc_after_trial=True) # Add GC
    except KeyboardInterrupt:
         logger.warning("Optimization interrupted by user.")
    except Exception as e:
         logger.error(f"Optimization loop failed with unexpected error: {e}", exc_info=True)

    # --- Report Results ---
    logger.info("\n" + "="*60)
    logger.info(" Optimization Finished (or stopped) ".center(60, "="))
    logger.info("="*60)
    logger.info(f"Total number of trials in study '{study.study_name}': {len(study.trials)}")

    # Find best trial among completed ones
    try:
        completed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        if not completed_trials:
            logger.warning("No trials completed successfully. Cannot determine best trial.")
        else:
            best_trial = study.best_trial # Optuna finds the best among completed
            logger.info("--- Best Trial Found ---")
            logger.info(f"  Number: {best_trial.number}")
            logger.info(f"  Value ({args.metric_direction}d): {best_trial.value:.5f}")
            logger.info("  Params: ")
            for key, value in best_trial.params.items():
                logger.info(f"    {key}: {value}")

            # Save best parameters to the specified output directory
            best_params_file = output_dir / f"best_params_{args.model_type}.json"
            try:
                with open(best_params_file, 'w') as f:
                    json.dump(best_trial.params, f, indent=4)
                logger.info(f"Best parameters saved to: {best_params_file}")
            except OSError as e:
                logger.error(f"Failed to save best parameters to {best_params_file}. Error: {e}")

    except ValueError:
         # Should be caught by checking completed_trials, but as a fallback
         logger.warning("Could not determine best trial (e.g., no completed trials).")
    except Exception as e:
        logger.error(f"Error reporting best trial: {e}", exc_info=True)

    logger.info("="*60)
    logger.info(" Tuning Script Finished ".center(60, "="))
    logger.info("="*60)