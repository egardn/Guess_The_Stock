# CFM Data Challenge (2024) - Guess the Stock

## 1. Project Description

This project tackles the Capital Fund Management (CFM) Data Challenge from 2024. The primary goal is to identify which stock a given sequence of tick-by-tick exchange data belongs to. This is framed as a **multi-class classification problem** with 24 distinct stock categories (0-23).

The input data consists of sequential order book updates from an aggregated order book across multiple exchanges. While the data is anonymized, characteristics like spread, typical quantities, trade frequency, and venue distribution are expected to hold clues to the stock's identity.

This repository explores two modeling approaches:

1. **Gradient Boosting** model (LightGBM) using engineered features derived from the time series.
2. **Gated Recurrent Unit (GRU)** neural network that processes the raw sequence data directly.

## 2. Data Description

### Input Data (X)

The dataset consists of sequences, each containing 100 consecutive order book events (`obs_id`). The data includes:

- **`obs_id`**: Uniquely identifies a sequence of 100 events for a specific stock on a specific day.
- **`venue`**: Integer encoding for the exchange where the event occurred.
- **`action`**: Type of order book event:
  - `'A'`: Add - Volume added (new order).
  - `'D'`: Delete - Order deleted.
  - `'U'`: Update - Order updated.
- **`order_id`**: Unique identifier for a specific order within its lifetime (obfuscated, starts at 0 for the first order seen in an `obs_id`).
- **`side`**: Side of the order book: `'A'` (Ask/Offer) or `'B'` (Bid).
- **`price`**: Price of the order affected by the event.
- **`bid`**: Best bid price in the aggregated book *after* the event.
- **`ask`**: Best ask price in the aggregated book *after* the event.
- **`bid_size`**: Total volume available at the best bid price.
- **`ask_size`**: Total volume available at the best ask price.
- **`flux`**: Change in volume at the price level affected by the event (positive for additions, negative for deletions/updates reducing volume).
- **`trade`**: Boolean indicating if a deletion or update event was caused by a trade (`True`) or a cancellation (`False`).

**Normalization Note:** As per the challenge description, the `price`, `bid`, and `ask` fields are normalized by subtracting the `bid` price of the *first event* in the sequence.

### Target Data (Y)

- **`eqt_code_cat`**: An integer between 0 and 23 representing the unique stock identifier (the target variable for classification).

### Train/Test Split

The training set is drawn from one period, while the test set observations are drawn from a different, future period using the same stocks.

## 3. Modeling Approach

### 3.1. Gradient Boosting (GB)

- Uses LightGBM.
- Relies on feature engineering performed by `GBPipeline` and `GBFeatureExtractor` to create summary statistics from the 100-event sequences.
- The specific features used can be configured in the notebook.

### 3.2. Gated Recurrent Unit (GRU)

This model processes the sequence data more directly:

1. **Input Preprocessing**: Each observation is transformed into a tensor of shape `(100, 30)`.
2. **Input Vector (30 dimensions)**: Composed of:
   - Venue Embedding (8 dims)
   - Action Embedding (8 dims)
   - Trade Embedding (8 dims)
   - Normalized Bid (1 dim)
   - Normalized Ask (1 dim)
   - Normalized Price (1 dim)
   - `log(bid_size + 1)` (1 dim)
   - `log(ask_size + 1)` (1 dim)
   - `log(flux)` (1 dim) - *Note: Handling potential zero or negative flux values for the log transform needs care.*
3. **Recurrent Layers**: The sequence tensor is processed by two **128-unit** GRU layers with **dropout** (`gru_dropout` parameter in notebook):
   - One forward pass.
   - One backward pass.
4. **Concatenation**: The final hidden states of the forward and backward GRUs are concatenated into a 256-dimensional vector.
5. **Dense Layers**:
   - Dense layer: 256 -> 64 units (parameter `dense_units` in notebook) with activation (e.g., ReLU or SeLU).
   - Output layer: 64 -> 24 units (logits for each stock class).
6. **Output Activation**: Softmax is applied to the logits to get class probabilities.

## 4. Installation

This project is designed to run with **Python 3.12**. Ensure that this version is installed on your computer before proceeding. You can verify your Python version by running:

```bash
python --version
```

If Python 3.12 is not installed, download and install it from the [official Python website](https://www.python.org/downloads/).

### Steps to Set Up the Environment

The easiest way to set up the environment and launch the API is by using the provided `launch_api.bat` script. This script will:

1. Check for Python 3.12.
2. Create a virtual environment (if it doesn't already exist).
3. Install the required dependencies from `requirements.txt`.
4. Set the necessary environment variables for models and pipelines.
5. Launch the API server.

To use the script, simply run:

```cmd
launch_api.bat
```

### Manual Setup (Optional)

If you prefer to set up the environment manually, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/egardn/Guess_The_Stock.git
   cd <your-repo-directory>
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/macOS
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set environment variables for models and pipelines:
   ```bash
   set GRU_MODEL_PATH=..\models\gru\final_model_gru.pkl
   set GRU_PIPELINE_PATH=..\data\preprocessed_data\gru_pipeline.pkl
   set GB_MODEL_PATH=..\models\gb\final_model_gb.pkl
   set GB_PIPELINE_PATH=..\data\preprocessed_data\gb_pipeline.pkl
   ```

5. (Optional) Force TensorFlow to use CPU mode:
   ```bash
   set CUDA_VISIBLE_DEVICES=-1
   ```

6. Launch the API:
   ```bash
   uvicorn gts_challenge.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

## 5. Usage

### 5.1. Running the Analysis Notebook

The main analysis, including data visualization, model training (GB and GRU), evaluation, and payload generation, is performed in the Jupyter notebook.

### 5.2. Running the Prediction API

An API is provided for serving predictions and explanations.

#### Set Environment Variables (Important!)

The API relies on environment variables to find the trained model and pipeline files. Set these before launching (example paths, adjust as needed):

```bash
# Linux/macOS
export GRU_MODEL_PATH=../models/gru/final_model_gru.pkl
export GRU_PIPELINE_PATH=../data/preprocessed_data/gru_pipeline.pkl
export GB_MODEL_PATH=../models/gb/final_model_gb.pkl
export GB_PIPELINE_PATH=../data/preprocessed_data/gb_pipeline.pkl

# Windows (Command Prompt)
set GRU_MODEL_PATH=..\models\gru\final_model_gru.pkl
set GRU_PIPELINE_PATH=..\data\preprocessed_data\gru_pipeline.pkl
set GB_MODEL_PATH=..\models\gb\final_model_gb.pkl
set GB_PIPELINE_PATH=..\data\preprocessed_data\gb_pipeline.pkl

# Windows (PowerShell)
$env:GRU_MODEL_PATH = "..\models\gru\final_model_gru.pkl"
$env:GRU_PIPELINE_PATH = "..\data\preprocessed_data\gru_pipeline.pkl"
$env:GB_MODEL_PATH = "..\models\gb\final_model_gb.pkl"
$env:GB_PIPELINE_PATH = "..\data\preprocessed_data\gb_pipeline.pkl"
```

#### Launch the API

Use the provided batch script or run uvicorn directly from the project root directory:

```bash
./launch_api.bat  # If using the batch script
# or
uvicorn gts_challenge.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at [http://localhost:8000](http://localhost:8000) (or your machine's IP). Documentation and API swagger is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 5.3. Sample Payloads

Sample JSON payloads for interacting with the API (`/predict` and `/explain` endpoints) are generated by the notebook and saved at the root of the directory as:

- `sample_payload_gb.json`
- `sample_payload_gru.json`

To use the payloads in the Swagger of the API, you just need to copy paste the corresponding JSON in the `/predict` and `/explain` endpoints.

## 6. Project Structure

```
.
├── data/                 # Raw and preprocessed data
│   ├── preprocessed_data/ # Stored features, labels, pipelines
│   ├── X_test.parquet
│   ├── X_train.parquet
│   └── y_train.parquet
├── gts_challenge/        # Main project source code
│   ├── api/              # FastAPI application code
│   └── order_book/       # Core logic for data, models, pipelines, workflows
│       ├── base/
│       ├── data/
│       ├── models/
│       ├── utils/
│       └── workflows/
├── models/               # Saved trained model checkpoints/final models
│   ├── gb/
│   └── gru/
├── notebooks/            # Jupyter notebooks
│   ├── project.ipynb     # Main analysis notebook
│   └── sample_payload_*.json # Example API request bodies
├── .gitignore
├── launch_api.bat        # Script to launch the API (Windows)
├── readme.md             # This file
└── requirements.txt      # Python dependencies
```
