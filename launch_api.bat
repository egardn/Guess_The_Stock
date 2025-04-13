@echo off
setlocal

REM --- Configuration ---
set PYTHON_VERSION=3.12
set VENV_NAME=.venv
set "SCRIPT_DIR=%~dp0"
set "VENV_PATH=%SCRIPT_DIR%%VENV_NAME%"
set "REQUIREMENTS_PATH=%SCRIPT_DIR%requirements.txt"

REM --- Check for Python ---
echo Checking for Python %PYTHON_VERSION%...
python --version 2>&1 | findstr /C:"Python %PYTHON_VERSION%" > nul
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python %PYTHON_VERSION% not found in PATH. Please install or activate the correct environment.
    goto :eof
)

REM --- Check/Create Virtual Environment ---
IF NOT EXIST "%VENV_PATH%\Scripts\activate.bat" (
    echo Creating virtual environment in "%VENV_PATH%"...
    python -m venv "%VENV_PATH%"
    IF %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment.
        goto :eof
    )
    echo Virtual environment created.
) ELSE (
    echo Virtual environment found in "%VENV_PATH%".
)

REM --- Activate Virtual Environment ---
echo Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment. Check the venv installation.
    goto :eof
)
echo Virtual environment activated.

REM --- Set environment variables for models and pipelines ---
echo Setting environment variables for models and pipelines...
set "GRU_MODEL_PATH=%SCRIPT_DIR%models\gru\final_model_gru.pkl"
set "GRU_PIPELINE_PATH=%SCRIPT_DIR%data\preprocessed_data\gru_pipeline.pkl"
set "GB_MODEL_PATH=%SCRIPT_DIR%models\gb\final_model_gb.pkl"
set "GB_PIPELINE_PATH=%SCRIPT_DIR%data\preprocessed_data\gb_pipeline.pkl"

REM --- Install dependencies ---
echo Installing dependencies from "%REQUIREMENTS_PATH%"...
python -m pip install -r "%REQUIREMENTS_PATH%"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies. Check requirements.txt and network connection.
    goto :eof
)

REM --- Force TensorFlow to use CPU (if needed) ---
echo Forcing TensorFlow to use CPU mode...
set CUDA_VISIBLE_DEVICES=-1

REM --- Launch the API ---
echo Starting API server...
python -m uvicorn gts_challenge.api.main:app --host 0.0.0.0 --port 8000 --reload

REM --- End of script ---
echo API server process finished or stopped.

:eof
endlocal