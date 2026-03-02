@echo off
REM Activation script for Self-Improving RAG System

echo ========================================
echo  Self-Improving RAG System
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [SUCCESS] Virtual environment activated!
echo.
echo Available commands:
echo   python examples.py          - Run examples
echo   python main.py dashboard    - Launch dashboard
echo   python main.py ingest PATH  - Ingest documents
echo   python main.py query "..."  - Query the system
echo   python main.py stats        - View statistics
echo.
echo To deactivate: deactivate
echo ========================================
echo.
