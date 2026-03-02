# Activation script for Self-Improving RAG System (PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Self-Improving RAG System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host "[ERROR] Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "[SUCCESS] Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Yellow
Write-Host "  python examples.py          - Run examples"
Write-Host "  python main.py dashboard    - Launch dashboard"
Write-Host "  python main.py ingest PATH  - Ingest documents"
Write-Host "  python main.py query `"...`"  - Query the system"
Write-Host "  python main.py stats        - View statistics"
Write-Host ""
Write-Host "To deactivate: deactivate" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
