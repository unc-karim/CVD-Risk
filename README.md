# Cardiovascular Disease Risk Prediction System

Research demo for cardiovascular risk assessment from retinal fundus images. It serves a FastAPI backend and React frontend that combine three model outputs with a fusion classifier to produce a single risk score. This repository is intended for research and education only.

## Quickstart

### macOS/Linux
```bash
git clone <repo-url>
cd CVD-Risk
python scripts/check_weights.py

docker-compose up --build
```

### Windows (PowerShell)
```powershell
git clone <repo-url>
cd CVD-Risk
python scripts/check_weights.py

docker-compose up --build
```

Open:
- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Project Structure

```
CVD-Risk/
├── backend/                 # FastAPI app, model loading, feature extraction
├── frontend/                # React UI
├── notebooks/               # Research notebooks (moved out of repo root)
├── scripts/                 # Utilities (weights checks, downloads)
├── tests/                   # Pytest suite (dummy-mode compatible)
├── weights/                 # Model checkpoints (not tracked)
├── docker-compose.yml
├── requirements-dev.txt
└── README.md
```

## Weights

Place trained weights in `weights/` and run:
```
python scripts/check_weights.py
```

If your weights are elsewhere, set `WEIGHTS_DIR` to that folder.

## Model Outputs

The pipeline returns the following high-level outputs:
- Hypertension (HTN): classification label + probability
- CIMT: regression value in mm with a simple risk category
- Vessel segmentation: segmentation mask + clinical vessel features
- Fusion score: combined CVD risk probability and risk level

## Privacy & Medical Disclaimer

- Images are processed in-memory and are not persisted to disk.
- Optional metrics are aggregate-only (request_count, latency_ms, error_count, model_version) and are OFF by default. Enable with `ENABLE_METRICS=1`.
- This is a research tool and is not approved for clinical diagnosis or treatment decisions.

## Development

Backend (macOS/Linux):
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend (PowerShell):
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:
```bash
cd frontend
npm install
REACT_APP_API_URL=http://localhost:8000 npm start
```

## Testing

Run tests without large weights:
```bash
USE_DUMMY_MODEL=1 pytest
```

Sync requirements from repo imports:
```bash
python scripts/extract_requirements.py --dry-run
```

## License

MIT (see LICENSE).
