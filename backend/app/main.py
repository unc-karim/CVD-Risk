"""
FastAPI application for CVD risk prediction

Main API with endpoints for:
- Health check
- Individual model predictions (HTN, CIMT, Vessel)
- Fusion prediction (complete CVD risk assessment)
"""

import io
import time
import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

from .config import settings
from .models.model_loader import model_loader
from .models.normalization import FeatureNormalizer
from .features.fusion_pipeline import FusionFeatureExtractor
from .api.schemas import (
    HealthCheckResponse,
    HTNPredictionResponse,
    CIMTValue,
    VesselFeatures,
    VesselSegmentationResponse,
    ContributingFactors,
    FusionPredictionResponse,
    RiskLevel,
    APIResponse
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# METRICS (OPTIONAL, PRIVACY-SAFE)
# ============================================================================

class MetricsStore:
    """In-memory, privacy-safe metrics store."""

    def __init__(self, model_version: str):
        self.model_version = model_version
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0

    def record(self, latency_ms: float, is_error: bool) -> None:
        self.request_count += 1
        if is_error:
            self.error_count += 1
        self.total_latency_ms += latency_ms

    def snapshot(self) -> dict:
        avg_latency = (
            self.total_latency_ms / self.request_count
            if self.request_count > 0
            else 0.0
        )
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "latency_ms": round(avg_latency, 2),
            "model_version": self.model_version,
        }


class DummyFusionExtractor:
    """Minimal extractor for dummy mode tests."""

    def extract_all_features(self, left_image, right_image, age, gender):
        features = np.zeros((1425,), dtype=np.float32)
        mask = np.zeros((512, 512), dtype=np.float32)
        metadata = {
            "htn_probability": 0.0,
            "cimt_prediction_mm": 0.0,
            "vessel_mask_left": mask,
            "vessel_density_avg": 0.0,
            "peripheral_density_avg": 0.0,
            "avg_vessel_width_avg": 0.0,
            "fractal_dimension_avg": 0.0,
            "branching_density_avg": 0.0,
            "avg_tortuosity_avg": 0.0,
        }
        return features, metadata

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""

    # ========== STARTUP ==========
    logger.info("=" * 80)
    logger.info("CVD RISK PREDICTION API - STARTUP")
    logger.info("=" * 80)

    try:
        if settings.USE_DUMMY_MODEL:
            logger.info("USE_DUMMY_MODEL=1 set; skipping weight validation")
        else:
            logger.info("Validating model paths...")
            settings.validate_model_paths()
            logger.info("✓ All model paths exist")

        resolved_paths = settings.resolve_model_paths()
        logger.info("\nLoading models...")
        model_loader.load_all_models(
            str(resolved_paths["HTN"]),
            str(resolved_paths["CIMT"]),
            str(resolved_paths["Vessel"]),
            str(resolved_paths["Fusion"])
        )

        logger.info("\nLoading normalization statistics...")
        app.state.normalizer = FeatureNormalizer(str(settings.NORMALIZATION_STATS_PATH))
        if not app.state.normalizer.is_ready():
            logger.warning(
                "⚠️  Normalization stats not loaded! "
                "Run: python scripts/compute_normalization_stats.py"
            )

        logger.info("\nInitializing feature extractors...")
        if settings.USE_DUMMY_MODEL:
            app.state.fusion_extractor = DummyFusionExtractor()
        else:
            app.state.fusion_extractor = FusionFeatureExtractor(
                htn_model=model_loader.get_model('htn'),
                cimt_model=model_loader.get_model('cimt'),
                vessel_model=model_loader.get_model('vessel'),
                device=model_loader.device
            )

        logger.info("=" * 80)
        logger.info("✓ ALL SYSTEMS READY")
        logger.info("=" * 80)

    except FileNotFoundError as e:
        logger.error(str(e))
        raise RuntimeError(str(e))
    except Exception as e:
        logger.error(f"STARTUP FAILED: {e}")
        raise

    yield

    # ========== SHUTDOWN ==========
    logger.info("\nShutting down...")
    model_loader.clear_cache()
    logger.info("✓ Shutdown complete")


# ============================================================================
# CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.ENABLE_METRICS:
    app.state.metrics = MetricsStore(model_version=settings.MODEL_VERSION)

    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000
        is_error = response.status_code >= 400
        app.state.metrics.record(latency_ms, is_error)
        return response

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_image(image_bytes: bytes) -> Image.Image:
    """Validate and load image from bytes"""

    try:
        # Check size
        if len(image_bytes) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Image too large (max {settings.MAX_IMAGE_SIZE_MB}MB)")

        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Check format
        if image.format and image.format.lower() not in settings.ALLOWED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format: {image.format}")

        return image

    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")


def mask_to_base64(mask: np.ndarray) -> str:
    """
    Convert segmentation mask to base64-encoded PNG

    Displays raw continuous probability values (0-1) from the model.
    Directly scales probabilities to 0-255 grayscale for display.
    """

    # Scale continuous probabilities [0, 1] to [0, 255] grayscale
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Convert to PIL Image (grayscale mode 'L')
    mask_image = Image.fromarray(mask_uint8, mode='L')

    # Encode to PNG format
    buffer = io.BytesIO()
    mask_image.save(buffer, format='PNG')
    buffer.seek(0)

    # Base64 encode to data URI
    b64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64_string}"

def ensure_fusion_extractor() -> None:
    """Ensure fusion extractor is initialized (dummy mode safe)."""

    if hasattr(app.state, "fusion_extractor"):
        return
    if settings.USE_DUMMY_MODEL:
        model_loader.load_htn_model(str(settings.HTN_CHECKPOINT))
        model_loader.load_cimt_model(str(settings.CIMT_CHECKPOINT))
        model_loader.load_vessel_model(str(settings.VESSEL_CHECKPOINT))
        model_loader.load_fusion_model(str(settings.FUSION_CHECKPOINT))
        app.state.fusion_extractor = DummyFusionExtractor()
        return
    raise RuntimeError("Fusion extractor not initialized. Ensure startup completed.")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""

    return HealthCheckResponse(
        status="healthy" if model_loader.is_ready() else "unhealthy",
        models_loaded=model_loader.is_ready(),
        models=list(model_loader._models.keys())
    )

@app.get("/metrics")
async def metrics():
    """Privacy-safe metrics endpoint"""

    if not settings.ENABLE_METRICS:
        return {"enabled": False}
    return app.state.metrics.snapshot()


@app.post("/api/predict/htn", response_model=APIResponse)
async def predict_htn(file: UploadFile = File(...)):
    """
    Hypertension classification from fundus image

    Inputs:
    - file: Fundus image (PNG or JPG)

    Returns:
    - 0: Normal (no hypertension)
    - 1: Hypertensive (hypertension detected)
    """

    start_time = time.time()

    try:
        ensure_fusion_extractor()
        # Validate image
        image_bytes = await file.read()
        image = validate_image(image_bytes)

        # Extract features
        htn_extractor = app.state.fusion_extractor.htn_extractor
        prob, embedding, _ = htn_extractor.extract(image)

        # Get optimal threshold from model config
        htn_config = model_loader.get_config('htn')
        optimal_threshold = htn_config.get('optimal_threshold', 0.5)

        # Use optimal threshold for prediction (CRITICAL!)
        prediction = 1 if prob >= optimal_threshold else 0

        # Determine confidence level
        if prob >= 0.7 or prob <= 0.3:
            confidence = "High"
        else:
            confidence = "Medium"

        # Determine label
        label = "Hypertensive" if prediction == 1 else "Normal"

        result = HTNPredictionResponse(
            prediction=prediction,
            probability=float(prob),
            confidence=confidence,
            label=label
        )

        return APIResponse(
            status="success",
            result=result.model_dump(),
            processing_time=time.time() - start_time
        )

    except Exception as e:
        logger.error(f"HTN prediction failed: {e}")
        return APIResponse(
            status="error",
            error=str(e),
            processing_time=time.time() - start_time
        )


@app.post("/api/predict/cimt", response_model=APIResponse)
async def predict_cimt(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    age: int = Form(...),
    gender: int = Form(...)
):
    """
    CIMT (Carotid Intima-Media Thickness) regression

    Inputs:
    - left_image: Left eye fundus image
    - right_image: Right eye fundus image
    - age: Age in years
    - gender: 0=Female, 1=Male

    Returns:
    - cimt_value: CIMT in mm
    - risk_category: Normal/Elevated/High
    - threshold: Clinical threshold (0.9mm)
    """

    start_time = time.time()

    try:
        ensure_fusion_extractor()
        # Validate inputs
        if age < 1 or age > 150:
            raise ValueError("Age must be between 1 and 150")
        if gender not in [0, 1]:
            raise ValueError("Gender must be 0 (Female) or 1 (Male)")

        # Load images
        left_bytes = await left_image.read()
        left_img = validate_image(left_bytes)

        right_bytes = await right_image.read()
        right_img = validate_image(right_bytes)

        # Extract features
        cimt_extractor = app.state.fusion_extractor.cimt_extractor
        pred, embedding, _ = cimt_extractor.extract(left_img, right_img, age, gender)

        # Categorize risk
        threshold = 0.9
        if pred < 0.7:
            risk_category = "Normal"
        elif pred < threshold:
            risk_category = "Borderline"
        else:
            risk_category = "Elevated"

        result = CIMTValue(
            value_mm=float(pred),
            risk_category=risk_category,
            threshold_mm=threshold,
            clinical_significance="CIMT ≥0.9mm indicates increased cardiovascular risk"
        )

        return APIResponse(
            status="success",
            result=result.model_dump(),
            processing_time=time.time() - start_time
        )

    except Exception as e:
        logger.error(f"CIMT prediction failed: {e}")
        return APIResponse(
            status="error",
            error=str(e),
            processing_time=time.time() - start_time
        )


@app.post("/api/predict/vessel", response_model=APIResponse)
async def predict_vessel(file: UploadFile = File(...)):
    """
    Vessel segmentation and feature extraction

    Inputs:
    - file: Fundus image

    Returns:
    - vessel_density: Overall vessel density
    - features: Clinical vessel features
    - segmentation_mask_base64: Base64-encoded segmentation mask
    """

    start_time = time.time()

    try:
        ensure_fusion_extractor()
        # Validate image
        image_bytes = await file.read()
        image = validate_image(image_bytes)

        # Extract features
        vessel_extractor = app.state.fusion_extractor.vessel_extractor
        learned, clinical, mask, _ = vessel_extractor.extract(image)

        # Prepare features response
        features = VesselFeatures(
            vessel_density=float(clinical[0]),
            peripheral_density=float(clinical[1]),
            avg_vessel_width=float(clinical[8]),
            fractal_dimension=float(clinical[11]),
            branching_density=float(clinical[12]),
            avg_tortuosity=float(clinical[6])
        )

        # Convert mask to base64
        mask_b64 = mask_to_base64(mask)

        result = VesselSegmentationResponse(
            vessel_density=float(clinical[0]),
            features=features,
            segmentation_mask_base64=mask_b64
        )

        return APIResponse(
            status="success",
            result=result.model_dump(),
            processing_time=time.time() - start_time
        )

    except Exception as e:
        logger.error(f"Vessel prediction failed: {e}")
        return APIResponse(
            status="error",
            error=str(e),
            processing_time=time.time() - start_time
        )


@app.post("/api/predict/fusion", response_model=APIResponse)
async def predict_fusion(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    age: int = Form(...),
    gender: int = Form(...)
):
    """
    Complete CVD risk prediction from fusion of all models

    MAIN ENDPOINT - Combines HTN, CIMT, and Vessel models

    Inputs:
    - left_image: Left eye fundus image
    - right_image: Right eye fundus image
    - age: Age in years
    - gender: 0=Female, 1=Male

    Returns:
    - cvd_risk_prediction: 0=Low Risk, 1=High Risk
    - cvd_probability: CVD risk probability [0, 1]
    - risk_level: Low/Medium/High
    - hypertension: HTN classification result
    - cimt: CIMT regression result
    - vessel: Vessel segmentation result
    - contributing_factors: Breakdown of factors
    - recommendation: Clinical recommendation
    """

    start_time = time.time()

    try:
        ensure_fusion_extractor()
        # Validate inputs
        if age < 1 or age > 150:
            raise ValueError("Age must be between 1 and 150")
        if gender not in [0, 1]:
            raise ValueError("Gender must be 0 (Female) or 1 (Male)")

        # Get model configs early (needed for thresholds and standardization)
        htn_config = model_loader.get_config('htn')

        # Load images
        left_bytes = await left_image.read()
        left_img = validate_image(left_bytes)

        right_bytes = await right_image.read()
        right_img = validate_image(right_bytes)

        # Extract 1425 features
        features_1425, metadata = app.state.fusion_extractor.extract_all_features(
            left_img, right_img, age, gender
        )

        # Get standardization parameters from fusion model config (CRITICAL!)
        fusion_config = model_loader.get_config('fusion')
        if 'fusion_mean' in fusion_config and 'fusion_std' in fusion_config:
            fusion_mean = fusion_config['fusion_mean']
            fusion_std = fusion_config['fusion_std']

            # Verify dimension match
            if features_1425.shape[0] != fusion_mean.shape[0]:
                raise ValueError(
                    f"Feature dimension mismatch! Expected {fusion_mean.shape[0]}, "
                    f"got {features_1425.shape[0]}"
                )

            # Standardize using saved parameters
            features_normalized = (features_1425 - fusion_mean) / (fusion_std + 1e-8)
            logger.debug(f"Features standardized using checkpoint parameters")
        else:
            logger.error("Fusion model missing standardization parameters!")
            raise ValueError("Invalid fusion model checkpoint - missing standardization parameters")

        # Predict CVD risk
        import torch
        fusion_model = model_loader.get_model('fusion')
        features_tensor = torch.tensor(
            features_normalized,
            dtype=torch.float32,
            device=model_loader.device
        ).unsqueeze(0)

        with torch.no_grad():
            logits = fusion_model(features_tensor)
            cvd_prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

        cvd_prediction = 1 if cvd_prob >= 0.5 else 0

        # Determine risk level
        if cvd_prob >= 0.7:
            risk_level = RiskLevel.HIGH
            recommendation = (
                "High CVD risk detected. Please consult with a cardiologist "
                "for further evaluation and preventive measures."
            )
        elif cvd_prob >= 0.4:
            risk_level = RiskLevel.MEDIUM
            recommendation = (
                "Medium CVD risk. Consider lifestyle modifications and "
                "regular follow-up with healthcare provider."
            )
        else:
            risk_level = RiskLevel.LOW
            recommendation = (
                "Low CVD risk based on current assessment. "
                "Maintain healthy lifestyle and regular checkups."
            )

        # Prepare individual results
        # Use optimal threshold for HTN (from model config)
        htn_optimal_threshold = htn_config.get('optimal_threshold', 0.5)
        htn_result = HTNPredictionResponse(
            prediction=1 if metadata['htn_probability'] >= htn_optimal_threshold else 0,
            probability=float(metadata['htn_probability']),
            confidence="High" if metadata['htn_probability'] >= 0.7 else "Medium",
            label="Hypertensive" if metadata['htn_probability'] >= htn_optimal_threshold else "Normal"
        )

        cimt_threshold = 0.9
        cimt_result = CIMTValue(
            value_mm=float(metadata['cimt_prediction_mm']),
            risk_category="Elevated" if metadata['cimt_prediction_mm'] >= cimt_threshold else "Normal",
            threshold_mm=cimt_threshold,
            clinical_significance="CIMT ≥0.9mm indicates increased CVD risk"
        )

        vessel_mask_b64 = mask_to_base64(metadata['vessel_mask_left'])
        vessel_result = VesselSegmentationResponse(
            vessel_density=float(metadata['vessel_density_avg']),
            features=VesselFeatures(
                vessel_density=float(metadata['vessel_density_avg']),
                peripheral_density=float(metadata['peripheral_density_avg']),
                avg_vessel_width=float(metadata['avg_vessel_width_avg']),
                fractal_dimension=float(metadata['fractal_dimension_avg']),
                branching_density=float(metadata['branching_density_avg']),
                avg_tortuosity=float(metadata['avg_tortuosity_avg'])
            ),
            segmentation_mask_base64=vessel_mask_b64
        )

        contributing = ContributingFactors(
            hypertension_probability=float(metadata['htn_probability']),
            cimt_elevated=float(metadata['cimt_prediction_mm']) >= cimt_threshold,
            vessel_abnormalities=1.0 - float(metadata['vessel_density_avg'])
        )

        result = FusionPredictionResponse(
            cvd_risk_prediction=cvd_prediction,
            cvd_probability=float(cvd_prob),
            risk_level=risk_level,
            hypertension=htn_result,
            cimt=cimt_result,
            vessel=vessel_result,
            contributing_factors=contributing,
            recommendation=recommendation,
            processing_time_seconds=time.time() - start_time
        )

        return APIResponse(
            status="success",
            result=result.model_dump(),
            processing_time=time.time() - start_time
        )

    except Exception as e:
        logger.error(f"Fusion prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return APIResponse(
            status="error",
            error=str(e),
            processing_time=time.time() - start_time
        )


# ============================================================================
# TEST ENDPOINTS
# ============================================================================

@app.get("/api/test/transforms")
async def test_transforms():
    """Test that transforms produce correct outputs"""
    import torch
    from PIL import Image
    from .preprocessing.transforms import transform_htn, transform_cimt, transform_vessel

    # Create test image
    test_img = Image.new('RGB', (1024, 1024), color=(128, 128, 128))

    # Test transforms
    htn_out = transform_htn(test_img)
    cimt_out = transform_cimt(test_img)
    vessel_out = transform_vessel(test_img)

    return {
        'htn': {
            'shape': list(htn_out.shape),
            'range': [float(htn_out.min()), float(htn_out.max())],
            'expected': 'Shape [3, 224, 224], Range ~ [-2, 2]',
            'correct': htn_out.shape == torch.Size([3, 224, 224])
        },
        'cimt': {
            'shape': list(cimt_out.shape),
            'range': [float(cimt_out.min()), float(cimt_out.max())],
            'expected': 'Shape [3, 512, 512], Range ~ [-2, 2]',
            'correct': cimt_out.shape == torch.Size([3, 512, 512])
        },
        'vessel': {
            'shape': list(vessel_out.shape),
            'range': [float(vessel_out.min()), float(vessel_out.max())],
            'expected': 'Shape [3, 512, 512], Range [0, 1]',
            'correct': vessel_out.shape == torch.Size([3, 512, 512]) and
                      vessel_out.min() >= 0 and vessel_out.max() <= 1
        }
    }


@app.get("/api/test/model_configs")
async def test_model_configs():
    """Test that model configs are loaded correctly"""

    htn_config = model_loader.get_config('htn')
    fusion_config = model_loader.get_config('fusion')

    return {
        'htn': {
            'has_optimal_threshold': 'optimal_threshold' in htn_config,
            'optimal_threshold': htn_config.get('optimal_threshold', 'NOT FOUND'),
            'threshold_info': htn_config.get('threshold_info', {})
        },
        'fusion': {
            'has_standardization': all(k in fusion_config for k in ['fusion_mean', 'fusion_std']),
            'mean_shape': str(fusion_config.get('fusion_mean', np.array([])).shape) if 'fusion_mean' in fusion_config else 'NOT FOUND',
            'std_shape': str(fusion_config.get('fusion_std', np.array([])).shape) if 'fusion_std' in fusion_config else 'NOT FOUND',
            'expected_dim': 1425
        }
    }


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API info"""

    return {
        "title": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "endpoints": {
            "htn": "POST /api/predict/htn",
            "cimt": "POST /api/predict/cimt",
            "vessel": "POST /api/predict/vessel",
            "fusion": "POST /api/predict/fusion"
        },
        "test_endpoints": {
            "transforms": "GET /api/test/transforms",
            "model_configs": "GET /api/test/model_configs"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
