import importlib
import io
import os

from fastapi.testclient import TestClient
from PIL import Image


def _get_client():
    os.environ["USE_DUMMY_MODEL"] = "1"
    from backend.app import config as config_module
    importlib.reload(config_module)
    from backend.app import main as main_module
    importlib.reload(main_module)
    return TestClient(main_module.app)


def _dummy_image_bytes():
    image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def test_health_endpoint():
    client = _get_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "models_loaded" in payload
    assert "models" in payload


def test_fusion_schema_dummy():
    client = _get_client()
    image_bytes = _dummy_image_bytes()
    files = {
        "left_image": ("left.png", image_bytes, "image/png"),
        "right_image": ("right.png", image_bytes, "image/png"),
    }
    data = {"age": 50, "gender": 1}
    response = client.post("/api/predict/fusion", files=files, data=data)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    result = payload["result"]
    assert "cvd_risk_prediction" in result
    assert "cvd_probability" in result
    assert "risk_level" in result
    assert "hypertension" in result
    assert "cimt" in result
    assert "vessel" in result
    assert "contributing_factors" in result
    assert "recommendation" in result
    assert "processing_time_seconds" in result
