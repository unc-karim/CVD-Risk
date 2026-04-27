import importlib
import os


def test_imports_work():
    os.environ["USE_DUMMY_MODEL"] = "1"
    from backend.app import config as config_module
    importlib.reload(config_module)
    from backend.app import main as main_module
    importlib.reload(main_module)
    assert main_module.app is not None
