import importlib.util
import sys

from pathlib import Path
from types import ModuleType

MODULES_DIR = Path(__file__).resolve().parents[1] / "inference-modules"


def load_module_file(module_dir: str, name: str) -> ModuleType:
    """Import a file of an inference module, whose directory is not a package and reuses names across modules."""
    spec = importlib.util.spec_from_file_location(f"{module_dir}_{name}", MODULES_DIR / module_dir / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
