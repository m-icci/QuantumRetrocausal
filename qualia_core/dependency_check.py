"""
Dependency checker for quantum core modules.
"""
import sys
import importlib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_imports(module_name: str, seen=None):
    """Check imports for a given module."""
    if seen is None:
        seen = set()

    if module_name in seen:
        logger.warning(f"⚠️ Circular import detected: {module_name}")
        return

    seen.add(module_name)

    try:
        module = importlib.import_module(module_name)
        logger.info(f"✓ Successfully imported: {module_name}")

        # Check submodules
        if hasattr(module, '__path__'):
            pkg_path = Path(module.__file__).parent
            for item in pkg_path.glob('*.py'):
                if item.stem != '__init__':
                    submodule = f"{module_name}.{item.stem}"
                    check_imports(submodule, seen)

    except ImportError as e:
        logger.error(f"❌ Import failed for {module_name}: {str(e)}")
    except Exception as e:
        logger.error(f"⚠️ Error checking {module_name}: {str(e)}")

def main():
    """Run dependency check on quantum core modules."""
    # Add project root to Python path
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    core_modules = [
        'quantum.core.qtypes',  # Updated from types to qtypes
        'quantum.core.QUALIA',
        'quantum.core.portfolio.sacred.retrocausal_trading',
        'quantum.experiments.hawking_radiation.quantum.test_consciousness'
    ]

    logger.info("Starting dependency check...")
    for module in core_modules:
        check_imports(module)

if __name__ == '__main__':
    main()
