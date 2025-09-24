"""
Environment setup and validation for IS-160 CNN Employment Trends Analysis Project.

This module handles PyTorch environment setup, dependency checking, and
provides utilities for environment management.
"""

import sys
import subprocess
import importlib
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Required packages and their minimum versions
REQUIRED_PACKAGES = {
    'torch': '2.0.0',
    'pandas': '1.5.0',
    'sklearn': '1.2.0',  # scikit-learn
    'matplotlib': '3.6.0',
    'seaborn': '0.12.0',
    'numpy': '1.21.0',
    'requests': '2.28.0'
}

def check_package_version(package_name: str, min_version: str) -> bool:
    """
    Check if a package is installed and meets the minimum version requirement.

    Args:
        package_name: Name of the package to check
        min_version: Minimum required version

    Returns:
        bool: True if package is installed and version is sufficient
    """
    try:
        # Handle scikit-learn import
        if package_name == 'sklearn':
            import sklearn
            version = sklearn.__version__
        else:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', '0.0.0')

        # Simple version comparison (could be enhanced)
        installed_parts = [int(x) for x in version.split('.') if x.isdigit()]
        required_parts = [int(x) for x in min_version.split('.') if x.isdigit()]

        # Pad shorter version to same length
        max_len = max(len(installed_parts), len(required_parts))
        installed_parts.extend([0] * (max_len - len(installed_parts)))
        required_parts.extend([0] * (max_len - len(required_parts)))

        if installed_parts >= required_parts:
            logger.info(f"✓ {package_name} {version} >= {min_version}")
            return True
        else:
            logger.warning(f"✗ {package_name} {version} < {min_version}")
            return False

    except ImportError:
        logger.error(f"✗ {package_name} not installed")
        return False
    except Exception as e:
        logger.error(f"Error checking {package_name}: {e}")
        return False

def check_pytorch_setup() -> bool:
    """
    Perform additional PyTorch-specific checks.

    Returns:
        bool: True if PyTorch setup is valid
    """
    try:
        import torch

        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
        else:
            logger.info("ℹ CUDA not available, using CPU")

        # Check MPS availability (Apple Silicon)
        if hasattr(torch, 'mps') and torch.mps.is_available():
            logger.info("✓ MPS (Apple Silicon) available")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("✓ MPS (Apple Silicon) available")

        return True

    except Exception as e:
        logger.error(f"Error checking PyTorch setup: {e}")
        return False

def install_requirements(requirements_file: str = "requirements.txt") -> bool:
    """
    Install requirements from a requirements.txt file.

    Args:
        requirements_file: Path to requirements file

    Returns:
        bool: True if installation successful
    """
    if not Path(requirements_file).exists():
        logger.error(f"Requirements file {requirements_file} not found")
        return False

    try:
        logger.info(f"Installing requirements from {requirements_file}")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], capture_output=True, text=True, check=True)

        logger.info("Requirements installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False

def validate_environment() -> bool:
    """
    Validate that all required packages are installed and meet version requirements.

    Returns:
        bool: True if environment is valid
    """
    logger.info("Validating Python environment...")

    all_good = True

    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        logger.info(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        logger.error(f"✗ Python {python_version.major}.{python_version.minor}.{python_version.micro} < 3.8 required")
        all_good = False

    # Check required packages
    for package, min_version in REQUIRED_PACKAGES.items():
        if not check_package_version(package, min_version):
            all_good = False

    # PyTorch-specific checks
    if not check_pytorch_setup():
        all_good = False

    if all_good:
        logger.info("✓ Environment validation passed")
    else:
        logger.error("✗ Environment validation failed")

    return all_good

def setup_environment(auto_install: bool = False) -> bool:
    """
    Set up the Python environment for the project.

    Args:
        auto_install: Whether to automatically install missing requirements

    Returns:
        bool: True if setup successful
    """
    logger.info("Setting up IS-160 CNN Employment Trends Analysis environment...")

    # First validate current environment
    if validate_environment():
        logger.info("Environment is already properly configured")
        return True

    if not auto_install:
        logger.info("Environment validation failed. Run with auto_install=True to install requirements.")
        return False

    # Try to install requirements
    if install_requirements():
        # Re-validate after installation
        return validate_environment()
    else:
        logger.error("Failed to install requirements")
        return False

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="IS-160 Environment Setup")
    parser.add_argument("--install", action="store_true",
                       help="Automatically install missing requirements")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate environment without setup")

    args = parser.parse_args()

    if args.validate_only:
        success = validate_environment()
    else:
        success = setup_environment(auto_install=args.install)

    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)