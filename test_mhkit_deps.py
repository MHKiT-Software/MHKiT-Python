import subprocess
import sys
import os
import shutil
from pathlib import Path


def run_command(command):
    """Run a shell command and print its output"""
    print(f"\nExecuting: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0


def cleanup_venv(venv_name):
    """Remove a virtual environment if it exists"""
    if os.path.exists(venv_name):
        print(f"\nCleaning up existing environment: {venv_name}")
        shutil.rmtree(venv_name)


def create_venv(venv_name):
    """Create a new virtual environment and return python and pip paths"""
    print(f"\nCreating virtual environment: {venv_name}")
    if sys.platform == "win32":
        python = "python"
        venv_path = venv_name
    else:
        python = "python3"
        venv_path = venv_name
    if not run_command(f"{python} -m venv {venv_path}"):
        print(f"Failed to create virtual environment: {venv_name}")
        return None, None
    if sys.platform == "win32":
        python_path = os.path.join(venv_path, "Scripts", "python")
        pip_path = os.path.join(venv_path, "Scripts", "pip")
    else:
        python_path = os.path.join(venv_path, "bin", "python")
        pip_path = os.path.join(venv_path, "bin", "pip")
    if not run_command(f'"{python_path}" -m pip install --upgrade pip'):
        print(f"Failed to upgrade pip in {venv_name}")
        return None, None
    return python_path, pip_path


def get_installed_packages(venv_path):
    """Get list of installed packages in the virtual environment"""
    if sys.platform == "win32":
        pip = os.path.join(venv_path, "Scripts", "pip")
    else:
        pip = os.path.join(venv_path, "bin", "pip")

    result = subprocess.run(f'"{pip}" list', shell=True, capture_output=True, text=True)
    return set(
        line.split()[0].lower()
        for line in result.stdout.split("\n")[2:]
        if line.strip()
    )


def get_expected_dependencies(extras=None):
    """Get expected dependencies from pyproject.toml"""
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)

    # Get base dependencies
    base_deps = set(
        dep.split(">=")[0].split("==")[0].strip().lower()
        for dep in pyproject["project"]["dependencies"]
    )

    if not extras:
        return base_deps

    # Get optional dependencies
    optional_deps = set()
    for extra in extras:
        if extra in pyproject["project"]["optional-dependencies"]:
            deps = pyproject["project"]["optional-dependencies"][extra]
            optional_deps.update(
                dep.split(">=")[0].split("==")[0].strip().lower() for dep in deps
            )

    return base_deps.union(optional_deps)


def verify_dependencies(venv_path, expected_deps):
    """Verify that only expected dependencies are installed"""
    installed_packages = get_installed_packages(venv_path)

    # Get all dependencies including transitive ones
    if sys.platform == "win32":
        pip = os.path.join(venv_path, "Scripts", "pip")
    else:
        pip = os.path.join(venv_path, "bin", "pip")

    # Get all dependencies including transitive ones
    result = subprocess.run(
        f'"{pip}" show mhkit', shell=True, capture_output=True, text=True
    )
    requires = set()
    for line in result.stdout.split("\n"):
        if line.startswith("Requires:"):
            requires.update(pkg.strip().lower() for pkg in line[9:].split(","))

    # Check for unexpected packages
    unexpected = installed_packages - expected_deps - requires
    if unexpected:
        print(f"\nWarning: Unexpected packages found in {venv_path}:")
        for pkg in sorted(unexpected):
            print(f"  - {pkg}")
        return False
    return True


def run_pytest_on_extra(extra, test_dir):
    """Create venv, install MHKiT with a specific extra, install pytest, and run pytest on a test directory."""
    venv_name = f"mhkit_{extra}_test"
    cleanup_venv(venv_name)
    python, pip = create_venv(venv_name)
    if not python:
        print(f"Failed to create venv for {extra}")
        return
    if not run_command(f'"{pip}" install -e ".[{extra}]"'):
        print(f"Failed to install MHKiT with .[{extra}]")
        return
    if not run_command(f'"{pip}" install pytest'):
        print(f"Failed to install pytest in {venv_name}")
        return
    print(f"\nRunning pytest on {test_dir} in {venv_name}...")
    run_command(f'"{python}" -m pytest {test_dir}')


if __name__ == "__main__":
    module_extras = [
        "wave",
        "tidal",
        "river",
        "dolfyn",
        "power",
        "loads",
        "mooring",
        "acoustics",
        "qc",
        "utils",
    ]
    for module in module_extras:
        test_dir = f"mhkit/tests/{module}/"
        print(f"\n=== Testing {module} extra with its own tests ===")
        run_pytest_on_extra(module, test_dir)
