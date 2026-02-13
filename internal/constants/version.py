import tomllib
from pathlib import Path


def _get_version() -> str:
    pyproject_file = Path(__file__).parent.parent.parent / "pyproject.toml"

    if pyproject_file.exists():
        with open(pyproject_file, "rb") as f:
            data = tomllib.load(f)
            return data["project"]["version"]
    return ""
