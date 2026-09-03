import nbformat
import pytest
from conftest import PROJECT_ROOT


@pytest.mark.parametrize(
    "name",
    [
        "01_inspect_experiment.ipynb",
        "02_run_quick_experiment.ipynb",
        "03_analyze_results.ipynb",
    ],
)
def test_notebook_is_valid_v4(name: str) -> None:
    path = PROJECT_ROOT / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert notebook.cells
