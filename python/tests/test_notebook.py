import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "quickstart.ipynb"


def code_cells():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return [cell for cell in cells if cell["cell_type"] == "code"]


def test_notebook_runs():
    # Magic and shell lines install the package in a notebook; this environment already has it.
    lines = (line for cell in code_cells() for line in "".join(cell["source"]).splitlines())
    code = "\n".join(line for line in lines if not line.lstrip().startswith(("%", "!")))
    exec(compile(code, str(NOTEBOOK), "exec"), {"__name__": "__main__"})


def test_notebook_outputs_are_cleared():
    for cell in code_cells():
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
