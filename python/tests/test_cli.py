import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from names import MINILM, RERANKER, UNREGISTERED

import kodamaembed
from kodamaembed import cli


def run(capsys, *args):
    code = cli.main(list(args))
    out, err = capsys.readouterr()
    return code, out, err


def test_models_lists_every_model_with_its_kind_width_length_and_download(capsys):
    code, out, _ = run(capsys, "models")
    assert code == 0
    header, *lines = out.splitlines()
    assert header.split() == ["NAME", "KIND", "DIM", "TOKENS", "DOWNLOAD"]
    rows = {line.split()[0]: line.split()[1:] for line in lines}
    assert rows[MINILM] == ["embedding", "384", "256", "91.3", "MB"]
    assert rows["intfloat/e5-large"] == ["embedding", "1024", "512", "1.3", "GB"]
    assert rows[RERANKER] == ["reranker", "-", "512", "91.6", "MB"]


def test_embed_prints_one_json_line_per_text(embedder, capsys):
    # Python before 3.12 drops texts after an option unless parsing intermixes.
    code, out, _ = run(capsys, "embed", MINILM, "--device", "cpu", "first text", "second text")
    assert code == 0
    rows = [json.loads(line) for line in out.splitlines()]
    assert [row["index"] for row in rows] == [0, 1]
    vectors = np.array([row["embedding"] for row in rows], np.float32)
    np.testing.assert_array_equal(vectors, embedder.embed_passages(["first text", "second text"]))


def test_embed_reads_stdin_lines_and_saves_npy(embedder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("alpha\nbeta\n"))
    target = tmp_path / "vectors.npy"
    code, out, _ = run(capsys, "embed", MINILM, "--device", "cpu", "--query", "--output", str(target))
    assert code == 0
    assert out == ""
    np.testing.assert_array_equal(np.load(target), embedder.embed_queries(["alpha", "beta"]))


def test_failures_exit_with_one_and_say_why(capsys):
    code, _, err = run(capsys, "embed", UNREGISTERED, "text")
    assert code == 1
    assert "all-MiniLM-L6-v2" in err
    code, _, err = run(capsys, "embed", MINILM, "--device", "tpu", "text")
    assert code == 1
    assert "device" in err


@pytest.mark.parametrize("args", [
    ["translate"], ["embed"], ["models", "--extra"],
    ["embed", MINILM, "--threads", "-1"], ["serve", MINILM, "--port", "65536"],
])
def test_usage_errors_exit_with_two(capsys, args):
    with pytest.raises(SystemExit) as exit:
        cli.main(args)
    assert exit.value.code == 2
    assert "usage: kodamaembed" in capsys.readouterr().err


def test_download_prints_the_snapshot_directory(capsys):
    code, out, _ = run(capsys, "download", MINILM)
    assert code == 0
    assert (Path(out.strip()) / "model.safetensors").is_file()


def test_the_module_runs_as_a_command():
    result = subprocess.run(
        [sys.executable, "-m", "kodamaembed", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"kodamaembed {kodamaembed.__version__}"
