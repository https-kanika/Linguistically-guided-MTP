"""Resume discovery: an earlier week-1 run is found however it was attached on Kaggle."""
import zipfile
from pathlib import Path

from lgmtp.week1 import _resume_copy, locate_run

FILES = {"data/sources.json": "{}", "data/unit_vocab.json": "[]", "data/qwen/dev.jsonl": "x\n",
         "results/log.txt": "old log\n"}


def _write(base: Path, prefix: str = "") -> None:
    for rel, txt in FILES.items():
        p = base / prefix / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")


def _zip(path: Path, prefix: str = "", sep: str = "/") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for rel, txt in FILES.items():
            z.writestr((prefix + rel).replace("/", sep), txt)


def _check(run: Path) -> None:
    assert run is not None
    for rel, txt in FILES.items():
        assert (run / rel).read_text(encoding="utf-8") == txt


def test_notebook_output_folder(tmp_path):
    _write(tmp_path / "input" / "mtp-final-boss", "week1/")
    run = locate_run([str(tmp_path / "input")], "week1")
    assert run == tmp_path / "input" / "mtp-final-boss" / "week1"
    _check(run)


def test_dataset_without_run_folder(tmp_path):
    _write(tmp_path / "input" / "old-week1-output")
    _check(locate_run([str(tmp_path / "input")], "week1"))


def test_prefers_folder_named_like_the_run(tmp_path):
    _write(tmp_path / "input" / "a", "other/")
    _write(tmp_path / "input" / "b", "week1/")
    assert locate_run([str(tmp_path / "input")], "week1") == tmp_path / "input" / "b" / "week1"


def test_zip_kept_zipped(tmp_path):
    for prefix in ("", "week1/", "kaggle/working/week1/"):
        inp = tmp_path / prefix.replace("/", "_") / "input"
        _zip(inp / "ds" / "results.zip", prefix)
        _check(locate_run([str(inp)], "week1"))


def test_zip_with_backslash_names(tmp_path):
    _zip(tmp_path / "input" / "ds" / "week1.zip", "week1/", sep="\\")
    _check(locate_run([str(tmp_path / "input")], "week1"))


def test_unrelated_inputs_are_ignored(tmp_path):
    inp = tmp_path / "input"
    (inp / "p1").mkdir(parents=True)
    with zipfile.ZipFile(inp / "p1" / "traces_v2_dev.zip", "w") as z:
        z.writestr("traces_dev.npz", "npz")
    (inp / "p1" / "not_a.zip").write_text("garbage")
    (inp / "p1" / "sources.json").write_text("{}")          # not under data/
    assert locate_run([str(inp), None, str(tmp_path / "missing")], "week1") is None


def test_explicit_zip_path(tmp_path):
    _zip(tmp_path / "w.zip", "week1/")
    _check(locate_run([str(tmp_path / "w.zip")], "week1"))


def test_resume_copy_never_overwrites(tmp_path):
    _write(tmp_path / "old")
    root = tmp_path / "new" / "week1"
    (root / "results").mkdir(parents=True)
    (root / "results" / "log.txt").write_text("new log\n", encoding="utf-8")
    msgs = []
    _resume_copy(tmp_path / "old", root, msgs.append)
    assert (root / "results" / "log.txt").read_text(encoding="utf-8") == "new log\n"
    assert (root / "data" / "qwen" / "dev.jsonl").read_text(encoding="utf-8") == "x\n"
    assert "copied 3 files (1 already present)" in msgs[0] and "['qwen']" in msgs[0]
