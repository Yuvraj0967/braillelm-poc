"""Thin wrapper around the liblouis command-line translator.

liblouis is the reference open-source braille translator. It is used only to
build training and test data and as the rule-based baseline; the released model
does not need it at run time.

    apt-get install liblouis-bin liblouis-data        # provides lou_translate

Set LOU_TRANSLATE to the binary and LOUIS_TABLEPATH to the tables directory if
they are not on the default paths. On Windows, unzip the official
liblouis-<version>-win64.zip release into a folder named liblouis-<version>
in your user folder (or tools/liblouis/ in this repository) and it is found
automatically.
"""
import os
import shutil
import subprocess
from pathlib import Path

from braillelm.paths import REPO

G1 = "unicode.dis,en-ueb-g1.ctb"
G2 = "unicode.dis,en-ueb-g2.ctb"
TABLES = {"g1": G1, "g2": G2}


def _local():
    """An unzipped Windows release: tools/liblouis in the repository, or ~/liblouis*."""
    for base in [REPO / "tools" / "liblouis", *sorted(Path.home().glob("liblouis*"), reverse=True)]:
        if (base / "bin" / "lou_translate.exe").exists():
            return base
    return None


def binary():
    found = os.environ.get("LOU_TRANSLATE") or shutil.which("lou_translate")
    if not found and _local():
        found = str(_local() / "bin" / "lou_translate.exe")
    return found


def _env():
    """The local Windows copy does not know where its tables are; tell it."""
    env = dict(os.environ)
    base = _local()
    if "LOUIS_TABLEPATH" not in env and base and (base / "share" / "liblouis" / "tables").exists():
        env["LOUIS_TABLEPATH"] = str(base / "share" / "liblouis" / "tables")
    return env


def available():
    return bool(binary())


def _run(lines, table, direction):
    exe = binary()
    if not exe:
        raise RuntimeError("lou_translate not found; install liblouis-bin or set LOU_TRANSLATE")
    for line in lines:
        if "\n" in line or "\r" in line:
            raise ValueError("lines must not contain line breaks")
    out = []
    for s in range(0, len(lines), 20000):
        part = lines[s:s + 20000]
        res = subprocess.run([exe, direction, table], input=("\n".join(part) + "\n").encode("utf-8"),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, env=_env())
        got = res.stdout.decode("utf-8").replace("\r\n", "\n").split("\n")
        if got and got[-1] == "":
            got.pop()
        if len(got) != len(part):
            raise RuntimeError(f"liblouis returned {len(got)} lines for {len(part)} inputs")
        out.extend(got)
    return out


def to_braille(lines, grade="g2"):
    """Print text -> Unicode braille, one output line per input line."""
    return [b.replace(" ", "⠀") for b in _run(list(lines), TABLES[grade], "--forward")]


def to_print(lines, grade="g2"):
    """Unicode braille -> print text using liblouis rules (the baseline)."""
    return _run([b.replace("⠀", " ") for b in lines], TABLES[grade], "--backward")


def version():
    exe = binary()
    if not exe:
        return None
    return subprocess.run([exe, "--version"], stdout=subprocess.PIPE).stdout.decode().splitlines()[0]
