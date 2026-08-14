"""UTF-8 / ASCII-only policy for documentation and YAML sources."""

from __future__ import annotations

from pathlib import Path

from biosensor_priors.common.config import REPO_ROOT

_GLOBS = (
    "README.md",
    "CHANGELOG.md",
    "docs/**/*.md",
    "docs/conf.py",
    "docs/_static/**/*.svg",
    "configs/**/*.yaml",
    "data/ligands/README.md",
)


def _policy_files() -> list[Path]:
    files: list[Path] = []
    for pattern in _GLOBS:
        if "*" in pattern:
            files.extend(sorted(REPO_ROOT.glob(pattern)))
        else:
            path = REPO_ROOT / pattern
            if path.is_file():
                files.append(path)
    unique = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen or any(part == "_build" for part in path.parts):
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def test_docs_and_configs_are_utf8_ascii() -> None:
    """Docs and YAML must decode as UTF-8 with no BOM and ASCII-only source.

    Math belongs in MyST/LaTeX dollarmath, not raw Greek, Unicode minus,
    em-dashes, or smart quotes.
    """
    files = _policy_files()
    assert files, "expected documentation/config files under the repo root"
    failures: list[str] = []
    for path in files:
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            failures.append(f"{path.relative_to(REPO_ROOT)}: UTF-8 BOM is not allowed")
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            failures.append(f"{path.relative_to(REPO_ROOT)}: not UTF-8 ({exc})")
            continue
        non_ascii = sorted({ch for ch in text if ord(ch) > 127})
        if non_ascii:
            codes = ", ".join(f"U+{ord(ch):04X}" for ch in non_ascii)
            failures.append(f"{path.relative_to(REPO_ROOT)}: non-ASCII {codes}")
    assert not failures, "ASCII/UTF-8 policy failures:\n" + "\n".join(failures)
