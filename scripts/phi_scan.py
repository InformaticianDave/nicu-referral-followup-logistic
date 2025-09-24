"""Lightweight static PHI/identifier scanner.

Run before committing: flags filenames and column headers that look
like one of the HIPAA Safe Harbor identifiers (names, MRNs, exact
dates of service, geographic subdivisions smaller than a state, etc.)
so a raw extract can't quietly slip into version control.

This is a guardrail, not a substitute for judgment -- it scans
filenames and header rows of data/code files, not every line of
prose, and an inline `# phi-scan-ok` marker can exempt a specific
line that a reviewer has confirmed is safe (e.g. a variable literally
named "mrn" in a docstring explaining why MRNs are never used).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_CODE_EXTENSIONS = {".csv", ".tsv", ".yml", ".yaml", ".R", ".py", ".sql", ".json"}

FILENAME_PATTERNS = [
    re.compile(r"(?i)mrn"),
    re.compile(r"(?i)rawdata"),
    re.compile(r"(?i)raw[_-]?extract"),
    re.compile(r"(?i)clarity[_-]?extract"),
    re.compile(r"(?i)vernov"),
]

HEADER_PATTERNS = [
    re.compile(r"(?i)^mrn$"),
    re.compile(r"(?i)^patient[_-]?name$"),
    re.compile(r"(?i)^(dob|date[_-]?of[_-]?birth)$"),
    re.compile(r"(?i)^(ssn|social[_-]?security)$"),
    re.compile(r"(?i)^address$"),
    re.compile(r"(?i)^phone(_?number)?$"),
]

INLINE_EXEMPT_MARKER = "phi-scan-ok"


def is_data_or_code_file(path: Path) -> bool:
    return path.suffix in DATA_CODE_EXTENSIONS


def scan_filename(path: Path) -> list[str]:
    hits = []
    for pattern in FILENAME_PATTERNS:
        if pattern.search(path.name):
            hits.append(f"{path}: filename matches identifier pattern /{pattern.pattern}/")
    return hits


def scan_header_line(path: Path, line: str, lineno: int) -> list[str]:
    if INLINE_EXEMPT_MARKER in line:
        return []
    hits = []
    candidates = re.split(r"[,\t]", line.strip())
    for col in candidates:
        col_clean = col.strip().strip('"\'')
        for pattern in HEADER_PATTERNS:
            if pattern.match(col_clean):
                hits.append(f"{path}:{lineno}: header column '{col_clean}' matches identifier pattern")
    return hits


def scan_tree(root: Path) -> list[str]:
    findings = []
    for path in root.rglob("*"):
        if ".git" in path.parts or "node_modules" in path.parts:
            continue
        if path.is_dir():
            continue
        findings.extend(scan_filename(path.relative_to(root)))
        if is_data_or_code_file(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline()
            except OSError:
                continue
            findings.extend(scan_header_line(path.relative_to(root), first_line, 1))
    return findings


if __name__ == "__main__":
    results = scan_tree(ROOT)
    if results:
        print("PHI scan found potential issues:")
        for item in results:
            print(f"  - {item}")
        sys.exit(1)
    print("PHI scan: clean.")
    sys.exit(0)
