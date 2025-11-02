"""phi-scan-ok -- this file intentionally contains identifier-shaped
strings (mrn, dob, ssn) to test that the scanner flags them; none are
real values.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import phi_scan  # noqa: E402


def test_filename_pattern_flags_mrn(tmp_path):
    bad = tmp_path / "patient_mrn_list.csv"
    bad.write_text("a,b\n1,2\n")
    hits = phi_scan.scan_filename(bad.relative_to(tmp_path))
    assert hits


def test_filename_pattern_flags_vernov(tmp_path):
    bad = tmp_path / "Post_Clean_VernovData.csv"
    bad.write_text("a,b\n1,2\n")
    hits = phi_scan.scan_filename(bad.relative_to(tmp_path))
    assert hits


def test_clean_filename_is_not_flagged(tmp_path):
    good = tmp_path / "referral_followup_synthetic.csv"
    good.write_text("subject_id,GA\nSYN0001,28.0\n")
    hits = phi_scan.scan_filename(good.relative_to(tmp_path))
    assert not hits


def test_header_pattern_flags_mrn_column():
    hits = phi_scan.scan_header_line(Path("x.csv"), "mrn,GA,outcome", 1)
    assert hits


def test_header_pattern_allows_coded_columns():
    hits = phi_scan.scan_header_line(
        Path("x.csv"), "subject_id,GA,coi,Insurance_type,Referred_vs_NotReferred", 1
    )
    assert not hits


def test_inline_marker_exempts_line():
    hits = phi_scan.scan_header_line(Path("x.py"), "mrn  # phi-scan-ok", 1)
    assert not hits


def test_repo_itself_is_clean():
    root = Path(__file__).resolve().parents[1]
    findings = phi_scan.scan_tree(root)
    # test_phi_scan.py's own docstring mentions mrn/dob/ssn by name but
    # not as a filename or a header column, so it should not trip the
    # scanner.
    assert findings == []
