# data-raw/

This folder is intentionally empty in version control.

Raw Epic/Clarity exports and REDCap pulls (e.g. the department's cleaned
discharge-referral extract) are never committed here or anywhere else in
this repository. Place a local extract in this folder only on your own
machine; `.gitignore` blocks it from being tracked, and `scripts/phi_scan.py`
will refuse to let a commit proceed if an identifier-shaped file ends up
staged anyway.

Everything checked into `data/synthetic/` is fabricated and carries no
relationship to any real patient record.
