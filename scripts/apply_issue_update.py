#!/usr/bin/env python3
"""
Append a new tournament section from a GitHub Issue body to data/TournamentResults.txt.

The Issue body should contain a fenced code block:

```text
2026 Northern
Non-qualifiers
...
Final
...
```

This script:
- extracts the first fenced block (``` or ```text)
- basic validation: first non-empty line starts with 20..
- prevents duplicates by checking if that tournament header already exists in the file
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: apply_issue_update.py <github_event.json> <results_file> <output_results_file>")
        sys.exit(2)

    event_path = Path(sys.argv[1])
    results_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    event = json.loads(event_path.read_text(encoding="utf-8"))
    body = (event.get("issue") or {}).get("body") or ""
    m = re.search(r"```(?:text)?\s*\n(.*?)\n```", body, flags=re.S | re.I)
    if not m:
        print("No fenced code block found in issue body.")
        sys.exit(1)

    block = m.group(1).strip("\n")
    lines = [l.rstrip() for l in block.splitlines()]
    lines = [l for l in lines if l.strip() != ""]
    if not lines or not re.match(r"^20..\s+\w+", lines[0]):
        print("Block must start with a tournament header like '2026 Northern'.")
        sys.exit(1)

    header = lines[0].strip()
    existing = results_path.read_text(encoding="utf-8", errors="ignore")
    if re.search(rf"^{re.escape(header)}\s*$", existing, flags=re.M):
        print(f"Tournament '{header}' already exists in results file; refusing to duplicate.")
        sys.exit(1)

    # Append with a preceding newline to ensure clean separation.
    new_text = existing.rstrip() + "\n\n" + "\n".join(lines) + "\n"
    out_path.write_text(new_text, encoding="utf-8")
    print(f"Appended {header} to {out_path}")


if __name__ == "__main__":
    main()
