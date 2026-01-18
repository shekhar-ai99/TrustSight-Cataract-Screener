import json
import os
from datetime import datetime

REPORT_FILE = "run_reports/report.json"

def log_run(outcome):
    """Log the outcome of a run to the report JSON."""
    timestamp = datetime.now().isoformat()
    entry = {
        "timestamp": timestamp,
        "outcome": outcome
    }
    
    # Load existing reports
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, 'r') as f:
            reports = json.load(f)
    else:
        reports = []
    
    # Append new entry
    reports.append(entry)
    
    # Save back
    with open(REPORT_FILE, 'w') as f:
        json.dump(reports, f, indent=2)
    
    print(f"Logged run outcome at {timestamp}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        outcome = " ".join(sys.argv[1:])
        log_run(outcome)
    else:
        print("Usage: python utils/log_run.py <outcome>")