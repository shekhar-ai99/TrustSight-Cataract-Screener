#!/usr/bin/env python3
import sys
import json
from src.inference import infer

# Usage: smoke_test.py clean.jpg bad.jpg
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: smoke_test.py <clean_image> <bad_image>")
        sys.exit(2)
    clean, bad = sys.argv[1], sys.argv[2]

    print("Running smoke test on clean image:")
    out_clean = infer(clean, explain=False)
    print(out_clean)
    res_clean = json.loads(out_clean)
    if res_clean.get("status") != "PREDICT":
        print("SMOKE FAILED: clean image did not PREDICT")
        sys.exit(1)

    print("Running smoke test on bad image (should be REJECT):")
    out_bad = infer(bad, explain=False)
    print(out_bad)
    res_bad = json.loads(out_bad)
    if res_bad.get("status") != "REJECT":
        print("SMOKE FAILED: bad image did not REJECT")
        sys.exit(1)

    print("SMOKE PASSED")
    sys.exit(0)
