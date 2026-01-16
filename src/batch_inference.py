import os
import sys
import csv
import json
from .infer import infer


def batch_run(input_dir: str, out_csv: str = "batch_results.csv"):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    rows = []
    for fn in sorted(files):
        path = os.path.join(input_dir, fn)
        try:
            out = infer(path, explain=False)
            data = json.loads(out)
            rows.append({
                'filename': fn,
                'prediction': data.get('cataract_prob'),
                'confidence': data.get('confidence'),
                'rejected': data.get('status') != 'PREDICT'
            })
        except Exception as e:
            rows.append({
                'filename': fn,
                'prediction': None,
                'confidence': None,
                'uncertainty': None,
                'rejected': True
            })

    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'prediction', 'confidence', 'uncertainty', 'rejected'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python -m src.batch_inference <image_dir>')
        sys.exit(1)
    batch_run(sys.argv[1])
