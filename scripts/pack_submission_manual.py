"""Package a weights-only submission tarball from best_model/model.pth.
Output: test_submission/model_F1_0.6800/model.tar.gz
"""
import tarfile
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_WEIGHTS = ROOT / 'best_model' / 'model.pth'
OUT_DIR = ROOT / 'test_submission' / 'model_F1_0.6800'

if not SRC_WEIGHTS.exists():
    sys.exit('best_model/model.pth not found')

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Copy files
shutil.copy(SRC_WEIGHTS, OUT_DIR / 'model.pth')
for fname in ['requirements.txt', 'README.md']:
    shutil.copy(ROOT / fname, OUT_DIR / fname)
shutil.copytree(ROOT / 'code', OUT_DIR / 'code', dirs_exist_ok=True)

# Create tar.gz
TAR_PATH = OUT_DIR / 'model.tar.gz'
with tarfile.open(TAR_PATH, 'w:gz') as tar:
    tar.add(OUT_DIR / 'model.pth', arcname='model.pth')
    tar.add(OUT_DIR / 'requirements.txt', arcname='requirements.txt')
    tar.add(OUT_DIR / 'README.md', arcname='README.md')
    tar.add(OUT_DIR / 'code', arcname='code')

print(f'Created submission: {TAR_PATH}')
