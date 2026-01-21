"""Repackage existing test submissions as weights-only tarballs.

For each folder under test_submission/, load the old model.pth (full pickle),
extract state_dict, and build a new archive under tst_submissions/ containing:
  - model.pth (state_dict)
  - code/ (copied from original submission folder)
  - requirements.txt (from project root)
  - README.md (from project root)
"""
import os
import sys
import tarfile
import shutil
from pathlib import Path

import types
import torch

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import model definition so torch.load can resolve it
from src.model import CataractModel  # noqa: E402

# Register a fake 'model' module for legacy pickles
fake_model_module = types.ModuleType("model")
fake_model_module.CataractModel = CataractModel
sys.modules["model"] = fake_model_module

SUBMISSIONS_DIR = ROOT / "test_submission"
OUTPUT_ROOT = ROOT / "tst_submissions"
OUTPUT_ROOT.mkdir(exist_ok=True)

readme_src = ROOT / "README.md"
req_src = ROOT / "requirements.txt"


def repackage_submission(submission_path: Path) -> Path:
    name = submission_path.name
    print(f"Processing {name}...")
    old_model_path = submission_path / "model.pth"
    if not old_model_path.exists():
        raise FileNotFoundError(f"model.pth not found in {submission_path}")

    # Load full model object (trusted) and extract weights
    model = torch.load(old_model_path, map_location="cpu")
    state = model.state_dict()

    # Prepare output folder
    out_folder = OUTPUT_ROOT / f"{name}_safe"
    out_folder.mkdir(parents=True, exist_ok=True)

    # Save safe weights
    safe_model_path = out_folder / "model.pth"
    torch.save(state, safe_model_path)

    # Copy code folder
    shutil.copytree(submission_path / "code", out_folder / "code", dirs_exist_ok=True)

    # Copy README and requirements
    if readme_src.exists():
        shutil.copy(readme_src, out_folder / "README.md")
    if req_src.exists():
        shutil.copy(req_src, out_folder / "requirements.txt")

    # Create tar.gz
    tar_path = out_folder / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(safe_model_path, arcname="model.pth")
        tar.add(out_folder / "requirements.txt", arcname="requirements.txt")
        tar.add(out_folder / "README.md", arcname="README.md")
        tar.add(out_folder / "code", arcname="code")

    print(f"  ✓ Repkg complete → {tar_path}")
    return tar_path


def main():
    submissions = sorted(p for p in SUBMISSIONS_DIR.iterdir() if p.is_dir())
    print(f"Found {len(submissions)} submissions")
    tar_outputs = []
    for sub in submissions:
        tar_outputs.append(repackage_submission(sub))

    print("\nSummary:")
    for t in tar_outputs:
        print(f"  - {t}")


if __name__ == "__main__":
    main()
