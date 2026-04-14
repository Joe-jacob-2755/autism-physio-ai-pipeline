"""Run Module 4 EDA directly on existing run_038 training data."""
import sys, json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "module_4_eda"))

from analyser import CombinedEDA

# Paths
train_dir = ROOT / "outputs" / "run_038" / "module_3_splitting" / "M3_v1.0.0_run_001" / "train"
manifest_path = ROOT / "outputs" / "run_038" / "run_manifest.json"
output_dir = ROOT / "outputs" / "run_038" / "module_4_eda_v2"

# Load demographics from manifest
demographics = {}
with open(manifest_path) as f:
    manifest = json.load(f)
for udata in manifest.get("users", []):
    uid = udata.get("user_id")
    if uid:
        demographics[uid] = {k: v for k, v in udata.items() if k != "user_id"}

# Discover training user folders
user_folders = {}
for d in sorted(train_dir.iterdir()):
    if d.is_dir() and d.name.startswith("user_"):
        user_folders[d.name] = d

print(f"Found {len(user_folders)} training users: {list(user_folders.keys())}")
print(f"Demographics for {len(demographics)} users")

# Run EDA
eda = CombinedEDA(verbose=True)
results = eda.run(
    source={"folders": user_folders},
    demographics=demographics,
    output_dir=str(output_dir),
)

print(f"\nReport: {results.report_path}")
print(f"Output: {results.output_dir}")
print("Done!")
