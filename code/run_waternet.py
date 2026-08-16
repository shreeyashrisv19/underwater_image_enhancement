import subprocess
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

source = os.path.join(ROOT_DIR, "waternet_inputs")
weights = os.path.join(ROOT_DIR, "models", "waternet_exported_state_dict-daa0ee.pt")

cmd = [
    "python",
    "inference.py",
    "--source",
    source,
    "--weights",
    weights,
    "--name",
    "batch_run"
]

subprocess.run(cmd)
