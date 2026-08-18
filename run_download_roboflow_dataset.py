"""
Download a labeled dataset from Roboflow Universe for local fine-tuning.

MEASURED (see results/detection_finding.md's addendum, or the conversation
that led here): none of the three Roboflow Universe ultimate-frisbee
projects in NEXT_STEPS.md B3 offer downloadable pretrained WEIGHTS via the
free API -- "Ultimate Player" has zero trained versions at all (despite
NEXT_STEPS.md saying "pretrained model available" -- wrong, corrected), and
even the two that DO have trained versions (this one, and "Frisbee
Tracking") only expose that training via Roboflow's hosted cloud inference
API (sends frames to their servers -- the same redistribution concern that
keeps videos/ out of this repo) or a DATASET download for training your own
model. This script does the dataset download half; run_finetune_detector.py
does the actual local training, so nothing of this project's footage is ever
uploaded anywhere.

Needs ROBOFLOW_API_KEY in .env (gitignored) -- private key, not the
publishable one (see the conversation: the SDK's download flow authenticates
your account even for public datasets, it doesn't expose anything of yours).

Run:  python run_download_roboflow_dataset.py [workspace] [project] [version] [format] [location]
Needs `pip install roboflow python-dotenv`.
"""
import sys
import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

workspace = sys.argv[1] if len(sys.argv) > 1 else "frisbee-tracker"
project_name = sys.argv[2] if len(sys.argv) > 2 else "tracking-w8biu"
version_n = int(sys.argv[3]) if len(sys.argv) > 3 else 26
model_format = sys.argv[4] if len(sys.argv) > 4 else "yolov11"
location = sys.argv[5] if len(sys.argv) > 5 else f"roboflow_datasets/{project_name}-{version_n}"

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    sys.exit("ROBOFLOW_API_KEY not set -- put it in .env (gitignored), see the module docstring")

rf = Roboflow(api_key=api_key)
proj = rf.workspace(workspace).project(project_name)
version = proj.version(version_n)
dataset = version.download(model_format, location=location, overwrite=True)
print(f"\nDownloaded to: {dataset.location}")
print(f"data.yaml: {os.path.join(dataset.location, 'data.yaml')}")
