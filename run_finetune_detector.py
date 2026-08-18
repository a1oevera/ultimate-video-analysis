"""
Fine-tune yolo11n on the downloaded Roboflow "Tracking" dataset (423 images,
classes: frisbee/observer/player -- see run_download_roboflow_dataset.py)
instead of using it zero-shot (generic COCO "person" class, no frisbee/
observer distinction). Runs entirely locally -- CPU-only on this machine
(no ROCm on Intel Mac/AMD GPU, see requirements.txt), starting from the same
yolo11n.pt checkpoint already used elsewhere in this project.

Run:  python run_finetune_detector.py [data.yaml] [epochs] [imgsz]
Needs ultralytics (Track B deps) + the dataset already downloaded via
run_download_roboflow_dataset.py.
"""
import sys
from ultralytics import YOLO

data_yaml = sys.argv[1] if len(sys.argv) > 1 else "roboflow_datasets/tracking-w8biu-26/data.yaml"
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640

model = YOLO("yolo11n.pt")
model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device="cpu",
            project="runs/detect", name="frisbee_finetune", exist_ok=True)
