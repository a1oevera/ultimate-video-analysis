"""
Cheap feasibility test for B3: how well does an off-the-shelf, COCO-pretrained
YOLO (generic "person" class, zero custom training) do on real footage? Same
spirit as B1's feasibility test -- measure before committing to the seed
dataset download + fine-tuning pipeline.

NEXT_STEPS.md flags far-side-player recall as the specific risk (partial-field
framing makes small, distant players common) -- this script reports detection
count/confidence and saves an annotated image so that's checkable by eye, not
just trusted from an aggregate number.

Run:  python run_player_detection_test.py <video_path> <t_sec> [out.jpg] [conf] [imgsz]

MEASURED: default imgsz=640 (YOLO downscales the full 1920x1080 frame,
shrinking already-small far-side players below detectability) missed the
ENTIRE field of play -- 20 detections, all near-camera sideline/bench people,
zero on-field players. imgsz=1920 (native resolution, no downscale) recovers
far more far-field detections at the same confidence. Default here is 1920,
not YOLO's own default -- don't silently regress to the failing case.

Needs opencv-python + ultralytics (Track B deps, see requirements.txt).
"""
import sys
import cv2
from ultralytics import YOLO

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/ojuc.mp4"
t_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 320.0
out_path = sys.argv[3] if len(sys.argv) > 3 else "outputs/detection_sample.jpg"
conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
imgsz = int(sys.argv[5]) if len(sys.argv) > 5 else 1920

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit(f"could not read frame at t={t_sec}s from {video_path}")

model = YOLO("yolo11n.pt")  # COCO-pretrained, zero custom training -- class 0 = person
results = model.predict(frame, classes=[0], conf=conf, imgsz=imgsz, verbose=False)
r = results[0]
boxes = r.boxes

print(f"Frame: t={t_sec}s, {frame.shape[1]}x{frame.shape[0]}, inference imgsz={imgsz}")
print(f"Detections (person, conf>={conf}): {len(boxes)}")

if len(boxes):
    confs = boxes.conf.cpu().numpy()
    areas = ((boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])).cpu().numpy()
    frame_area = frame.shape[0] * frame.shape[1]
    print(f"  confidence: mean={confs.mean():.2f} min={confs.min():.2f} max={confs.max():.2f}")
    print(f"  box area as % of frame: mean={100*areas.mean()/frame_area:.3f}%  "
          f"min={100*areas.min()/frame_area:.3f}%  max={100*areas.max()/frame_area:.3f}%")
    small = (areas / frame_area < 0.001).sum()  # rough "far/small player" proxy
    print(f"  small detections (<0.1% of frame area, likely far-side players): {small}/{len(boxes)}")

annotated = r.plot()
cv2.imwrite(out_path, annotated)
print(f"\nSaved annotated frame to {out_path}")
