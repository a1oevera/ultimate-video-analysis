"""
Does the locally fine-tuned detector (run_finetune_detector.py, trained on
Roboflow's "Tracking" dataset -- a DIFFERENT collection of ultimate footage,
not the person's own) actually transfer to the person's real broadcast
video? Domain shift (different camera, compression, jersey colors, lighting)
is a real risk the fine-tune's own training-set metrics (mAP50=0.957 on
Roboflow's val split) can't answer -- this compares generic COCO yolo11n
against the fine-tuned model on the SAME real frame, side by side.

Run:  python run_finetune_compare_test.py <video_path> <t_sec> [finetuned_pt] [out_prefix]
Needs opencv-python + ultralytics.
"""
import sys
import cv2
from ultralytics import YOLO

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/videoplayback.mp4"
t_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 1170.0
finetuned_pt = sys.argv[3] if len(sys.argv) > 3 else "runs/detect/frisbee_finetune/weights/best.pt"
out_prefix = sys.argv[4] if len(sys.argv) > 4 else "outputs/finetune_compare"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit(f"could not read frame at t={t_sec}s from {video_path}")

print(f"Frame: t={t_sec}s, {frame.shape[1]}x{frame.shape[0]}")

# --- generic COCO yolo11n, same settings used throughout this project ---
generic = YOLO("yolo11n.pt")
r_generic = generic.predict(frame, classes=[0], conf=0.25, imgsz=1920, verbose=False)[0]
print(f"\nGeneric COCO yolo11n (person class): {len(r_generic.boxes)} detections")
if len(r_generic.boxes):
    confs = r_generic.boxes.conf.cpu().numpy()
    print(f"  confidence: mean={confs.mean():.2f} min={confs.min():.2f} max={confs.max():.2f}")
cv2.imwrite(f"{out_prefix}_generic.jpg", r_generic.plot())

# --- fine-tuned model: classes are frisbee/observer/player, not COCO's 80 ---
tuned = YOLO(finetuned_pt)
print(f"\nFine-tuned model classes: {tuned.names}")
r_tuned = tuned.predict(frame, conf=0.25, imgsz=1920, verbose=False)[0]
print(f"Fine-tuned (all classes): {len(r_tuned.boxes)} detections")
if len(r_tuned.boxes):
    cls_ids = r_tuned.boxes.cls.cpu().numpy().astype(int)
    confs = r_tuned.boxes.conf.cpu().numpy()
    for cid in sorted(set(cls_ids)):
        name = tuned.names[cid]
        n = (cls_ids == cid).sum()
        c = confs[cls_ids == cid]
        print(f"  {name}: {n} detections, confidence mean={c.mean():.2f} min={c.min():.2f} max={c.max():.2f}")
cv2.imwrite(f"{out_prefix}_finetuned.jpg", r_tuned.plot())

print(f"\nSaved {out_prefix}_generic.jpg and {out_prefix}_finetuned.jpg for a visual side-by-side.")
