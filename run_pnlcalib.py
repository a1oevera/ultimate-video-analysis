"""Run PnLCalib on a frame and return an image->field homography in metres.

PnLCalib is trained on broadcast soccer and regresses pitch keypoints, so unlike
our line fits it brings far-field landmarks. It outputs camera parameters; on the
ground plane (z=0) those reduce to a homography: take columns 0,1,3 of the 3x4
projection, in pitch-centred metres (x along the 105m length, y across the 68m).

Verify the result the same way as any other calibration -- measure the PAINTED
centre circle through it and check 9.15m. The network never sees that constraint
directly, and it failed visibly on a tight view (t=3235) while succeeding on a
wide one (t=1700), so per-frame checking is not optional.

Run:  python3 run_pnlcalib.py <frame.jpg> [out.npy]
"""
import sys
import os
import numpy as np
import cv2
import torch
import yaml
import torchvision.transforms as T
import torchvision.transforms.functional as f
from PIL import Image

PNL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "PnLCalib")
sys.path.insert(0, PNL)
from model.cls_hrnet import get_cls_net                      # noqa: E402
from model.cls_hrnet_l import get_cls_net as get_cls_net_l   # noqa: E402
from utils.utils_calib import FramebyFrameCalib              # noqa: E402
from utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool,      # noqa: E402
                                 get_keypoints_from_heatmap_batch_maxpool_l,
                                 complete_keypoints, coords_to_dict)

DEVICE = "cpu"


def load_models():
    cfg = yaml.safe_load(open(os.path.join(PNL, "config/hrnetv2_w48.yaml"), "r"))
    cfg_l = yaml.safe_load(open(os.path.join(PNL, "config/hrnetv2_w48_l.yaml"), "r"))
    m = get_cls_net(cfg)
    m.load_state_dict(torch.load(os.path.join(PNL, "SV_kp"), map_location=DEVICE))
    m.to(DEVICE).eval()
    ml = get_cls_net_l(cfg_l)
    ml.load_state_dict(torch.load(os.path.join(PNL, "SV_lines"), map_location=DEVICE))
    ml.to(DEVICE).eval()
    return m, ml


def homography_for(frame_bgr, m, ml, kp_th=0.3434, line_th=0.7867):
    h0, w0 = frame_bgr.shape[:2]
    cam = FramebyFrameCalib(iwidth=w0, iheight=h0, denormalize=True)
    t = f.to_tensor(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))).float().unsqueeze(0)
    if t.size()[-1] != 960:
        t = T.Resize((540, 960))(t)
    t = t.to(DEVICE)
    with torch.no_grad():
        hm, hml = m(t), ml(t)
    kp = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1, :, :]), threshold=kp_th)
    ln = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool_l(hml[:, :-1, :, :]), threshold=line_th)
    kp, ln = complete_keypoints(kp[0], ln[0], w=t.size(3), h=t.size(2), normalize=True)
    cam.update(kp, ln)
    p = cam.heuristic_voting(refine_lines=True)
    if p is None:
        return None
    P = projection(p)
    Hf2i = P[:, [0, 1, 3]]              # ground plane z=0, pitch-centred metres
    return np.linalg.inv(Hf2i)          # image -> field


def projection(params):
    # Use the rotation_matrix the solver already returned. Rebuilding it from
    # pan/tilt/roll gives a different orientation and puts the pitch centre
    # thousands of metres away -- inference.py itself reads rotation_matrix.
    c = params["cam_params"]
    pos = np.array(c["position_meters"])
    rot = np.array(c["rotation_matrix"])
    It = np.eye(4)[:-1]
    It[:, -1] = -pos
    Q = np.array([[c["x_focal_length"], 0, c["principal_point"][0]],
                  [0, c["y_focal_length"], c["principal_point"][1]],
                  [0, 0, 1]])
    return Q @ (rot @ It)


if __name__ == "__main__":
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    frame = cv2.imread(path)
    m, ml = load_models()
    H = homography_for(frame, m, ml)
    if H is None:
        sys.exit("PnLCalib returned no calibration for this frame")
    print(f"{os.path.basename(path)}: homography obtained")
    if out:
        np.save(out, H)
        print(f"  saved {out}")
