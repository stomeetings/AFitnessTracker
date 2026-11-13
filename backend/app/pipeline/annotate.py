"""Draw bbox + skeleton (lines and joint dots only) onto a BGR frame in place."""
import cv2

from app.engines import SKELETON, PoseResult


def draw(frame, res: PoseResult, min_conf: float = 0.5) -> None:
    b = res.bbox
    cv2.rectangle(frame, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), (0, 200, 255), 2)
    for a, c in SKELETON:
        ka, kc = res.get(a, min_conf), res.get(c, min_conf)
        if ka and kc:
            cv2.line(frame, (int(ka.x), int(ka.y)), (int(kc.x), int(kc.y)), (0, 255, 0), 2)
    for k in res.keypoints.values():
        if k.conf >= min_conf:
            cv2.circle(frame, (int(k.x), int(k.y)), 4, (255, 80, 0), -1)
