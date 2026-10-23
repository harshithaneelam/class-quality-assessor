"""Zoom/branded-background-aware visual compliance features (local only, no OCR, no external APIs).

Two layouts appear in branded-background class recordings:
  * speaker_fullscreen: the active speaker's camera fills the frame
    (cam ROI = top 88 %, dropping Zoom's toolbar / bottom-left name tag);
  * screenshare: a shared window fills the canvas and, when a camera tile is
    shown, it sits in a fixed top-right PiP box (x >= 0.833 W, y < 0.175 H).
    With no tile visible the frame is screenshare_no_pip.

Pipeline: `measure_frame` (slow: decode + face detection, numbers only) for
every sampled frame, then `aggregate_visual` (fast: layout rules, thresholds,
flags) so thresholds can be tuned without re-decoding video.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_FILE = "face_detection_yunet_2023mar.onnx"

# Bundled, relative-path assets (no runtime download, no dependency on the challenge work dir):
# repo-root assets/ first, then src/assets/.
ASSET_DIRS = (Path(__file__).resolve().parent.parent / "assets", Path(__file__).resolve().parent / "assets")
ASSETS_DIR = ASSET_DIRS[0]


def find_asset(name: str) -> Path | None:
    for d in ASSET_DIRS:
        if (d / name).is_file():
            return d / name
    return None


LOGO_TEMPLATES_ASSET = find_asset("logo_templates.npz") or ASSETS_DIR / "logo_templates.npz"


def default_models_dir() -> Path:
    """Where the YuNet ONNX lives: the challenge work dir when present, /tmp/models on
    Hugging Face Spaces, else a per-user cache (never inside the repo)."""
    import os
    work = Path(os.environ.get("CHALLENGE_DATA_DIR") or r"C:\challenge-data") / "_work" / "models"
    if work.is_dir():
        return work
    if os.environ.get("SPACE_ID"):
        return Path("/tmp/models")
    return Path.home() / ".cache" / "class_assessor" / "models"


def load_logo_templates(models_dir: str | Path | None = None) -> np.ndarray | None:
    """Expanded logo templates from the bundled assets/logo_templates.npz (3 x 18 x 32
    coarse numeric grids); falls back to a locally rebuilt .npy in models_dir."""
    asset = find_asset("logo_templates.npz")
    if asset is not None:
        with np.load(asset) as z:
            return expand_logo_templates(z["templates"].astype(np.float32))
    for c in ([Path(models_dir)] if models_dir else []) + [default_models_dir()]:
        f = c / "8x8_logo_templates.npy"
        if f.is_file():
            return expand_logo_templates(np.load(f).astype(np.float32))
    return None


# ---- geometry (fractions of W/H) --------------------------------------------
PIP_X, PIP_Y = 0.833, 0.175
LEFT_STRIP_X0, LEFT_STRIP_X1 = 0.78, 0.83
BOTTOM_STRIP_Y0, BOTTOM_STRIP_Y1 = 0.18, 0.25
CENTER = (0.12, 0.85, 0.10, 0.80)  # y0, y1, x0, x1
FULL_CAM_Y1 = 0.88
PATCH_W, PATCH_H = 320, 180
PERSON_ZONE = (65, 180, 85, 235)  # y0, y1, x0, x1 in the 320x180 patch

# ---- thresholds (calibrated on the 12 labelled classes, see step 05) ---------
T = dict(
    full_face_area=1.5, hidden_black_mean=3.0, logo_corr=0.50,
    nopip_black_std=10.0, nopip_black_mean=28.0, nopip_border=1.8, nopip_color_diff=12.0,
    nopip_static_diff=1.0, nopip_static_run=3,
    bg8_dark=0.55, bg8_white=0.008, bg8_score=0.55,
    glitch_bleed=0.12,
    flag_no_8x8=15.0, flag_no_8x8_active=35.0, flag_face_on_8x8=35.0, flag_face_overall=20.0,
    flag_cam_off_pct=20.0, flag_cam_off_streak=120.0,
    flag_dark_face=60.0, flag_bleed=0.10, flag_glitch_frames=30.0,
)


# =============================================================== face detection
class FaceDetector:
    """YuNet ONNX (downloaded once) with a Haar-cascade fallback."""

    def __init__(self, models_dir: str | Path | None = None):
        import cv2
        self.cv2 = cv2
        self.yunet = None
        self.kind = "haar"
        bundled = find_asset(YUNET_FILE)
        if bundled is not None or models_dir is not None:
            path = bundled or Path(models_dir) / YUNET_FILE
            try:
                if not path.is_file():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".part")
                    urllib.request.urlretrieve(YUNET_URL, tmp)
                    tmp.replace(path)
                self.yunet = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.6, 0.3, 50)
                self.kind = "yunet"
            except Exception:
                self.yunet = None
        if self.yunet is None:
            self.haar = [cv2.CascadeClassifier(cv2.data.haarcascades + f)
                         for f in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")]

    def detect(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Face boxes (x, y, w, h) in bgr coordinates."""
        h, w = bgr.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(bgr)
            if faces is None:
                return []
            return [tuple(int(v) for v in f[:4]) for f in faces]
        gray = self.cv2.equalizeHist(self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2GRAY))
        ms = max(20, int(min(h, w) * 0.06))
        for c in self.haar:
            det = c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(ms, ms))
            if len(det):
                return [tuple(int(v) for v in d) for d in det]
        return []


# =============================================================== ROI metrics
def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


GRID_W, GRID_H = 32, 18
_GRID_BG = np.ones((GRID_H, GRID_W), bool)
_GRID_BG[7:18, 8:24] = False  # person zone (PERSON_ZONE / 10)


def pack_grid(mask: np.ndarray) -> str:
    return np.packbits(mask.astype(np.uint8).ravel()).tobytes().hex()


def unpack_grid(hex_str: str) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(hex_str), np.uint8))[: GRID_W * GRID_H]
    return bits.reshape(GRID_H, GRID_W).astype(np.float32)


LOGO_SHIFT = 3  # grid cells of translation tolerance (~10 % of the ROI)
LOGO_SCALES = (0.8, 1.0, 1.25, 1.5)  # the virtual background is cropped / zoomed per camera


def expand_logo_templates(templates: np.ndarray) -> np.ndarray:
    """Add scaled copies (centre-cropped or zero-padded back to the grid size)."""
    import cv2
    out = []
    for tpl in (templates[None] if templates.ndim == 2 else templates):
        for sc in LOGO_SCALES:
            if sc == 1.0:
                out.append(tpl)
                continue
            r = cv2.resize(tpl, (int(round(GRID_W * sc)), int(round(GRID_H * sc))), interpolation=cv2.INTER_LINEAR)
            if sc > 1:
                y0, x0 = (r.shape[0] - GRID_H) // 2, (r.shape[1] - GRID_W) // 2
                out.append(r[y0:y0 + GRID_H, x0:x0 + GRID_W])
            else:
                pad = np.zeros((GRID_H, GRID_W), np.float32)
                y0, x0 = (GRID_H - r.shape[0]) // 2, (GRID_W - r.shape[1]) // 2
                pad[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
                out.append(pad)
    return np.stack(out).astype(np.float32)


def build_logo_template(rois: list[dict]) -> np.ndarray | None:
    """Mean coarse white-logo mask over ROIs where the dark-share rule fired confidently."""
    grids = [unpack_grid(r["white_grid"]) for r in rois if r.get("white_grid")]
    return np.mean(grids, axis=0) if grids else None


def logo_corr(white_grid_hex: str | None, templates: np.ndarray | None) -> float:
    """Best normalised cross-correlation of a ROI's coarse logo mask with any template.

    `templates` is one (18, 32) mask or a stack (k, 18, 32): one per reference tutor,
    because the virtual background is scaled / cropped differently per camera. Each
    template is also tried mirrored (self-view mirroring varies) and shifted by up to
    LOGO_SHIFT cells.
    """
    import cv2
    if not white_grid_hex or templates is None:
        return 0.0
    g = unpack_grid(white_grid_hex)
    if g.std() == 0:
        return 0.0
    tpls = templates[None] if templates.ndim == 2 else templates
    k = LOGO_SHIFT
    img = cv2.copyMakeBorder(g, k, k, k, k, cv2.BORDER_CONSTANT, value=0.0)
    best = 0.0
    for tpl in tpls:
        for t in (tpl, tpl[:, ::-1]):
            if t.std() == 0:
                continue
            r = cv2.matchTemplate(img, np.ascontiguousarray(t, np.float32), cv2.TM_CCOEFF_NORMED)
            r = r[np.isfinite(r)]
            if r.size:
                best = max(best, float(r.max()))
    return round(best, 3)


def roi_metrics(cv2, roi: np.ndarray, detector: FaceDetector, upscale: bool) -> dict:
    """branded virtual background, face, lighting, clarity metrics for one camera ROI."""
    patch = cv2.resize(roi, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    py0, py1, px0, px1 = PERSON_ZONE
    bg_mask = np.ones((PATCH_H, PATCH_W), bool)
    bg_mask[py0:py1, px0:px1] = False
    upper = np.zeros_like(bg_mask)
    upper[0:105, :] = True
    upper &= bg_mask

    dark = (V < 60) & (S < 100)
    white = (V > 155) & (S < 45)
    dark_ratio = float(dark[bg_mask].mean())
    white_ratio = float(white[upper].mean())
    dx = np.abs(np.diff(gray.astype(np.int16), axis=1)) > 45
    trans = float(dx[upper[:, 1:]].mean())
    score = 0.55 * _clip01((dark_ratio - 0.35) / 0.45) + 0.25 * _clip01(white_ratio / 0.05) + 0.20 * _clip01(trans / 0.02)

    # clarity / segmentation bleed on the outer background band
    outer = np.zeros_like(bg_mask)
    outer[:, : int(0.22 * PATCH_W)] = True
    outer[:, int(0.78 * PATCH_W):] = True
    outer[: int(0.30 * PATCH_H), :] = True
    logo_dil = cv2.dilate(white.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    bleed = ((S >= 42) & (V >= 50)) | ((V >= 65) & (V <= 145) & ~logo_dil)
    bleed_pct = float(bleed[outer].mean())
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # face (PiP crops are upscaled 2x so small / dim faces are caught)
    det_img = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) if upscale else roi
    if not upscale and det_img.shape[1] > 960:  # big full-screen ROI: faces are large, detect at 960 px
        f = 960 / det_img.shape[1]
        det_img = cv2.resize(det_img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    faces = detector.detect(det_img)
    face_lum = face_area = None
    if faces:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        g = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
        ix0, iy0 = max(0, x + int(0.15 * w)), max(0, y + int(0.15 * h))
        ix1, iy1 = min(g.shape[1], x + int(0.85 * w)), min(g.shape[0], y + int(0.85 * h))
        if ix1 > ix0 and iy1 > iy0:
            face_lum = float(g[iy0:iy1, ix0:ix1].mean())
        face_area = 100.0 * (w * h) / (det_img.shape[0] * det_img.shape[1])

    ycc = cv2.cvtColor(patch[50:175, 70:250], cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    skin = float(((Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127) & (Y > 40)).mean())

    return {"bg8_score": round(score, 3), "dark_bg_ratio": round(dark_ratio, 3),
            "white_logo_ratio": round(white_ratio, 4), "logo_transitions": round(trans, 4),
            "face": bool(faces), "face_count": len(faces),
            "face_luminance": round(face_lum, 1) if face_lum is not None else None,
            "face_area_pct": round(face_area, 2) if face_area is not None else None,
            "skin_center": round(skin, 3), "color_bleed_pct": round(bleed_pct, 3),
            "sharpness": round(sharp, 1), "mean_v": round(float(V.mean()), 1),
            "white_grid": pack_grid(cv2.resize(white.astype(np.float32), (GRID_W, GRID_H),
                                               interpolation=cv2.INTER_AREA) > 0.15)}


def temporal_metrics(cv2, roi_a: np.ndarray, roi_b: np.ndarray) -> dict:
    """Mean abs grey change over ~0.3 s: outer background band vs person zone."""
    a = cv2.cvtColor(cv2.resize(roi_a, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(roi_b, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    outer = np.zeros((PATCH_H, PATCH_W), bool)
    outer[:, : int(0.22 * PATCH_W)] = True
    outer[:, int(0.78 * PATCH_W):] = True
    outer[: int(0.30 * PATCH_H), :] = True
    py0, py1, px0, px1 = PERSON_ZONE
    return {"bg_flicker": round(float(d[outer].mean()), 2),
            "person_motion": round(float(d[py0:py1, px0:px1].mean()), 2)}


def _edge_ratio(profile: np.ndarray, at: int, search: int = 6, base: tuple[int, int] = (10, 40)) -> float:
    """Peak gradient near `at` relative to the median gradient just before it."""
    lo, hi = max(0, at - search), min(len(profile), at + search + 1)
    b0, b1 = max(0, at - base[1]), max(1, at - base[0])
    peak = float(profile[lo:hi].max()) if hi > lo else 0.0
    baseline = float(np.median(profile[b0:b1])) if b1 > b0 else 0.0
    return peak / (baseline + 1.0)


def measure_frame(cv2, frame: np.ndarray, detector: FaceDetector, prev_pip_small: np.ndarray | None,
                  frame_next: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    H, W = frame.shape[:2]
    x0, y1 = int(PIP_X * W), int(PIP_Y * H)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    pip = frame[0:y1, x0:W]
    left = frame[0:y1, int(LEFT_STRIP_X0 * W):int(LEFT_STRIP_X1 * W)]
    pg = cv2.cvtColor(pip, cv2.COLOR_BGR2GRAY)

    # PiP border: vertical edge at x0 (rows 0..y1). The tile's bottom edge varies by
    # recording (measured 0.10-0.20 H), so search for the strongest horizontal edge
    # across 0.08-0.24 H relative to the median row gradient there.
    vprof = np.abs(np.diff(gray[0:y1, x0 - 60:min(W, x0 + 20)], axis=1)).mean(axis=0)
    border_v = _edge_ratio(vprof, 59)
    hprof = np.abs(np.diff(gray[int(0.08 * H):int(0.24 * H), x0:W], axis=0)).mean(axis=1)
    border_h = float(hprof.max() / (np.median(hprof) + 1.0)) if hprof.size else 0.0
    color_diff = float(np.abs(pip.reshape(-1, 3).mean(0) - left.reshape(-1, 3).mean(0)).mean())

    small = cv2.resize(pg, (64, 38), interpolation=cv2.INTER_AREA).astype(np.int16)
    static_diff = float(np.abs(small - prev_pip_small).mean()) if prev_pip_small is not None else None

    cy0, cy1, cx0, cx1 = CENTER
    center = frame[int(cy0 * H):int(cy1 * H), int(cx0 * W):int(cx1 * W)]
    c_small = cv2.resize(center, (448, 264), interpolation=cv2.INTER_AREA)
    cg = cv2.cvtColor(c_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_std = np.sqrt(np.maximum(cv2.blur(cg * cg, (3, 3)) - cv2.blur(cg, (3, 3)) ** 2, 0))
    flat = float((local_std < 1.5).mean())
    sx = np.abs(cv2.Sobel(cg, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    sy = np.abs(cv2.Sobel(cg, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
    lines = float(((sx > 4 * (np.median(sx) + 1)).mean() + (sy > 4 * (np.median(sy) + 1)).mean()) / 2)
    c_hsv = cv2.cvtColor(c_small, cv2.COLOR_BGR2HSV)

    rec = {
        "pip_mean": round(float(pg.mean()), 1), "pip_std": round(float(pg.std()), 1),
        "border_v": round(border_v, 2), "border_h": round(border_h, 2),
        "pip_left_color_diff": round(color_diff, 1),
        "pip_static_diff": round(static_diff, 2) if static_diff is not None else None,
        "center_flat": round(flat, 3), "center_lines": round(lines, 4),
        "center_v": round(float(c_hsv[..., 2].mean()), 1), "center_s": round(float(c_hsv[..., 1].mean()), 1),
        "pip": roi_metrics(cv2, pip, detector, upscale=True),
        "full": roi_metrics(cv2, frame[0:int(FULL_CAM_Y1 * H), 0:W], detector, upscale=False),
    }
    if frame_next is not None and frame_next.shape == frame.shape:
        rec["pip"].update(temporal_metrics(cv2, pip, frame_next[0:y1, x0:W]))
        fy = int(FULL_CAM_Y1 * H)
        rec["full"].update(temporal_metrics(cv2, frame[0:fy, 0:W], frame_next[0:fy, 0:W]))
    return rec, small


TEMPORAL_GAP_FRAMES = 8  # ~0.3 s at 25 fps


def measure_video(video_path: str | Path, detector: FaceDetector, sample_every_sec: float = 10.0,
                  skip_sec: float = 15.0) -> dict:
    """Sample one frame every `sample_every_sec` (skipping the first/last 15 s)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"open_ok": False, "frames": []}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    frames, prev = [], None
    t = skip_sec
    while t <= dur - skip_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            for _ in range(TEMPORAL_GAP_FRAMES - 1):
                cap.grab()
            ok2, frame2 = cap.read()
            rec, prev = measure_frame(cv2, frame, detector, prev, frame2 if ok2 else None)
            rec["t"] = round(t, 1)
            rec["H"], rec["W"] = frame.shape[:2]
            frames.append(rec)
        t += sample_every_sec
    cap.release()
    return {"open_ok": True, "duration_sec": round(dur, 1), "sample_every_sec": sample_every_sec,
            "detector": detector.kind, "frames": frames}


# =============================================================== aggregation
def _ts(sec: float) -> str:
    sec = max(0, int(sec))
    return f"[{sec // 60:02d}:{sec % 60:02d}]"


def is_8x8_dark_rule(m: dict, t: dict = T) -> bool:
    return m["dark_bg_ratio"] >= t["bg8_dark"] and m["white_logo_ratio"] >= t["bg8_white"] and m["bg8_score"] >= t["bg8_score"]


def is_8x8(m: dict, t: dict = T) -> bool:
    """branded virtual background if EITHER the dark-share rule OR the logo-structure match fires.

    The logo match matters when the white logo fills a large share of the ROI
    (full-screen close-ups, small PiP tiles) and the dark share drops.
    `logo_corr` is attached by `aggregate_visual` when a template is supplied.
    """
    return is_8x8_dark_rule(m, t) or m.get("logo_corr", 0.0) >= t["logo_corr"]


def classify_frames(frames: list[dict], t: dict = T) -> list[str]:
    """Per-frame layout. Full-screen camera = a large face in the main canvas (faces
    are ~0.3 % of the frame inside the PiP vs > 2 % full screen), or a continuous
    branded-background frame with no PiP border. Everything else is a screen share.

    Note: compressed camera video is as 'flat' and line-rich as a shared screen at
    analysis scale, so centre flatness / straight-line cues are kept only as
    diagnostics — they did not separate the layouts on the 12 classes.
    """
    modes, static_run = [], 0
    for f in frames:
        full = f["full"]
        big_face = bool(full["face"]) and (full.get("face_area_pct") or 0) >= t["full_face_area"]
        no_border = f["border_v"] < t["nopip_border"] and f["border_h"] < t["nopip_border"]
        sd = f.get("pip_static_diff")
        static_run = static_run + 1 if (sd is not None and sd < t["nopip_static_diff"]) else 0
        if big_face or (no_border and is_8x8(full, t)):
            modes.append("speaker_fullscreen")
            continue
        p = f["pip"]
        p8 = is_8x8(p, t)
        # Pure-black panel beside live shared content = the video thumbnail panel is
        # hidden; the real camera-off canvas is flat grey (~27), not black.
        if (f["pip_mean"] < t["hidden_black_mean"] and f["pip_std"] < t["nopip_black_std"]
                and f["border_v"] >= t["nopip_border"]):
            modes.append("screenshare_pip_hidden")
            continue
        black = f["pip_std"] < t["nopip_black_std"] and f["pip_mean"] < t["nopip_black_mean"]
        seamless = (f["border_v"] < t["nopip_border"] and f["pip_left_color_diff"] < t["nopip_color_diff"]
                    and not p8 and not p["face"])
        frozen = static_run >= t["nopip_static_run"] - 1 and not p["face"] and not p8
        modes.append("screenshare_no_pip" if (black or seamless or frozen) else "screenshare_pip")
    return modes


def aggregate_visual(measured: dict, t: dict = T, logo_template: np.ndarray | None = None) -> dict:
    frames = measured.get("frames", [])
    if logo_template is not None:
        for f in frames:
            for k in ("pip", "full"):
                f[k]["logo_corr"] = logo_corr(f[k].get("white_grid"), logo_template)
    n = len(frames)
    if not n:
        return {"sampled_frames": 0}
    step = measured.get("sample_every_sec", 10.0)
    modes = classify_frames(frames, t)
    cams = []  # (frame, roi metrics) for active camera frames
    for f, m in zip(frames, modes):
        if m == "speaker_fullscreen":
            cams.append((f, f["full"]))
        elif m == "screenshare_pip":
            cams.append((f, f["pip"]))

    def pct(k, d):
        return round(100.0 * k / d, 1) if d else 0.0

    off = [f for f, m in zip(frames, modes) if m == "screenshare_no_pip"]
    streak = best = 0
    best_end = None
    for f, m in zip(frames, modes):
        streak = streak + 1 if m == "screenshare_no_pip" else 0
        if streak > best:
            best, best_end = streak, f["t"]
    bg8 = [(f, r) for f, r in cams if is_8x8(r, t)]
    bg8_face = [(f, r) for f, r in bg8 if r["face"]]
    tutor_faces = [r["face_luminance"] for _, r in bg8_face if r["face_luminance"] is not None]
    if not bg8:  # no branded-background tile at all: use every active camera face
        tutor_faces = [r["face_luminance"] for _, r in cams if r["face"] and r["face_luminance"] is not None]
    # Clarity measures only on frames where the dark navy field dominates: on logo-heavy
    # ROIs (caught by the logo match) the logo's anti-aliased mid-tones read as "bleed".
    bg8_dark = [(f, r) for f, r in bg8 if is_8x8_dark_rule(r, t)]
    bleed = [r["color_bleed_pct"] for _, r in bg8_dark]
    glitch = [(f, r) for f, r in bg8_dark if r["color_bleed_pct"] >= t["glitch_bleed"]]
    any_face = sum(1 for f, r in cams if r["face"])
    bg8_pct = pct(len(bg8), n)
    lum = float(np.mean(tutor_faces)) if tutor_faces else 0.0
    student_cam = [(f, r) for f, r in cams if not is_8x8(r, t) and (r["face"] or r["skin_center"] > 0.15)]

    out = {
        "sampled_frames": n,
        "fullscreen_cam_pct": pct(modes.count("speaker_fullscreen"), n),
        "screenshare_pip_pct": pct(modes.count("screenshare_pip"), n),
        "camera_off_or_missing_pct": pct(len(off), n),
        "camera_active_pct": pct(len(cams), n),
        "pip_panel_hidden_pct": pct(modes.count("screenshare_pip_hidden"), n),
        "sample_every_sec": step,
        "camera_off_times_sec": [f["t"] for f in off],  # for camera-off-during-speech refinement
        "max_camera_off_streak_sec": float(best * step),
        "bg_8x8_frame_pct": bg8_pct,
        "bg_8x8_when_cam_active_pct": pct(len(bg8), len(cams)),
        "tutor_face_visible_when_8x8_pct": pct(len(bg8_face), len(bg8)),
        "overall_face_visible_pct": pct(any_face, n),
        "tutor_face_luminance_mean": round(lum, 1),
        "bg_color_bleed_mean_pct": round(100 * float(np.mean(bleed)), 1) if bleed else 0.0,
        "bg_glitch_frames_pct": pct(len(glitch), len(bg8_dark)),
        "roi_sharpness_mean": round(float(np.mean([r["sharpness"] for _, r in bg8_dark])), 1) if bg8_dark else None,
        "bg8_by_dark_rule_pct": pct(sum(1 for _, r in cams if is_8x8_dark_rule(r, t)), n),
        "bg8_by_logo_match_pct": pct(sum(1 for _, r in cams if r.get("logo_corr", 0.0) >= t["logo_corr"]), n),
        "bg_flicker_steady_face": _steady_flicker(bg8_dark),
        "student_active_speaker_cam_pct": pct(len(student_cam), len(cams)) if bg8_pct >= t["flag_no_8x8"] else None,
        "visual_evidence_timestamps": {
            "camera_off_windows": [_ts(f["t"]) for f in off[:3]] + (
                [f"longest {_ts(best_end - (best - 1) * step)}-{_ts(best_end)}"] if best >= 2 else []),
            "bg_8x8_missing_samples": [_ts(f["t"]) for f, r in cams if not is_8x8(r, t)][:3],
            "face_missing_on_8x8_samples": [_ts(f["t"]) for f, r in bg8 if not r["face"]][:3],
            "low_face_lighting_samples": [_ts(f["t"]) for f, r in (bg8_face or cams)
                                          if r["face_luminance"] is not None and r["face_luminance"] < t["flag_dark_face"]][:3],
            "bg_glitch_samples": [_ts(f["t"]) for f, _ in glitch[:3]],
        },
    }
    out.update({
        # Share of active-camera frames, so hidden-panel / camera-off time doesn't count
        # against the background (good_1: 62 % hidden panel, 77 % branded-background when visible).
        "flag_no_8x8_background": out["bg_8x8_when_cam_active_pct"] < t["flag_no_8x8_active"],
        "flag_face_not_visible": (bg8_pct >= t["flag_no_8x8"] and out["tutor_face_visible_when_8x8_pct"] < t["flag_face_on_8x8"])
                                 or out["overall_face_visible_pct"] < t["flag_face_overall"],
        "flag_intermittent_camera_off": out["camera_off_or_missing_pct"] >= t["flag_cam_off_pct"]
                                        or out["max_camera_off_streak_sec"] >= t["flag_cam_off_streak"],
        "flag_poor_lighting": lum > 0 and lum < t["flag_dark_face"],
        # needs enough frames where the dark field is measurable (dark-share rule)
        "flag_bg_clarity_issue": pct(len(bg8_dark), n) >= t["flag_no_8x8"] and (
            out["bg_glitch_frames_pct"] >= t["flag_glitch_frames"] or out["bg_color_bleed_mean_pct"] >= 100 * t["flag_bleed"]),
    })
    return out


def _steady_flicker(bg8: list) -> float | None:
    """Mean background change over ~0.3 s on branded-background frames where the person is steady
    (person-zone motion at or below that video's median)."""
    pairs = [(r["bg_flicker"], r["person_motion"]) for _, r in bg8 if r.get("bg_flicker") is not None]
    if len(pairs) < 5:
        return None
    med = float(np.median([pm for _, pm in pairs]))
    steady = [bf for bf, pm in pairs if pm <= med]
    return round(float(np.mean(steady)), 2) if steady else None


def extract_visual_features(video_path: str | Path, sample_every_sec: float = 10.0,
                            models_dir: str | Path | None = None) -> dict:
    """Measure + aggregate in one call (see module docstring)."""
    models_dir = models_dir or default_models_dir()
    measured = measure_video(video_path, FaceDetector(models_dir), sample_every_sec)
    return {**aggregate_visual(measured, logo_template=load_logo_templates(models_dir)),
            "detector": measured.get("detector")}
