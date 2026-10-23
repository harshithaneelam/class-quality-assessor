"""Visual + audio feature tests on synthetic frames / audio (no real recordings)."""
import wave

import cv2
import numpy as np
import pytest

from src import visual_features as vf
from src.audio_features import extract_audio_features


class NoFaces:
    kind = "none"

    def detect(self, bgr):
        return []


class OneFace:
    kind = "fake"

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        return [(int(0.4 * w), int(0.3 * h), int(0.2 * w), int(0.3 * h))]


def bg8_tile(w=320, h=180):
    """Dark navy field + white checkerboard logo in the upper band + a person blob."""
    img = np.full((h, w, 3), (30, 22, 19), np.uint8)  # BGR dark navy (V~30, S~93)
    for i in range(8):  # checkerboard logo, top-left
        for j in range(3):
            if (i + j) % 2 == 0:
                img[10 + j * 10:20 + j * 10, 20 + i * 10:30 + i * 10] = 235
    cv2.putText(img, "branded-background", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 240, 240), 3)
    cv2.ellipse(img, (160, 150), (55, 70), 0, 0, 360, (120, 150, 200), -1)  # skin-ish person
    return img


def room_tile(w=320, h=180):
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (90, 160, 210)  # warm wall
    img[100:, :120] = (40, 90, 150)  # wooden furniture
    return cv2.add(img, rng.integers(0, 25, img.shape, dtype=np.uint8))


def test_8x8_background_detected_and_room_rejected():
    m8 = vf.roi_metrics(cv2, bg8_tile(), NoFaces(), upscale=True)
    mr = vf.roi_metrics(cv2, room_tile(), NoFaces(), upscale=True)
    assert vf.is_8x8(m8) and m8["bg8_score"] > 0.6
    assert not vf.is_8x8(mr)
    assert m8["color_bleed_pct"] < mr["color_bleed_pct"]


def test_bleed_rises_when_room_leaks_through_8x8():
    clean = bg8_tile()
    leaky = clean.copy()
    leaky[60:180, 0:70] = room_tile()[60:180, 0:70]  # chair / wall bleeding in on the left
    c = vf.roi_metrics(cv2, clean, NoFaces(), upscale=True)["color_bleed_pct"]
    lk = vf.roi_metrics(cv2, leaky, NoFaces(), upscale=True)["color_bleed_pct"]
    assert lk > c + 0.1


def test_face_luminance_uses_inner_box():
    img = bg8_tile()
    m = vf.roi_metrics(cv2, img, OneFace(), upscale=False)
    assert m["face"] and m["face_luminance"] is not None and m["face_area_pct"] == pytest.approx(6.0, abs=0.5)


def _frame(center_flat, center_lines, pip, border_v=3.0, pip_std=40.0, pip_mean=80.0, diff=40.0, static=5.0, t=0.0):
    return {"t": t, "center_flat": center_flat, "center_v": 120.0, "center_lines": center_lines,
            "border_v": border_v, "border_h": border_v, "pip_std": pip_std, "pip_mean": pip_mean,
            "pip_left_color_diff": diff, "pip_static_diff": static, "pip": pip, "full": pip}


def _roi(is8, face, lum=130.0, bleed=0.02):
    return {"bg8_score": 0.9 if is8 else 0.1, "dark_bg_ratio": 0.8 if is8 else 0.1,
            "white_logo_ratio": 0.05 if is8 else 0.0, "face": face, "face_luminance": lum if face else None,
            "skin_center": 0.3 if face else 0.0, "color_bleed_pct": bleed, "sharpness": 300.0}


def test_layout_classification():
    frames = [
        _frame(0.1, 0.0, _roi(True, True), border_v=1.0),                # continuous camera -> fullscreen
        _frame(0.7, 0.05, _roi(True, True)),                             # board + branded-background PiP
        _frame(0.7, 0.05, _roi(False, False), pip_std=3, pip_mean=10),   # black PiP -> no pip
        _frame(0.7, 0.05, _roi(False, False), border_v=1.0, diff=4.0),   # seamless desktop -> no pip
    ]
    assert vf.classify_frames(frames) == ["speaker_fullscreen", "screenshare_pip",
                                          "screenshare_no_pip", "screenshare_no_pip"]


def test_static_pip_without_face_is_camera_off():
    frames = [_frame(0.7, 0.05, _roi(False, False), static=0.2, t=10 * i) for i in range(4)]
    assert vf.classify_frames(frames)[-1] == "screenshare_no_pip"


def test_aggregate_flags():
    on8 = [_frame(0.7, 0.05, _roi(True, False), t=10 * i) for i in range(10)]           # branded-background tile, no face
    off = [_frame(0.7, 0.05, _roi(False, False), pip_std=2, pip_mean=5, t=100 + 10 * i) for i in range(15)]
    agg = vf.aggregate_visual({"frames": on8 + off, "sample_every_sec": 10.0})
    assert agg["bg_8x8_frame_pct"] == 40.0 and agg["tutor_face_visible_when_8x8_pct"] == 0.0
    assert agg["flag_face_not_visible"] and agg["flag_intermittent_camera_off"]
    assert agg["max_camera_off_streak_sec"] == 150.0
    assert not agg["flag_no_8x8_background"]
    assert agg["visual_evidence_timestamps"]["camera_off_windows"][0] == "[01:40]"

    lit = [_frame(0.1, 0.0, _roi(True, True, lum=40.0, bleed=0.3), border_v=1.0, t=10 * i) for i in range(10)]
    agg2 = vf.aggregate_visual({"frames": lit, "sample_every_sec": 10.0})
    assert agg2["flag_poor_lighting"] and agg2["flag_bg_clarity_issue"] and not agg2["flag_face_not_visible"]


def test_audio_silence_and_prosody(tmp_path):
    sr = 16000
    t = np.arange(sr) / sr
    speech = lambda f: (0.3 * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)))
    silence = np.zeros(sr)
    blocks = [speech(150 + 20 * (i % 5)) for i in range(20)] + [silence] * 12 + \
             [speech(200) for _ in range(10)] + [silence] * 6 + [speech(180) for _ in range(10)]
    audio = np.concatenate(blocks) + np.random.default_rng(0).normal(0, 1e-4, len(blocks) * sr)
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    a = extract_audio_features(p)
    assert a["speech_ratio_pct"] == pytest.approx(100 * 40 / 58, abs=1)
    assert a["silence_gaps_gt_5s_count"] == 2 and a["silence_gaps_gt_10s_count"] == 1
    assert a["max_silence_gap_sec"] == 12.0 and a["max_silence_gap_ts"] == "[00:20]-[00:32]"
    assert a["dead_air_total_sec"] == 18.0
    assert 140 <= a["pitch_f0_hz_median"] <= 240
    assert 0 <= a["energy_liveliness_score"] <= 1


def test_logo_match_catches_logo_heavy_tile_dark_rule_misses():
    ref = [vf.roi_metrics(cv2, bg8_tile(), NoFaces(), upscale=True) for _ in range(3)]
    tpl = vf.build_logo_template(ref)
    # Close-up: the same logo layout but much larger white share, little dark area left.
    heavy = bg8_tile()
    heavy[0:105][heavy[0:105] < 60] = 120  # lift the dark field to mid-grey -> dark-share rule fails
    m = vf.roi_metrics(cv2, heavy, NoFaces(), upscale=True)
    assert not vf.is_8x8_dark_rule(m)
    m["logo_corr"] = vf.logo_corr(m["white_grid"], tpl)
    assert m["logo_corr"] >= vf.T["logo_corr"] and vf.is_8x8(m)
    room = vf.roi_metrics(cv2, room_tile(), NoFaces(), upscale=True)
    assert vf.logo_corr(room["white_grid"], tpl) < vf.T["logo_corr"]
    mirrored = vf.roi_metrics(cv2, bg8_tile()[:, ::-1].copy(), NoFaces(), upscale=True)
    assert vf.logo_corr(mirrored["white_grid"], tpl) >= vf.T["logo_corr"]


def test_pure_black_panel_is_hidden_not_camera_off():
    hidden = _frame(0.7, 0.05, _roi(False, False), border_v=7.0, pip_std=2.0, pip_mean=0.1)
    grey_off = _frame(0.7, 0.05, _roi(False, False), border_v=0.0, pip_std=0.0, pip_mean=27.0, diff=0.0)
    assert vf.classify_frames([hidden, grey_off]) == ["screenshare_pip_hidden", "screenshare_no_pip"]
    agg = vf.aggregate_visual({"frames": [hidden] * 5, "sample_every_sec": 10.0})
    assert agg["pip_panel_hidden_pct"] == 100.0 and agg["camera_off_or_missing_pct"] == 0.0


def test_temporal_flicker_on_steady_face():
    a = bg8_tile()
    b = a.copy()
    b[:, :60] = cv2.add(b[:, :60], np.full_like(b[:, :60], 40))  # background band flickers
    t = vf.temporal_metrics(cv2, a, b)
    assert t["bg_flicker"] > 5 and t["person_motion"] < 1
    rois = [(None, {**_roi(True, True), "bg_flicker": bf, "person_motion": pm})
            for bf, pm in [(10, 1), (12, 1), (2, 9), (3, 8), (11, 1), (1, 9)]]
    assert vf._steady_flicker(rois) == pytest.approx(11.0)


def test_no_8x8_flag_uses_active_camera_share():
    hidden = [_frame(0.7, 0.05, _roi(False, False), border_v=7.0, pip_std=2.0, pip_mean=0.1, t=10 * i) for i in range(8)]
    tiles = [_frame(0.7, 0.05, _roi(True, True), t=100 + 10 * i) for i in range(2)]
    agg = vf.aggregate_visual({"frames": hidden + tiles, "sample_every_sec": 10.0})
    assert agg["bg_8x8_frame_pct"] == 20.0 and agg["bg_8x8_when_cam_active_pct"] == 100.0
    assert not agg["flag_no_8x8_background"]
    rooms = [_frame(0.7, 0.05, _roi(False, True), t=10 * i) for i in range(4)]
    assert vf.aggregate_visual({"frames": rooms + tiles, "sample_every_sec": 10.0})["flag_no_8x8_background"]
