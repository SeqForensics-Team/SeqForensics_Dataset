"""
╔══════════════════════════════════════════════════════════════════╗
║   VIDEO FORENSICS — GPU PIPELINE  (60-SEQUENCE BALANCED EDITION) ║
║                                                                  ║
║  Generates ALL 60 possible ordered 3-step manipulation sequences║
║  (P(5,3) = 60) for EVERY source video, guaranteeing perfect      ║
║  class balance both overall and per sequence-position — see the ║
║  combinatorial proof worked out before this code was written:   ║
║    - 10 distinct 3-of-5 method combinations                     ║
║    - x 6 orderings each = 60 total ordered sequences            ║
║    - every method appears in exactly 36/60 sequences overall    ║
║    - every method appears in exactly 12/60 sequences at EACH    ║
║      of step1/step2/step3 individually                          ║
║                                                                  ║
║  1000 source videos x 60 sequences = 60,000 output videos.      ║
║                                                                  ║
║  All heavy computation runs on GPU via PyTorch CUDA tensors.    ║
║  OpenCV used ONLY for video I/O (read/write frames).             ║
║                                                                  ║
║  Tested on: NVIDIA RTX 4500 Ada, CUDA 12.x driver                ║
╚══════════════════════════════════════════════════════════════════╝

WHAT CHANGED FROM THE PREVIOUS SCRIPT (generate_sequences.py), AND WHY:

  1. GENERATION PLAN — completely replaced.
     OLD: 25% of sources -> fixed combo {FS,FSh,DF} x6 perms
          25% of sources -> fixed combo {NT,FS,F2F} x6 perms
          50% of sources -> 1 random 3-of-5 combo, 1 order only
          This structurally over-represented FS (it was in BOTH fixed
          combos), which is what caused the FS-collapse during training.
     NEW: every source video gets ALL 60 possible ordered sequences.
          No combo or order is ever more represented than any other,
          by construction — confirmed by exhaustive enumeration.

  2. build_video_map() — now reports EXACTLY which source IDs were
     dropped and WHICH specific manipulation file was missing for each,
     instead of a single aggregate "N skipped" count with no detail.
     This is what silently cost you ~45 source videos last time with
     no way to know which ones or why.

  3. process_all() — no longer wraps an entire source video's 60
     sequences in one try/except. A single missing/corrupt file for
     one source no longer takes out all 60 of that source's outputs;
     each sequence's success/failure is tracked independently, and the
     source's frames are loaded ONCE and reused across all 60 sequences
     for efficiency (loading 1000 source/manip videos 60 times each
     would be enormously wasteful).

  4. RECONCILIATION CHECK — new, runs at the very end. Computes the
     exact expected output count (n_sources_processed x 60), compares
     against actual files written, and if they don't match, prints the
     EXACT list of missing (video_id, sequence) pairs so you know
     precisely what to re-run, instead of discovering a shortfall by
     counting CSV rows after the fact.

Setup:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install cupy-cuda12x opencv-python numpy

Verify GPU:
    python -c "import torch; print(torch.cuda.is_available())"

Usage:
    python generate_sequences_60k.py                  # all videos, all 60 sequences
    python generate_sequences_60k.py --video 000       # single source video, all 60 sequences
    python generate_sequences_60k.py --preview          # mask stats only, no video output
    python generate_sequences_60k.py --batch-size 64    # frames per GPU batch (tune to VRAM)
    python generate_sequences_60k.py --reconcile-only   # skip generation, just check what's missing
"""

import cv2
import numpy as np
import json
import csv
import argparse
import time
import sys
import subprocess
import shutil
import traceback
from pathlib import Path
from itertools import combinations, permutations
from datetime import datetime
from typing import TypedDict

import torch
import torch.nn.functional as F

# ── Verify GPU at import time ────────────────────────────────────
if not torch.cuda.is_available():
    print("[ERROR] No CUDA GPU detected by PyTorch.")
    print("  Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("  Then verify: python -c \"import torch; print(torch.cuda.is_available())\"")
    sys.exit(1)

DEVICE     = torch.device("cuda:0")
GPU_NAME   = torch.cuda.get_device_name(0)
VRAM_GB    = torch.cuda.get_device_properties(0).total_memory / (1024**3)


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

ROOT      = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "archive"
    / "FaceForensics++_C23"
)
OUT_DIR   = ROOT / "generated_videos_60k"
LABEL_DIR = ROOT / "labels_60k"
LOG_DIR   = ROOT / "logs_60k"

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

CRF     = 23
PIX_FMT = "yuv444p"

# Mask parameters (tuned for FF++ face region)
DIFF_THRESHOLD = 8      # LAB diff sensitivity
MORPH_CLOSE_PX = 25     # fills holes inside face
DILATE_PX      = 15     # expands mask past detected boundary
FEATHER_PX     = 31     # gaussian feather kernel size

# Edit registry — maps short key -> human name
EDIT_REGISTRY = {
    "FS":  {"name": "FaceSwap"},
    "FSh": {"name": "FaceShifter"},
    "DF":  {"name": "DeepFake"},
    "NT":  {"name": "NeuralTextures"},
    "F2F": {"name": "Face2Face"},
}

ALL_EDIT_KEYS: list[str] = ["FS", "FSh", "DF", "NT", "F2F"]

# ── Folder names on disk for each edit key ───────────────────────
# FF++ uses "DeepFakes" (plural) on disk but "DF" key maps to it.
FOLDER_MAP = {
    "FS":  "FaceSwap",
    "FSh": "FaceShifter",
    "DF":  "DeepFakes",       # note: plural on disk
    "NT":  "NeuralTextures",
    "F2F": "Face2Face",
}


# ══════════════════════════════════════════════════════════════════
#  THE 60-SEQUENCE UNIVERSE  — derived once, used for every source
# ══════════════════════════════════════════════════════════════════
#
# P(5,3) = 5!/(5-3)! = 60 ordered sequences. Generated directly via
# itertools.permutations rather than combinations+permutations nested,
# since permutations() over the full 5-item list with r=3 already
# enumerates every ordered 3-tuple without repetition — simpler and
# exactly equivalent to the combo-then-permute construction we verified
# by hand (10 combos x 6 orderings = 60).

ALL_60_SEQUENCES: list[tuple] = list(permutations(ALL_EDIT_KEYS, 3))

assert len(ALL_60_SEQUENCES) == 60, (
    f"Expected exactly 60 ordered sequences from P(5,3), got {len(ALL_60_SEQUENCES)}. "
    f"This should be mathematically impossible to fail — check ALL_EDIT_KEYS has 5 unique entries."
)


def verify_60_sequence_balance():
    """
    Defensive check run once at startup: confirms the 60-sequence
    universe is balanced overall AND per-position, exactly as derived
    by hand before this script was written. If this ever fails, ALL_EDIT_KEYS
    was changed incorrectly (e.g. a duplicate or a typo) — fail loudly
    rather than silently generating a biased dataset again.
    """
    from collections import Counter
    overall = Counter()
    per_position = [Counter() for _ in range(3)]
    for seq in ALL_60_SEQUENCES:
        for pos, method in enumerate(seq):
            overall[method] += 1
            per_position[pos][method] += 1

    expected_overall = 36     # 60 sequences x 3 positions / 5 methods
    expected_per_pos = 12     # 60 sequences / 5 methods

    for method in ALL_EDIT_KEYS:
        assert overall[method] == expected_overall, (
            f"Balance check FAILED: {method} appears {overall[method]} times overall, "
            f"expected {expected_overall}."
        )
        for pos in range(3):
            assert per_position[pos][method] == expected_per_pos, (
                f"Balance check FAILED: {method} appears {per_position[pos][method]} times "
                f"at position {pos+1}, expected {expected_per_pos}."
            )
    return True


# ══════════════════════════════════════════════════════════════════
#  VIDEO MAP — auto-discovered from disk, WITH DETAILED SKIP REPORTING
# ══════════════════════════════════════════════════════════════════

def build_video_map(log_fn=print) -> tuple:
    """
    Scan ROOT/original/*.mp4 and find matching manipulation files
    in each of the 5 method folders.

    FIX vs the old version: returns BOTH the video_map AND a detailed
    list of (video_id, reason) for every source dropped, instead of
    just a count. This is the single biggest fix for the "generated
    fewer videos than expected, don't know why" problem — now you get
    an exact, itemized report every run.

    Returns:
        (video_map, skip_report)
        video_map:   { "000": {"original": Path, "FS": Path, ...}, ... }
        skip_report: [ {"video_id": "037", "reason": "missing DF file"}, ... ]
    """
    video_map = {}
    skip_report = []
    orig_dir  = ROOT / "original"

    if not orig_dir.exists():
        log_fn(f"[ERROR] original folder not found: {orig_dir}")
        sys.exit(1)

    originals = sorted(orig_dir.glob("*.mp4"))
    if not originals:
        log_fn(f"[ERROR] No .mp4 files found in {orig_dir}")
        sys.exit(1)

    for orig_path in originals:
        vid = orig_path.stem

        entry = {"original": orig_path}
        missing_methods = []

        for key, folder in FOLDER_MAP.items():
            manip_dir = ROOT / folder
            if not manip_dir.exists():
                missing_methods.append(f"{key} (folder {folder}/ does not exist)")
                continue

            matches = sorted(manip_dir.glob(f"{vid}_*.mp4"))
            if not matches:
                matches = sorted(manip_dir.glob(f"{vid}.mp4"))
            if not matches:
                missing_methods.append(f"{key} (no {vid}_*.mp4 or {vid}.mp4 in {folder}/)")
                continue

            entry[key] = matches[0]

        if missing_methods:
            skip_report.append({
                "video_id": vid,
                "reason": "; ".join(missing_methods),
            })
        else:
            video_map[vid] = entry

    log_fn(f"[INFO] Discovered {len(video_map)} complete source videos "
          f"({len(skip_report)} skipped — see detailed report below)")

    if skip_report:
        log_fn(f"[INFO] Detailed skip report ({len(skip_report)} source videos dropped):")
        for entry in skip_report:
            log_fn(f"  {entry['video_id']}: {entry['reason']}")

    return video_map, skip_report


# Build once at module load
VIDEO_MAP, VIDEO_MAP_SKIP_REPORT = build_video_map()


# ══════════════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f     = open(path, "w", encoding="utf-8")
        self.start = time.time()

    def _w(self, line):
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def _ts(self): return datetime.now().strftime("%H:%M:%S")
    def info(self,  m): self._w(f"[{self._ts()}] [INFO ] {m}")
    def ok(self,    m): self._w(f"[{self._ts()}] [OK   ] {m}")
    def warn(self,  m): self._w(f"[{self._ts()}] [WARN ] {m}")
    def error(self, m): self._w(f"[{self._ts()}] [ERROR] {m}")

    def section(self, t):
        self._w(f"\n{'─'*64}\n  {t}\n{'─'*64}")

    def progress(self, done, total, label=""):
        elapsed = time.time() - self.start
        rate    = done / elapsed if elapsed > 0 else 1e-9
        eta     = (total - done) / rate
        msg = (f"[{self._ts()}] [INFO ]   {label}{done}/{total}  "
               f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
        print(msg, end="\r")
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        print()
        self.info(f"Total runtime: {time.time()-self.start:.1f}s")
        self.f.close()


# ══════════════════════════════════════════════════════════════════
#  VIDEO I/O  (CPU — OpenCV for reading/writing only)
# ══════════════════════════════════════════════════════════════════

def load_video(path: Path, log: Logger):
    """Load all frames into RAM as uint8 numpy arrays."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    log.info(f"  Loaded {len(frames):>4} frames  <-  {path.name}")
    return frames, fps, width, height


def validate_inputs(video_id: str, edit_keys: list[str], log: Logger) -> tuple:
    """
    Returns (ok: bool, missing: list[str]) instead of just a bool, so
    callers can report exactly what's missing rather than a generic failure.
    """
    info = VIDEO_MAP[video_id]
    required = ["original"] + edit_keys
    missing = [
        str(info[k])
        for k in required
        if k not in info or not info[k].exists()
    ]
    if missing:
        for m in missing:
            log.error(f"  Missing: {m}")
    return (len(missing) == 0, missing)


def write_video_ffmpeg(frames: list, out_path: Path,
                       fps: float, width: int, height: int, log: Logger):
    """
    Write frames via ffmpeg pipe -> libx264 / yuv444p / CRF23.
    Identical encoding to FF++ C23 — important for forensic consistency.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which(FFMPEG) is None and not Path(FFMPEG).exists():
        raise RuntimeError(
            f"ffmpeg not found. Install: winget install ffmpeg\n"
            f"Then restart PowerShell."
        )

    cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
        "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-crf", str(CRF),
        "-pix_fmt", PIX_FMT,
        "-movflags", "+faststart",
        "-loglevel", "error",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read().decode(errors="ignore")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path.name}:\n{stderr}")
    mb = out_path.stat().st_size / (1024 * 1024)
    log.ok(f"  Saved -> {out_path.name}  ({mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════
#  GPU TENSOR UTILITIES
# ══════════════════════════════════════════════════════════════════

def frames_to_gpu(frames: list, start: int, end: int) -> torch.Tensor:
    batch = np.stack(frames[start:end], axis=0).astype(np.float32)
    return torch.from_numpy(batch).to(DEVICE)


def tensor_to_frames(t: torch.Tensor) -> list:
    arr = t.clamp(0, 255).byte().cpu().numpy()
    return [arr[i] for i in range(arr.shape[0])]


# ══════════════════════════════════════════════════════════════════
#  GPU COLOR CONVERSION — BGR -> LAB
# ══════════════════════════════════════════════════════════════════

def bgr_to_lab_gpu(bgr: torch.Tensor) -> torch.Tensor:
    rgb = bgr[..., [2, 1, 0]] / 255.0

    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )

    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=torch.float32, device=DEVICE)

    xyz = torch.einsum("...c,dc->...d", linear, M)

    xyz[..., 0] /= 0.95047
    xyz[..., 2] /= 1.08883

    eps   = 0.008856
    kappa = 903.3
    f = torch.where(
        xyz > eps,
        xyz ** (1.0 / 3.0),
        (kappa * xyz + 16.0) / 116.0
    )

    L = (116.0 * f[..., 1] - 16.0)
    A = 500.0 * (f[..., 0] - f[..., 1])
    B = 200.0 * (f[..., 1] - f[..., 2])

    return torch.stack([L, A, B], dim=-1)


# ══════════════════════════════════════════════════════════════════
#  GPU MORPHOLOGY
# ══════════════════════════════════════════════════════════════════

def morph_dilate_gpu(mask: torch.Tensor, kernel_px: int) -> torch.Tensor:
    pad = kernel_px // 2
    return F.max_pool2d(mask, kernel_size=kernel_px, stride=1, padding=pad)


def morph_erode_gpu(mask: torch.Tensor, kernel_px: int) -> torch.Tensor:
    return 1.0 - morph_dilate_gpu(1.0 - mask, kernel_px)


def morph_close_gpu(mask: torch.Tensor, kernel_px: int) -> torch.Tensor:
    return morph_erode_gpu(morph_dilate_gpu(mask, kernel_px), kernel_px)


# ══════════════════════════════════════════════════════════════════
#  GPU GAUSSIAN FEATHER
# ══════════════════════════════════════════════════════════════════

def make_gaussian_kernel(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g1d    = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g2d    = g1d[:, None] * g1d[None, :]
    g2d   /= g2d.sum()
    return g2d[None, None, :, :]

_FEATHER_K    = FEATHER_PX if FEATHER_PX % 2 == 1 else FEATHER_PX + 1
_GAUSS_KERNEL = make_gaussian_kernel(_FEATHER_K, _FEATHER_K / 6.0, DEVICE)


def feather_gpu(mask: torch.Tensor) -> torch.Tensor:
    global _GAUSS_KERNEL
    if _GAUSS_KERNEL.device != mask.device:
        _GAUSS_KERNEL = _GAUSS_KERNEL.to(mask.device)
    pad = _FEATHER_K // 2
    return F.conv2d(mask, _GAUSS_KERNEL, padding=pad)


# ══════════════════════════════════════════════════════════════════
#  GPU MASK COMPUTATION
# ══════════════════════════════════════════════════════════════════

def compute_masks_gpu(ref_batch: torch.Tensor,
                      manip_batch: torch.Tensor) -> torch.Tensor:
    if manip_batch.shape != ref_batch.shape:
        N, H, W, C = ref_batch.shape
        manip_nhwc = manip_batch.permute(0, 3, 1, 2)
        manip_nhwc = F.interpolate(manip_nhwc, size=(H, W), mode="bilinear",
                                   align_corners=False)
        manip_batch = manip_nhwc.permute(0, 2, 3, 1)

    ref_lab   = bgr_to_lab_gpu(ref_batch)
    manip_lab = bgr_to_lab_gpu(manip_batch)

    diff = torch.abs(ref_lab - manip_lab).mean(dim=-1)
    mask = (diff > DIFF_THRESHOLD).float().unsqueeze(1)
    mask = morph_close_gpu(mask, MORPH_CLOSE_PX)
    mask = morph_dilate_gpu(mask, DILATE_PX)
    mask = morph_dilate_gpu(mask, DILATE_PX)
    mask = feather_gpu(mask)
    mask = mask.clamp(0, 1)

    return mask.permute(0, 2, 3, 1)


# ══════════════════════════════════════════════════════════════════
#  GPU BLEND
# ══════════════════════════════════════════════════════════════════

def blend_gpu(base: torch.Tensor, edit: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    alpha  = compute_masks_gpu(ref, edit)
    result = edit * alpha + base * (1.0 - alpha)
    return result.clamp(0, 255)


# ══════════════════════════════════════════════════════════════════
#  GPU CHAIN
# ══════════════════════════════════════════════════════════════════

def chain_permutation_gpu(orig_frames: list, edit_frames: dict, perm: tuple,
                           batch_size: int, log: Logger) -> list:
    n       = min(len(orig_frames), *(len(edit_frames[k]) for k in perm))
    results = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        orig_batch = frames_to_gpu(orig_frames, start, end)
        edit_batches = {k: frames_to_gpu(edit_frames[k], start, end) for k in perm}

        current = orig_batch.clone()
        for step in perm:
            ref     = current          # no clone needed — blend_gpu only reads ref
            current = blend_gpu(current, edit_batches[step], ref)

        results.extend(tensor_to_frames(current))

        del orig_batch, edit_batches, current
        torch.cuda.empty_cache()

        log.progress(end, n, label="frames: ")

    print()
    return results


# ══════════════════════════════════════════════════════════════════
#  MASK COVERAGE STATS (GPU)
# ══════════════════════════════════════════════════════════════════

def mask_coverage_stats_gpu(orig_frames: list, manip_frames: list, sample: int = 30) -> dict:
    n       = min(len(orig_frames), len(manip_frames), sample)
    orig_b  = frames_to_gpu(orig_frames,  0, n)
    manip_b = frames_to_gpu(manip_frames, 0, n)
    masks   = compute_masks_gpu(orig_b, manip_b)
    pcts    = (masks > 0.5).float().mean(dim=(1, 2, 3)) * 100
    pcts_cpu = pcts.cpu().numpy()
    del orig_b, manip_b, masks
    torch.cuda.empty_cache()
    return {
        "mean_pct": round(float(pcts_cpu.mean()), 2),
        "min_pct":  round(float(pcts_cpu.min()),  2),
        "max_pct":  round(float(pcts_cpu.max()),  2),
        "sample":   n,
    }


# ══════════════════════════════════════════════════════════════════
#  LABELS + CSV
# ══════════════════════════════════════════════════════════════════

def save_label(video_id: str, perm: tuple, num_frames: int, fps: float,
               width: int, height: int, cov_stats: dict):
    perm_name  = "_".join(perm)
    label_path = LABEL_DIR / f"{video_id}_{perm_name}.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)

    label = {
        "video_id":      f"{video_id}_{perm_name}",
        "source_video":  f"{video_id}.mp4",
        "edit_sequence": [EDIT_REGISTRY[e]["name"] for e in perm],
        "edit_keys":     list(perm),
        "edit_count":    len(perm),
        "num_frames":    num_frames,
        "fps":           fps,
        "resolution":    f"{width}x{height}",
        "encoding":      {"codec": "libx264", "crf": CRF, "pix_fmt": PIX_FMT},
        "blending": {
            "method":         "LAB_diff_feathered_alpha_GPU",
            "compute":        f"PyTorch CUDA — {GPU_NAME}",
            "mask_anchor":    "previous_step",
            "diff_threshold": DIFF_THRESHOLD,
            "morph_close_px": MORPH_CLOSE_PX,
            "dilate_px":      DILATE_PX,
            "feather_px":     FEATHER_PX,
        },
        "mask_coverage": cov_stats,
        "created_at":    datetime.now().isoformat(),
    }
    with open(label_path, "w") as f:
        json.dump(label, f, indent=2)
    return label_path


def append_csv(csv_path: Path, rows: list):
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "video_name", "source_video", "sequence", "num_frames"
        ])
        if not exists:
            w.writeheader()
        w.writerows(rows)


# ══════════════════════════════════════════════════════════════════
#  CORE — process ALL 60 sequences for one source video
# ══════════════════════════════════════════════════════════════════

def process_video_all_sequences(video_id: str, batch_size: int,
                                preview_only: bool, log: Logger) -> dict:
    """
    Generates all 60 ordered sequences for ONE source video.

    FIX vs old process_video(): loads the original + ALL 5 manipulation
    videos ONCE (not per-sequence — that would mean reading from disk
    60 times for the same 5 files), then reuses those frames across all
    60 chain_permutation_gpu() calls. Each sequence's success/failure is
    tracked independently, so one failing sequence doesn't affect the
    other 59.

    Returns a dict: {"succeeded": [...], "failed": [(seq, reason), ...]}
    """
    log.section(f"Video: {video_id}  [60-sequence balanced mode]")
    t0 = time.time()

    result = {"succeeded": [], "failed": []}

    # Validate ALL 5 methods exist for this source up front — every
    # sequence uses 3-of-5, so we need all 5 loaded once anyway.
    ok, missing = validate_inputs(video_id, ALL_EDIT_KEYS, log)
    if not ok:
        for seq in ALL_60_SEQUENCES:
            result["failed"].append((seq, f"source-level missing files: {missing}"))
        return result

    info = VIDEO_MAP[video_id]

    log.info("Loading original + all 5 manipulation videos into RAM (once)...")
    try:
        orig_frames, fps, W, H = load_video(info["original"], log)
        edit_frames = {}
        for key in ALL_EDIT_KEYS:
            edit_frames[key], _, _, _ = load_video(info[key], log)
    except Exception as e:
        log.error(f"  Failed to load source-level frames for {video_id}: {e}")
        for seq in ALL_60_SEQUENCES:
            result["failed"].append((seq, f"frame load failure: {e}"))
        return result

    n = min(len(orig_frames), *(len(edit_frames[k]) for k in ALL_EDIT_KEYS))
    log.info(f"  {W}x{H}  @  {fps:.2f} fps  |  frames usable across all methods: {n}")
    log.info(f"  GPU: {GPU_NAME}  |  VRAM: {VRAM_GB:.1f} GB  |  batch: {batch_size} frames")

    log.info("\nComputing mask coverage (GPU) for all 5 methods...")
    cov_stats = {}
    for key in ALL_EDIT_KEYS:
        try:
            cov_stats[key] = mask_coverage_stats_gpu(orig_frames, edit_frames[key])
            s = cov_stats[key]
            log.info(f"  {key}: mean={s['mean_pct']}%  min={s['min_pct']}%  max={s['max_pct']}%")
        except Exception as e:
            log.warn(f"  Mask coverage stats failed for {key}: {e} (continuing anyway)")
            cov_stats[key] = {"mean_pct": None, "min_pct": None, "max_pct": None, "sample": 0}

    if preview_only:
        log.info("Preview-only — no videos written.")
        for seq in ALL_60_SEQUENCES:
            result["succeeded"].append(seq)
        return result

    csv_path = LABEL_DIR / "_metadata_index.csv"
    csv_rows = []

    log.info(f"\nGenerating all {len(ALL_60_SEQUENCES)} sequences for {video_id} on GPU...")

    for idx, perm in enumerate(ALL_60_SEQUENCES, 1):
        perm_name = "_".join(perm)
        out_path  = OUT_DIR / f"{video_id}_{perm_name}.mp4"

        if out_path.exists():
            log.info(f"  [{idx}/60] {perm_name}  [SKIP] already exists")
            result["succeeded"].append(perm)
            continue

        log.info(f"  [{idx}/60]  {perm_name}  ({' -> '.join(EDIT_REGISTRY[e]['name'] for e in perm)})")

        try:
            chained = chain_permutation_gpu(orig_frames, edit_frames, perm, batch_size, log)
            write_video_ffmpeg(chained, out_path, fps, W, H, log)

            lbl = save_label(video_id, perm, len(chained), fps, W, H, cov_stats)
            log.ok(f"  Label -> {lbl.name}")

            csv_rows.append({
                "video_name":   out_path.name,
                "source_video": f"{video_id}.mp4",
                "sequence":     "->".join(EDIT_REGISTRY[e]["name"] for e in perm),
                "num_frames":   len(chained),
            })
            result["succeeded"].append(perm)

        except Exception as e:
            log.error(f"  FAILED sequence {perm_name}: {e}")
            log.error(traceback.format_exc())
            torch.cuda.empty_cache()
            result["failed"].append((perm, str(e)))
            continue

    if csv_rows:
        append_csv(csv_path, csv_rows)

    log.info(f"\n  {video_id}: {len(result['succeeded'])}/60 succeeded, "
             f"{len(result['failed'])}/60 failed, done in {time.time()-t0:.1f}s")

    # Free this source's frames before moving to the next video — with
    # 1000 sources x 5 manip videos each held only one-at-a-time, this
    # keeps RAM bounded regardless of dataset size.
    del orig_frames, edit_frames

    return result


# ══════════════════════════════════════════════════════════════════
#  BATCH
# ══════════════════════════════════════════════════════════════════

def process_all(video_ids: list, batch_size: int, preview_only: bool, log: Logger):
    log.info(f"Total source videos discovered:  {len(VIDEO_MAP)}")
    log.info(f"Source videos to process:        {len(video_ids)}")
    log.info(f"Edit methods:                    {ALL_EDIT_KEYS}")
    log.info(f"Sequences per source video:       60 (all of P(5,3))")
    log.info(f"Expected total output videos:    {len(video_ids) * 60}")

    if VIDEO_MAP_SKIP_REPORT:
        log.warn(f"NOTE: {len(VIDEO_MAP_SKIP_REPORT)} source videos were already excluded "
                 f"at discovery time (see skip report above/in this log) — these are NOT "
                 f"counted in 'source videos to process' above, so the expected total "
                 f"already accounts for them.")

    all_succeeded = {}   # video_id -> list of succeeded sequences
    all_failed = {}       # video_id -> list of (sequence, reason)

    for i, vid in enumerate(video_ids, 1):
        log.info(f"\n[{i}/{len(video_ids)}] Processing source video {vid}...")
        result = process_video_all_sequences(vid, batch_size, preview_only, log)
        all_succeeded[vid] = result["succeeded"]
        all_failed[vid] = result["failed"]

    total_succeeded = sum(len(v) for v in all_succeeded.values())
    total_failed = sum(len(v) for v in all_failed.values())
    total_expected = len(video_ids) * 60

    summary = {
        "created_at":          datetime.now().isoformat(),
        "gpu":                 GPU_NAME,
        "vram_gb":             round(VRAM_GB, 1),
        "source_videos_processed": len(video_ids),
        "source_videos_skipped_at_discovery": len(VIDEO_MAP_SKIP_REPORT),
        "sequences_per_source": 60,
        "expected_total_outputs": total_expected,
        "actual_succeeded":     total_succeeded,
        "actual_failed":        total_failed,
        "manipulation_types":  {k: EDIT_REGISTRY[k]["name"] for k in EDIT_REGISTRY},
        "encoding":            {"codec": "libx264", "crf": CRF, "pix_fmt": PIX_FMT},
        "blending": {
            "method":      "LAB_diff_feathered_alpha_GPU",
            "mask_anchor": "previous_step",
            "compute":     f"PyTorch CUDA — {GPU_NAME}",
        },
    }
    sp = LABEL_DIR / "_dataset_summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    log.ok(f"Dataset summary -> {sp.name}")

    log.section("PIPELINE COMPLETE")
    log.info(f"  Expected: {total_expected}  |  Succeeded: {total_succeeded}  |  Failed: {total_failed}")

    if total_failed > 0:
        log.warn(f"\n  {total_failed} sequences failed. Detailed list:")
        for vid, failures in all_failed.items():
            for seq, reason in failures:
                seq_name = "_".join(seq)
                log.warn(f"    {vid}_{seq_name}: {reason}")

    return all_succeeded, all_failed


# ══════════════════════════════════════════════════════════════════
#  RECONCILIATION — verify expected vs actual output count exactly
# ══════════════════════════════════════════════════════════════════

def reconcile(video_ids: list, log: Logger):
    """
    NEW: scans OUT_DIR for actual .mp4 files and compares against the
    full expected set (every video_id x every one of the 60 sequences).
    Prints the EXACT list of missing (video_id, sequence) combinations,
    so a shortfall is immediately actionable instead of just a smaller-
    than-expected number in a CSV.
    """
    log.section("RECONCILIATION CHECK")

    expected = set()
    for vid in video_ids:
        for perm in ALL_60_SEQUENCES:
            expected.add(f"{vid}_{'_'.join(perm)}.mp4")

    actual = set(p.name for p in OUT_DIR.glob("*.mp4"))

    missing = expected - actual
    unexpected = actual - expected   # files present that don't match any expected name —
                                      # worth flagging too, could be stale runs or naming drift

    log.info(f"  Expected output videos: {len(expected)}")
    log.info(f"  Actual output videos found in {OUT_DIR}: {len(actual)}")
    log.info(f"  Missing: {len(missing)}")
    log.info(f"  Unexpected (present but not in expected set): {len(unexpected)}")

    if missing:
        log.warn(f"\n  Exact list of missing videos ({len(missing)}):")
        for name in sorted(missing):
            log.warn(f"    {name}")
        log.warn(f"\n  To regenerate just the missing ones, re-run this script normally —")
        log.warn(f"  existing files are skipped automatically, so only gaps get filled in.")
    else:
        log.ok("  No missing videos. Dataset generation is complete and accounted for.")

    if unexpected:
        log.warn(f"\n  Unexpected files present (first 20 shown):")
        for name in sorted(unexpected)[:20]:
            log.warn(f"    {name}")

    return missing, unexpected


# ══════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Video Forensics GPU Pipeline — 60-sequence balanced edition"
    )
    parser.add_argument("--video",      type=str, default=None,
                        help="Single source video ID e.g. --video 000")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Frames per GPU batch (default 32).")
    parser.add_argument("--preview",    action="store_true",
                        help="Mask stats only — no video output")
    parser.add_argument("--reconcile-only", action="store_true",
                        help="Skip generation entirely — just check what's missing "
                             "against what's already on disk.")
    args = parser.parse_args()

    for d in [OUT_DIR, LABEL_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = Logger(LOG_DIR / f"run_{ts}.log")

    log.section("VIDEO FORENSICS — GPU PIPELINE  (60-SEQUENCE BALANCED EDITION)")
    log.info(f"GPU:          {GPU_NAME}")
    log.info(f"VRAM:         {VRAM_GB:.1f} GB")
    log.info(f"Batch size:   {args.batch_size} frames")
    log.info(f"Root:         {ROOT}")
    log.info(f"Preview only: {args.preview}")
    log.info(f"Source videos found complete: {len(VIDEO_MAP)}")

    log.info("\nVerifying 60-sequence balance guarantee...")
    verify_60_sequence_balance()
    log.ok("Balance check passed: every method appears 36/60 overall, 12/60 at each position.")

    if args.video:
        if args.video not in VIDEO_MAP:
            log.error(f"Unknown video ID '{args.video}'. "
                      f"Not found in discovered VIDEO_MAP ({len(VIDEO_MAP)} entries).")
            log.close()
            sys.exit(1)
        video_ids = [args.video]
    else:
        video_ids = sorted(VIDEO_MAP.keys())

    if args.reconcile_only:
        reconcile(video_ids, log)
        log.close()
        return

    process_all(video_ids, args.batch_size, args.preview, log)

    # Always run reconciliation automatically at the end of a real run —
    # this is what tells you definitively whether you actually got all
    # 60,000 videos, instead of finding out later by counting CSV rows.
    reconcile(video_ids, log)

    log.close()


if __name__ == "__main__":
    main()
