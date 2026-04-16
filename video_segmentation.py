"""
video_segmentation.py
=====================
PSEUDO-CODE — Adaptive Video Segmentation for Long Traffic Videos.

Converts a single long traffic surveillance video into a sequence of
overlapping segments, each containing a bounded number of key-frames
suitable for VLM input.

Two strategies are combined:
  1. Uniform temporal slicing (fixed interval)
  2. Scene-change detection (content-adaptive boundaries)

The merge step reconciles the two boundary lists so that no segment
exceeds the token budget of the downstream vision encoder.
"""

from dataclasses import dataclass
from typing import List, Tuple

from pipeline_config import VideoSegmentationConfig


@dataclass
class VideoSegment:
    segment_id: int
    start_time_s: float
    end_time_s: float
    keyframe_indices: List[int]
    keyframe_embeddings: List["Tensor"]  # populated later by the vision encoder


# ──────────────────────────────────────────────────────────────────────
# STEP 1  Load the raw video and extract a dense frame sequence
# ──────────────────────────────────────────────────────────────────────
def load_video(video_path: str, fps: float) -> "FrameSequence":
    """
    Decode the video at *fps* frames-per-second.
    For a 30-min traffic video at 2 fps this yields ~3 600 frames.
    Returns a lightweight handle (memory-mapped) to avoid OOM.
    """
    raw_frames = VideoReader(video_path).decode(fps=fps)  # lazy iterator
    return raw_frames


# ──────────────────────────────────────────────────────────────────────
# STEP 2  Detect scene-change boundaries
# ──────────────────────────────────────────────────────────────────────
def detect_scene_boundaries(
    frames: "FrameSequence",
    threshold: float,
) -> List[float]:
    """
    Compute pairwise cosine similarity of consecutive frame embeddings.
    A boundary is inserted whenever similarity drops below *threshold*.

    In traffic video this captures:
      • Camera angle switches (multi-camera setups)
      • Sudden lighting changes (tunnel entry/exit)
      • Scene transitions in edited dashcam compilations
    """
    boundaries = [0.0]
    for i in range(1, len(frames)):
        sim = cosine_similarity(
            lightweight_encoder(frames[i - 1]),
            lightweight_encoder(frames[i]),
        )
        if sim < threshold:
            boundaries.append(timestamp_of(frames[i]))
    boundaries.append(total_duration(frames))
    return boundaries


# ──────────────────────────────────────────────────────────────────────
# STEP 3  Merge uniform + scene-change boundaries
# ──────────────────────────────────────────────────────────────────────
def build_segment_boundaries(
    total_duration_s: float,
    cfg: VideoSegmentationConfig,
    scene_boundaries: List[float],
) -> List[Tuple[float, float]]:
    """
    Produce the final list of (start, end) intervals.

    Algorithm
    ---------
    1. Generate uniform cuts every *uniform_segment_seconds*.
    2. Union with scene-change cuts.
    3. Sort and de-duplicate (merge cuts < 2 s apart).
    4. Walk through the merged list and create overlapping windows
       controlled by *overlap_seconds*.
    """
    uniform_cuts = list(
        range(0, int(total_duration_s), cfg.uniform_segment_seconds)
    )
    all_cuts = sorted(set(uniform_cuts) | set(scene_boundaries))

    # de-duplicate close cuts
    merged_cuts = [all_cuts[0]]
    for c in all_cuts[1:]:
        if c - merged_cuts[-1] >= 2.0:
            merged_cuts.append(c)

    segments = []
    for i in range(len(merged_cuts) - 1):
        start = max(0, merged_cuts[i] - cfg.overlap_seconds)
        end = min(total_duration_s, merged_cuts[i + 1] + cfg.overlap_seconds)
        segments.append((start, end))

    return segments


# ──────────────────────────────────────────────────────────────────────
# STEP 4  Sample key-frames within each segment
# ──────────────────────────────────────────────────────────────────────
def sample_keyframes(
    frames: "FrameSequence",
    start_s: float,
    end_s: float,
    max_frames: int,
) -> List[int]:
    """
    Within the segment [start_s, end_s], uniformly sample up to
    *max_frames* key-frame indices.  If the segment is very short
    (e.g. 5 s) fewer frames are returned to avoid redundancy.
    """
    candidate_indices = [
        i for i, f in enumerate(frames)
        if start_s <= timestamp_of(f) <= end_s
    ]
    if len(candidate_indices) <= max_frames:
        return candidate_indices

    step = len(candidate_indices) // max_frames
    return candidate_indices[::step][:max_frames]


# ──────────────────────────────────────────────────────────────────────
# STEP 5  Assemble the final segment objects
# ──────────────────────────────────────────────────────────────────────
def segment_video(
    video_path: str,
    cfg: VideoSegmentationConfig,
) -> List[VideoSegment]:
    """
    End-to-end segmentation entry point.

    Returns an ordered list of VideoSegment dataclasses ready for
    encoding by the vision backbone.
    """
    frames = load_video(video_path, fps=cfg.keyframe_sampling_fps)
    scene_bounds = detect_scene_boundaries(frames, cfg.scene_change_threshold)
    intervals = build_segment_boundaries(
        total_duration(frames), cfg, scene_bounds
    )

    segments = []
    for idx, (start, end) in enumerate(intervals):
        kf_indices = sample_keyframes(frames, start, end, cfg.max_frames_per_segment)
        segments.append(
            VideoSegment(
                segment_id=idx,
                start_time_s=start,
                end_time_s=end,
                keyframe_indices=kf_indices,
                keyframe_embeddings=[],  # filled during encoding
            )
        )
    return segments
