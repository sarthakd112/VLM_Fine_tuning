"""
dataset_builder.py
==================
PSEUDO-CODE — Dataset Construction & Instruction-Response Pairing
for Long Traffic Video Fine-Tuning.

Three annotation granularities are supported:
  • Frame-level  — object bounding boxes, lane markings, signal states
  • Segment-level — event labels with temporal boundaries
  • Video-level  — holistic summary, congestion score, risk assessment

Each training example is an (instruction, video_context, response) triple
assembled from the multi-level annotations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import random


# ═══════════════════════════════════════════════════════════════════════
# ANNOTATION SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FrameAnnotation:
    """Per-frame labels from human annotators or pre-trained detectors."""
    frame_idx: int
    timestamp_s: float
    objects: List[Dict[str, Any]]
    # Example object:
    # {
    #     "class": "vehicle",
    #     "bbox": [x1, y1, x2, y2],
    #     "attributes": {"type": "truck", "color": "white", "occluded": False},
    #     "lane_id": 2,
    # }
    traffic_signal_state: Optional[str] = None   # "red" | "green" | "yellow"
    weather_condition: Optional[str] = None       # "clear" | "rain" | "fog"
    visibility: Optional[str] = None              # "good" | "moderate" | "poor"


@dataclass
class SegmentAnnotation:
    """Event-level labels spanning a temporal interval."""
    segment_id: int
    start_s: float
    end_s: float
    event_type: str       # "congestion" | "accident" | "violation" | "normal_flow"
    severity: str         # "low" | "medium" | "high" | "critical"
    description: str      # free-text description of the event
    involved_objects: List[int]  # indices into frame-level object list
    causal_chain: Optional[str] = None  # e.g. "truck braked → rear-end collision"


@dataclass
class VideoAnnotation:
    """Holistic video-level labels."""
    video_id: str
    duration_s: float
    location: str                    # "Highway I-95 N, mile 42"
    overall_congestion_score: float  # 0.0 (free-flow) → 1.0 (gridlock)
    summary: str                     # 2-3 sentence narrative
    risk_assessment: str             # "low" | "moderate" | "high"
    segments: List[SegmentAnnotation] = field(default_factory=list)
    frames: List[FrameAnnotation] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# INSTRUCTION TEMPLATES
# ═══════════════════════════════════════════════════════════════════════

INSTRUCTION_TEMPLATES = {
    "temporal_grounding": [
        "At what time does the first traffic violation occur in this video?",
        "Identify the start and end timestamps of the congestion event.",
        "When does the accident happen? Provide the exact time range.",
    ],
    "event_description": [
        "Describe the traffic event occurring between {start_s}s and {end_s}s.",
        "What is happening in the segment from {start_s} to {end_s} seconds?",
        "Explain the sequence of events visible between timestamps {start_s}–{end_s}.",
    ],
    "video_summary": [
        "Provide a comprehensive summary of this traffic surveillance video.",
        "Summarise the key events, congestion patterns, and any incidents in this video.",
    ],
    "counting": [
        "How many vehicles are visible at timestamp {timestamp_s}?",
        "Count the number of red-light violations in this video.",
    ],
    "causal_reasoning": [
        "What caused the traffic jam visible after {start_s} seconds?",
        "Explain the chain of events leading to the accident at {start_s}s.",
    ],
    "risk_assessment": [
        "Assess the overall traffic risk level shown in this video.",
        "Rate the severity of congestion from 1 to 10 and justify your rating.",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# STEP 1  Load raw annotations
# ═══════════════════════════════════════════════════════════════════════
def load_annotations(annotation_path: str) -> List[VideoAnnotation]:
    """
    Parse the JSONL annotation file where each line is a complete
    VideoAnnotation for one video clip.
    """
    annotations = []
    with open(annotation_path) as f:
        for line in f:
            record = json.loads(line)
            annotations.append(VideoAnnotation(**record))
    return annotations


# ═══════════════════════════════════════════════════════════════════════
# STEP 2  Build instruction-response pairs from annotations
# ═══════════════════════════════════════════════════════════════════════
def build_instruction_pairs(
    annotation: VideoAnnotation,
) -> List[Dict[str, str]]:
    """
    For a single video, generate multiple instruction-response training
    examples spanning all task types.

    Returns a list of dicts:
        {"task": ..., "instruction": ..., "response": ..., "video_id": ...}
    """
    pairs = []

    # --- Temporal grounding tasks ---
    for seg in annotation.segments:
        template = random.choice(INSTRUCTION_TEMPLATES["temporal_grounding"])
        instruction = template
        response = (
            f"The {seg.event_type} event occurs between "
            f"{seg.start_s:.1f}s and {seg.end_s:.1f}s.  "
            f"Severity: {seg.severity}.  {seg.description}"
        )
        pairs.append({
            "task": "temporal_grounding",
            "instruction": instruction,
            "response": response,
            "video_id": annotation.video_id,
        })

    # --- Event description tasks ---
    for seg in annotation.segments:
        template = random.choice(INSTRUCTION_TEMPLATES["event_description"])
        instruction = template.format(start_s=seg.start_s, end_s=seg.end_s)
        response = seg.description
        if seg.causal_chain:
            response += f"  Causal chain: {seg.causal_chain}"
        pairs.append({
            "task": "event_description",
            "instruction": instruction,
            "response": response,
            "video_id": annotation.video_id,
        })

    # --- Video summary task ---
    template = random.choice(INSTRUCTION_TEMPLATES["video_summary"])
    pairs.append({
        "task": "video_summary",
        "instruction": template,
        "response": annotation.summary,
        "video_id": annotation.video_id,
    })

    # --- Causal reasoning tasks ---
    for seg in annotation.segments:
        if seg.causal_chain:
            template = random.choice(INSTRUCTION_TEMPLATES["causal_reasoning"])
            instruction = template.format(start_s=seg.start_s)
            response = seg.causal_chain
            pairs.append({
                "task": "causal_reasoning",
                "instruction": instruction,
                "response": response,
                "video_id": annotation.video_id,
            })

    # --- Risk assessment task ---
    template = random.choice(INSTRUCTION_TEMPLATES["risk_assessment"])
    pairs.append({
        "task": "risk_assessment",
        "instruction": template,
        "response": (
            f"Overall congestion score: {annotation.overall_congestion_score:.2f}/1.0.  "
            f"Risk level: {annotation.risk_assessment}.  {annotation.summary}"
        ),
        "video_id": annotation.video_id,
    })

    return pairs


# ═══════════════════════════════════════════════════════════════════════
# STEP 3  Construct the final training dataset
# ═══════════════════════════════════════════════════════════════════════
def build_dataset(annotation_path: str) -> List[Dict[str, str]]:
    """
    Full pipeline: load annotations → pair instructions → shuffle.

    The resulting list is written to a JSONL file consumed by the
    fine-tuning data-loader.
    """
    annotations = load_annotations(annotation_path)
    dataset = []
    for ann in annotations:
        dataset.extend(build_instruction_pairs(ann))

    random.shuffle(dataset)
    return dataset
