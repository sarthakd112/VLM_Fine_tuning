"""
inference.py
============
PSEUDO-CODE — Inference Pipeline for Deployed Long Traffic Video VLM.

Handles the full flow from a raw video file + user query to a
structured natural-language response, including segment-level
processing and temporal memory reconstruction.
"""

from typing import Dict

from pipeline_config import PipelineConfig
from video_segmentation import segment_video
from temporal_memory import TemporalMemoryBank


def run_inference(
    model,
    video_path: str,
    user_query: str,
    cfg: PipelineConfig,
) -> Dict[str, str]:
    """
    End-to-end inference for a single query on a long traffic video.

    Steps
    -----
    1. Segment the input video using the same adaptive strategy
       used during training.
    2. Initialise a fresh TemporalMemoryBank.
    3. Process each segment sequentially:
       a. Encode key-frames through the vision encoder.
       b. Ingest into the memory bank (compress + update global memory).
    4. Build the final context from the memory bank.
    5. Tokenise the user query and prepend a task-routing system prompt.
    6. Feed [system_prompt | context | user_query] into the language
       decoder with beam search (beam_width=4).
    7. Post-process: extract timestamps, event labels, and the
       narrative response.

    Returns
    -------
    {
        "response": str,                  # natural-language answer
        "detected_events": List[dict],    # [{type, start_s, end_s, severity}]
        "confidence": float,              # model's self-reported confidence
    }
    """
    # --- 1. Segment ---
    segments = segment_video(video_path, cfg.segmentation)

    # --- 2. Memory bank ---
    memory = TemporalMemoryBank(cfg.temporal_memory)

    # --- 3. Encode & ingest ---
    for seg in segments:
        frames = load_frames(video_path, seg.keyframe_indices)
        vision_tokens = model.vision_encoder(frames).flatten(0, 1)
        seg.keyframe_embeddings = vision_tokens
        memory.ingest_segment(seg.segment_id, vision_tokens,
                              (seg.start_time_s, seg.end_time_s))

    # --- 4. Context ---
    current_tokens = segments[-1].keyframe_embeddings
    context = memory.build_context(current_tokens)

    # --- 5. System prompt + query ---
    system_prompt = (
        "You are a traffic surveillance analysis assistant.  "
        "Answer the following question about the video using precise "
        "timestamps, event descriptions, and severity assessments.  "
        "If unsure, say so rather than guessing."
    )
    query_tokens = model.tokenizer.encode(
        f"[SYSTEM] {system_prompt}\n[USER] {user_query}"
    )

    # --- 6. Decode ---
    input_embeds = concatenate([context, query_tokens], dim=0)
    raw_output = model.language_decoder.generate(
        input_embeds,
        max_new_tokens=512,
        beam_width=4,
        temperature=0.3,
        repetition_penalty=1.15,
    )
    response_text = model.tokenizer.decode(raw_output)

    # --- 7. Post-process ---
    events = extract_structured_events(response_text)
    confidence = extract_confidence_score(response_text)

    return {
        "response": response_text,
        "detected_events": events,
        "confidence": confidence,
    }


# ═══════════════════════════════════════════════════════════════════════
# Helper: structured event extraction from free-text
# ═══════════════════════════════════════════════════════════════════════

def extract_structured_events(text: str):
    """
    Regex + heuristic parser that pulls structured events from the
    model's free-text output.

    Expected patterns:
      "congestion from 02:15 to 05:30 (severity: high)"
      "red-light violation at 01:42"

    Returns a list of dicts with keys:
      type, start_s, end_s (optional), severity
    """
    events = []
    for match in regex_findall(EVENT_PATTERN, text):
        events.append({
            "type": match.group("event_type"),
            "start_s": timestamp_to_seconds(match.group("start")),
            "end_s": timestamp_to_seconds(match.group("end")) if match.group("end") else None,
            "severity": match.group("severity") or "unknown",
        })
    return events
