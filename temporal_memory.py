"""
temporal_memory.py
==================
PSEUDO-CODE — Hierarchical Temporal Memory for Cross-Segment Reasoning.

Long traffic videos (10–60 min) produce hundreds of segments.  A VLM
cannot attend to all of them simultaneously.  This module maintains a
compressed, hierarchical memory that lets the model reason across the
full timeline without exceeding context-length limits.

Architecture
------------
Layer 0 (raw)       : per-segment key-frame token sequences
Layer 1 (summaries) : compressed segment summaries (fixed token budget)
Layer 2 (global)    : rolling global memory updated via gated write

At inference time the model sees:
    [global_memory] [recent_window_segments] [current_segment]
"""

from dataclasses import dataclass, field
from typing import List

from pipeline_config import TemporalMemoryConfig


@dataclass
class MemorySlot:
    slot_id: int
    token_embedding: "Tensor"     # shape (D,)
    relevance_score: float = 0.0  # updated by attention gate


@dataclass
class SegmentSummary:
    segment_id: int
    summary_tokens: "Tensor"      # shape (T_summary, D)
    timestamp_range: tuple         # (start_s, end_s)


class TemporalMemoryBank:
    """
    Maintains a three-level memory hierarchy and exposes a
    `build_context(...)` method that the VLM decoder calls at each
    generation step.
    """

    def __init__(self, cfg: TemporalMemoryConfig):
        self.cfg = cfg
        self.segment_summaries: List[SegmentSummary] = []
        self.global_memory: List[MemorySlot] = [
            MemorySlot(slot_id=i, token_embedding=zeros(D))
            for i in range(cfg.external_memory_slots)
        ]

    # ──────────────────────────────────────────────────────────────
    # STEP 1  Compress a raw segment into a fixed-length summary
    # ──────────────────────────────────────────────────────────────
    def compress_segment(self, segment_tokens: "Tensor") -> "Tensor":
        """
        Use a lightweight cross-attention pooler to compress the full
        segment token sequence (possibly 1 000+ tokens) into a
        fixed-budget summary of *summary_token_budget* tokens.

        Architecture:
            learnable_queries (T_summary, D) attend over segment_tokens
            → output (T_summary, D)

        This is analogous to a Perceiver-style resampler and is trained
        jointly with the rest of the model.
        """
        queries = learnable_summary_queries(self.cfg.summary_token_budget)
        summary = cross_attention(
            query=queries,
            key=segment_tokens,
            value=segment_tokens,
        )
        return summary  # (T_summary, D)

    # ──────────────────────────────────────────────────────────────
    # STEP 2  Update the global memory with the new summary
    # ──────────────────────────────────────────────────────────────
    def update_global_memory(self, summary: SegmentSummary):
        """
        Gated write inspired by Neural Turing Machines:

        1. Compute attention scores between the new summary and each
           existing memory slot.
        2. The slot with the *lowest* relevance is overwritten (LRU).
        3. All relevance scores are decayed by a factor to favour
           recent information.

        This ensures that globally important events (major accidents,
        prolonged congestion) persist while routine frames are evicted.
        """
        attn_scores = [
            dot(summary.summary_tokens.mean(0), slot.token_embedding)
            for slot in self.global_memory
        ]

        # find least relevant slot
        victim_idx = argmin([s.relevance_score for s in self.global_memory])

        # gated write: blend old content with new summary
        gate = sigmoid(linear(summary.summary_tokens.mean(0)))
        self.global_memory[victim_idx].token_embedding = (
            gate * summary.summary_tokens.mean(0)
            + (1 - gate) * self.global_memory[victim_idx].token_embedding
        )
        self.global_memory[victim_idx].relevance_score = max(attn_scores)

        # decay all slots
        for slot in self.global_memory:
            slot.relevance_score *= (1 - self.cfg.memory_compression_ratio)

    # ──────────────────────────────────────────────────────────────
    # STEP 3  Ingest a new segment into the memory hierarchy
    # ──────────────────────────────────────────────────────────────
    def ingest_segment(self, segment_id: int, segment_tokens: "Tensor",
                       time_range: tuple):
        """
        Called once per segment in chronological order.

        1. Compress raw tokens → summary.
        2. Append to the segment summary list.
        3. Update global memory.
        """
        compressed = self.compress_segment(segment_tokens)
        summary = SegmentSummary(
            segment_id=segment_id,
            summary_tokens=compressed,
            timestamp_range=time_range,
        )
        self.segment_summaries.append(summary)
        self.update_global_memory(summary)

    # ──────────────────────────────────────────────────────────────
    # STEP 4  Build the context window for the VLM decoder
    # ──────────────────────────────────────────────────────────────
    def build_context(self, current_segment_tokens: "Tensor") -> "Tensor":
        """
        Assemble the three-part context tensor:

            [GLOBAL_MEM | RECENT_WINDOW | CURRENT_SEGMENT]

        • GLOBAL_MEM      — external memory slots (128 × D)
        • RECENT_WINDOW    — last W segment summaries concatenated
        • CURRENT_SEGMENT  — full-resolution tokens for the active segment

        This fits within a 4 096–8 192 token context while giving the
        model access to the entire video history.
        """
        # global memory tokens
        global_tokens = stack([s.token_embedding for s in self.global_memory])

        # sliding window of recent summaries
        W = self.cfg.sliding_window_segments
        recent = self.segment_summaries[-W:]
        recent_tokens = concatenate([s.summary_tokens for s in recent])

        context = concatenate([
            global_tokens,
            recent_tokens,
            current_segment_tokens,
        ], dim=0)

        return context
