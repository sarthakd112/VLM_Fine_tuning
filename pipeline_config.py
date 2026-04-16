"""
pipeline_config.py
==================
Configuration schema for the Long Traffic Video VLM Fine-Tuning Pipeline.
All hyperparameters, paths, and architectural knobs are centralised here.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VideoSegmentationConfig:
    """Controls how raw long videos are split into digestible segments."""
    uniform_segment_seconds: int = 30
    overlap_seconds: int = 5
    scene_change_threshold: float = 0.35
    max_frames_per_segment: int = 64
    keyframe_sampling_fps: float = 2.0


@dataclass
class TemporalMemoryConfig:
    """Hierarchical memory architecture for cross-segment reasoning."""
    sliding_window_segments: int = 4
    summary_token_budget: int = 256
    external_memory_slots: int = 128
    memory_compression_ratio: float = 0.25
    use_hierarchical_summarisation: bool = True


@dataclass
class LoRAConfig:
    """Parameter-efficient fine-tuning via Low-Rank Adaptation."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "v_proj",           # attention projections
            "temporal_cross_attn",        # custom temporal layer
        ]
    )


@dataclass
class CurriculumConfig:
    """Curriculum learning schedule: short clips -> full-length videos."""
    stages: List[dict] = field(default_factory=lambda: [
        {"name": "warm-up",    "max_duration_s": 60,   "epochs": 3, "lr": 2e-4},
        {"name": "medium",     "max_duration_s": 300,  "epochs": 5, "lr": 1e-4},
        {"name": "long-form",  "max_duration_s": 1800, "epochs": 8, "lr": 5e-5},
    ])


@dataclass
class TrainingConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_epochs: int = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    fp16: bool = True
    grad_checkpoint: bool = True
    eval_every_n_steps: int = 500


@dataclass
class LossConfig:
    """Multi-task loss weights."""
    caption_loss_weight: float = 1.0
    temporal_grounding_loss_weight: float = 0.8
    classification_loss_weight: float = 0.6
    hallucination_penalty_weight: float = 0.3


@dataclass
class DatasetConfig:
    train_annotation_path: str = "data/annotations/train.jsonl"
    val_annotation_path: str = "data/annotations/val.jsonl"
    video_root: str = "data/videos/"
    max_video_duration_s: int = 1800
    instruction_templates_path: str = "data/instruction_templates.json"


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "caption_bleu4",
        "caption_cider",
        "temporal_grounding_iou",
        "event_classification_f1",
        "hallucination_rate",
    ])
    human_eval_sample_size: int = 200
    temporal_iou_thresholds: List[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.7]
    )


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating every sub-config."""
    model_name: str = "Qwen3-VL-32B"
    segmentation: VideoSegmentationConfig = field(default_factory=VideoSegmentationConfig)
    temporal_memory: TemporalMemoryConfig = field(default_factory=TemporalMemoryConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "checkpoints/"
    seed: int = 42
