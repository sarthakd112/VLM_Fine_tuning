# Fine-Tuning Design for Long Traffic Video Understanding Using Vision-Language Models (VLMs)

**A System Design Proposal**

---

**Date:** 16 April 2026

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [Literature Review & Model Analysis](#2-literature-review--model-analysis)
   - 2.2 [Foundational Models (2023–2024)](#22-foundational-models-2023-2024)
   - 2.3 [Updated Model Landscape (2025–2026)](#23-updated-model-landscape-2025-2026) — GPT-5.4, Gemini 2.5 Pro, Claude 4.6, Qwen3-VL, Gemma 3, InternVL3
   - 2.4 [Gap Analysis](#24-gap-analysis-do-2026-models-eliminate-the-need-for-our-architecture)
   - 2.5 [Model Selection Rationale](#25-model-selection-rationale) — Qwen3-VL-32B
   - 2.6 [Related Work](#26-related-work) — incl. TrafficVILA, SurveillanceVQA-589K
3. [Challenges in Long Video Understanding](#3-challenges-in-long-video-understanding)
4. [Dataset Design](#4-dataset-design)
5. [Fine-Tuning Strategy](#5-fine-tuning-strategy)
6. [Pseudo-Code Pipeline](#6-pseudo-code-pipeline)
7. [Instruction-Following Design](#7-instruction-following-design)
8. [Long-Video Memory Architecture](#8-long-video-memory-architecture)
9. [Evaluation Plan](#9-evaluation-plan)
10. [Failure Cases & Limitations](#10-failure-cases--limitations)
11. [Future Improvements](#11-future-improvements)
12. [Conclusion](#12-conclusion)
13. [References](#13-references)

---

## 1. Introduction & Motivation

### 1.1 Problem Context

Intelligent transportation systems (ITS) increasingly depend on continuous video surveillance to monitor highway corridors, urban intersections, and arterial roads. A single metropolitan traffic management centre may ingest thousands of camera hours daily. Manually reviewing this footage for incidents — accidents, congestion onset, signal violations, illegal lane changes — is infeasible at scale.

Recent Vision-Language Models (VLMs) have demonstrated impressive capabilities in image captioning, visual question answering, and short-clip summarisation. Models such as GPT-4V (OpenAI, 2023), Gemini 1.5 Pro (Google DeepMind, 2024), and Qwen-VL (Alibaba DAMO, 2023) can reason about visual content conditioned on natural-language instructions. However, these models are predominantly designed for **single images or short video clips** (typically < 1 minute), while real-world traffic surveillance produces **continuous streams lasting 10–60 minutes or longer**.

### 1.2 Motivation

This proposal presents a comprehensive system design for **fine-tuning a pre-trained VLM to understand long-duration traffic surveillance videos**. The core motivation is threefold:

- **Operational need:** Traffic management operators require automated systems that can summarise hours of footage, identify incidents with precise timestamps, and answer natural-language queries about road conditions.
- **Technical gap:** Current VLMs cannot process long videos end-to-end due to quadratic attention complexity and fixed context-length limits. A specialised fine-tuning and memory architecture is required.
- **Research opportunity:** The intersection of temporal reasoning, domain-specific visual understanding, and parameter-efficient fine-tuning presents rich algorithmic challenges that advance the state of the art.

### 1.3 Scope

This document covers:

- Comparative analysis of candidate VLM backbones
- A multi-level annotation schema for traffic videos
- A parameter-efficient fine-tuning strategy with curriculum learning
- A hierarchical temporal memory architecture for cross-segment reasoning
- Detailed pseudo-code for every pipeline component (see the `vlm_ft/` codebase)
- An evaluation plan combining automatic metrics and human judgement

---

## 2. Literature Review & Model Analysis

### 2.1 Overview of Vision-Language Models

Vision-Language Models unify a visual encoder (typically a Vision Transformer, ViT) with a large language model (LLM) decoder. The visual encoder converts image patches into token embeddings, which are projected into the LLM's embedding space via a learned projection layer. The LLM then generates text conditioned on both visual and textual tokens.

### 2.2 Foundational Models (2023–2024)

The following models represent the state of the art at the time this architecture was first conceived. They establish the baseline capabilities and limitations that motivated the design.

| Feature | GPT-4V (OpenAI, 2023) | Gemini 1.5 Pro (Google, 2024) | Qwen-VL-Chat (Alibaba, 2023) |
|---|---|---|---|
| **Vision encoder** | Proprietary ViT | Proprietary multi-modal encoder | ViT-bigG (open-weight) |
| **LLM backbone** | GPT-4 (~1.8T params est.) | Gemini MoE (~540B active) | Qwen-72B |
| **Max context length** | 128K tokens | 1M tokens (long context) | 32K tokens (extendable) |
| **Native video support** | Frame sampling only | Native video input (temporal tokens) | Frame sampling only |
| **Fine-tuning access** | API-only (no weight access) | API-only (limited tuning) | Full open-weight access |
| **Temporal reasoning** | Weak (no explicit temporal modelling) | Moderate (interleaved temporal tokens) | Weak (frame-level only) |
| **Licence** | Proprietary | Proprietary | Apache 2.0 (open) |
| **Inference cost** | High (API pricing) | High (API pricing) | Moderate (self-hosted) |

### 2.3 Updated Model Landscape (2025–2026)

The VLM landscape has evolved rapidly. Below is a comprehensive comparison of the latest frontier models evaluated for suitability to long traffic video understanding.

#### 2.3.1 GPT-5.4 (OpenAI, March 2026)

| Attribute | Detail |
|---|---|
| **Context window** | 1.05M tokens |
| **Input modalities** | Text + Image only; **video NOT supported** |
| **Vision strength** | Processes images at >10M pixels without compression; excellent for documents, charts, and diagrams |
| **Fine-tuning** | **Not available** for GPT-5.4; vision fine-tuning only supported on older GPT-4o |
| **Licence** | Proprietary (API-only) |
| **Pricing** | $2.50/M input tokens, $15/M output tokens |
| **Verdict for this project** | **Not suitable.** No video input modality and no fine-tuning access make it unusable for traffic video understanding. The 1M-token context is impressive but irrelevant without video support. |

#### 2.3.2 Gemini 2.5 Pro (Google DeepMind, June 2025 GA)

| Attribute | Detail |
|---|---|
| **Context window** | 1M tokens (2M experimental) |
| **Input modalities** | Text + Image + Audio + **Native video** |
| **Video limits** | ~1 hour (no audio), ~45 min (with audio), up to **~6 hours** in low-res mode (66 tokens/frame at 1 fps) |
| **Video benchmark** | 84.7% on VideoMME (state-of-the-art among all models) |
| **Fine-tuning** | API-only; **limited tuning** (no open weights) |
| **Temporal reasoning** | Strong — native temporal token interleaving with timestamp awareness |
| **Licence** | Proprietary (API-only) |
| **Verdict for this project** | **Strongest native long-video capability** among all models. Can process up to 6 hours of video in a single pass. However, **proprietary and not fine-tunable**, which prevents domain-specific adaptation for traffic surveillance. Best used as a **comparison baseline** or for zero-shot evaluation. |

#### 2.3.3 Claude Opus 4.6 / Sonnet 4.6 (Anthropic, 2025–2026)

| Attribute | Detail |
|---|---|
| **Context window** | 1M tokens (no long-context premium) |
| **Input modalities** | Text + Image + PDF only; **video NOT supported** |
| **Vision strength** | Strong image and document understanding; supports up to 100 images per API request |
| **Fine-tuning** | **Not available** |
| **Licence** | Proprietary (API-only) |
| **Verdict for this project** | **Not suitable.** No video input support and no fine-tuning access. While the 1M-token context and strong reasoning are noteworthy, the absence of video modality is a fundamental blocker. |

#### 2.3.4 Qwen3-VL (Alibaba, October 2025) — Recommended

| Attribute | Detail |
|---|---|
| **Context window** | 256K tokens native, expandable to **1M tokens** |
| **Input modalities** | Text + Image + **Native video** with temporal grounding |
| **Model sizes** | 2B / 4B / 8B / 32B (dense); 30B-A3B / 235B-A22B (MoE) |
| **Video capabilities** | Hours-long video understanding, second-level temporal indexing, dynamic FPS sampling |
| **Temporal reasoning** | Strong — interleaved-MRoPE for spatial-temporal modelling; text-based timestamp alignment |
| **Architecture advances** | DeepStack multi-level ViT feature fusion for improved vision-language alignment |
| **Fine-tuning** | **Full LoRA support** with official `qwen-vl-finetune` framework; LoRA rank/alpha/dropout configurable |
| **Licence** | Open-weight (Apache 2.0 equivalent) |
| **Verdict for this project** | **Best candidate.** Surpasses the original Qwen-VL in every dimension: 8× larger context (256K vs. 32K), native temporal grounding, hours-long video support, and production-ready LoRA fine-tuning. Open weights enable full architectural customisation. |

#### 2.3.5 Gemma 3 (Google, March 2025)

| Attribute | Detail |
|---|---|
| **Context window** | 128K tokens |
| **Input modalities** | Text + Image + **Short video** (via SigLIP encoder, 256 tokens/image) |
| **Model sizes** | 1B / 4B / 12B / 27B |
| **Video capabilities** | Limited — processes short clips by frame sampling; not designed for long-form video |
| **Fine-tuning** | Open weights; community fine-tuning support |
| **Licence** | Open (permissive) |
| **Performance** | 1338 Elo on Chatbot Arena; competitive with models 60% larger |
| **Verdict for this project** | **Partial fit.** Suitable for edge deployment or as a lightweight student model for distillation, but its limited video duration and 128K context make it insufficient as the primary backbone for long surveillance footage. |

#### 2.3.6 InternVL3 / InternVL3.5 (Shanghai AI Lab, 2025)

| Attribute | Detail |
|---|---|
| **Context window** | Extended via variable visual position encoding (V2PE) |
| **Input modalities** | Text + Image + Video |
| **Model sizes** | 38B / 78B (InternVL3); up to 241B-A28B MoE (InternVL3.5) |
| **Architecture** | Native multimodal pre-training (joint vision-language from scratch, not adapted from text-only LLM) |
| **Fine-tuning** | Open weights; SFT and MPO training recipes published |
| **Licence** | Open-weight |
| **Benchmark** | 72.2 MMMU (InternVL3-78B); competitive with GPT-4o and Claude 3.5 Sonnet |
| **Verdict for this project** | **Strong alternative.** Native multimodal pre-training is architecturally compelling. However, Qwen3-VL offers superior long-video-specific features (temporal grounding, hours-long support) and a more mature fine-tuning ecosystem. |

#### 2.3.7 Consolidated Comparison (2026)

| Model | Context | Video Input | Max Video Duration | Fine-Tunable | Open Weights | Temporal Grounding |
|---|---|---|---|---|---|---|
| GPT-5.4 | 1.05M | No | N/A | No | No | N/A |
| Gemini 2.5 Pro | 1M–2M | **Yes (native)** | **~6 hours** | Limited | No | Strong |
| Claude 4.6 | 1M | No | N/A | No | No | N/A |
| **Qwen3-VL** | **256K–1M** | **Yes (native)** | **Hours** | **Yes (LoRA)** | **Yes** | **Strong** |
| Gemma 3 | 128K | Short clips | ~minutes | Yes | Yes | Weak |
| InternVL3.5 | Extended | Yes | Moderate | Yes | Yes | Moderate |
| Qwen-VL (original) | 32K | Frame sampling | ~minutes | Yes | Yes | Weak |

### 2.4 Gap Analysis: Do 2026 Models Eliminate the Need for Our Architecture?

A natural question arises: with Gemini 2.5 Pro processing 6 hours of video natively and Qwen3-VL supporting 1M-token contexts, **is the proposed segmentation + memory architecture still necessary?**

The answer is **yes**, for three reasons:

1. **Fine-tuning requirement.** Gemini 2.5 Pro has the longest native video support but is not fine-tunable. Zero-shot performance on domain-specific traffic tasks (violation detection, congestion scoring, causal chain analysis) falls significantly short of fine-tuned models, as demonstrated by the SurveillanceVQA-589K benchmark where general VLMs show large performance gaps on causal and anomaly tasks (Pervaiz et al., 2025).

2. **Context budget at scale.** Even with 1M-token contexts, a 1-hour video at 2 fps with 256 tokens/frame produces ~1.8M tokens — exceeding the limit. The low-resolution mode (66 tokens/frame) sacrifices spatial detail critical for traffic analysis (reading licence plates, detecting small objects, identifying lane markings). Our hierarchical memory provides full resolution on the current segment while compressing history.

3. **Production deployment.** API-only models (GPT-5.4, Gemini, Claude) introduce latency, cost, and data-sovereignty concerns unacceptable for real-time traffic management. A self-hosted, fine-tuned Qwen3-VL with our memory architecture provides deterministic performance at predictable cost.

### 2.5 Model Selection Rationale

**Qwen3-VL-32B** is selected as the base model for this project, superseding the earlier Qwen-VL-Chat-72B recommendation. The rationale:

1. **Open weights with LoRA support:** Full parameter access with an official fine-tuning framework supporting configurable LoRA (default rank 64, alpha 128). Enables injection of custom temporal layers and domain adaptation.
2. **Native long-video understanding:** 256K-token context (expandable to 1M) with built-in dynamic FPS sampling and temporal grounding via interleaved-MRoPE. This natively handles videos up to hours in length.
3. **DeepStack vision encoder:** Multi-level ViT feature fusion produces richer visual representations than the single-level ViT-bigG in the original Qwen-VL, improving detection of small objects and subtle traffic cues.
4. **Temporal grounding built-in:** Text-based timestamp alignment evolved from T-RoPE provides second-level temporal indexing — a capability the original Qwen-VL entirely lacked.
5. **Efficient model sizes:** The 32B dense variant delivers strong performance while requiring only 2×A100 GPUs for LoRA fine-tuning (vs. 4×A100 for the 72B model), reducing infrastructure cost by ~50%.
6. **Cost efficiency:** Self-hosting avoids per-token API costs entirely. At the scale of traffic surveillance (thousands of video-hours per month), this is orders of magnitude cheaper than Gemini or GPT API calls.

For ablation studies and baseline comparisons, Gemini 2.5 Pro is used as a zero-shot reference to quantify the benefit of domain-specific fine-tuning.

### 2.6 Related Work

#### 2.6.1 Video-Language Models

- **Video-LLaVA (Lin et al., 2023):** Extends LLaVA to short video clips by sampling 8 frames. Limited to clips < 30 s; does not address long-form reasoning.
- **LLaVA-Video / LLaVA-NeXT-Video (2024):** Improved video variant using AnyRes frame representation with linear scaling for length generalisation. Available in 7B and 72B sizes. Achieves state-of-the-art among open-source models on VideoMME but does not incorporate explicit temporal memory for hour-long content.
- **LongViViT (Papalampidi et al., 2024):** Hierarchical ViT with temporal aggregation for long videos (up to 5 min). Does not integrate language generation.
- **TimeChat (Ren et al., 2024):** Introduces timestamp-aware tokens for temporal grounding in video QA. Handles up to 3 min; no traffic-domain evaluation.
- **MovieChat (Song et al., 2024):** Sliding-window memory for hour-long movie understanding. Closest to our architecture but operates on narrative films, not surveillance footage.

#### 2.6.2 Traffic-Domain Models and Benchmarks

- **TrafficVILA (Pervaiz et al., 2025):** Purpose-built for traffic surveillance, based on NVILA-15B-HRL with dynamic tiling (six tiles per frame), temporal localisation, and LLM-based fact checking to reduce hallucinations. Ranked top-3 in the 2025 AI City Challenge (score 58.85). Demonstrates that domain-specific adaptation significantly outperforms general-purpose VLMs on traffic tasks.
- **SurveillanceVQA-589K (2025):** The largest video question-answering benchmark for surveillance, containing 589,380 QA pairs across 12 question types including temporal reasoning, causal inference, and anomaly interpretation. Evaluation of eight leading VLMs revealed significant performance gaps, especially on causal and anomaly-related tasks — validating the need for domain-specific fine-tuning.

#### 2.6.3 Memory and Long-Context Architectures

- **MA-LMM (He et al., 2024):** Memory-Augmented Large Multimodal Model that processes videos online (segment by segment) with a long-term memory bank. Designed for arbitrarily long videos; our memory architecture draws on similar principles.
- **Video Panels (2025):** Visual prompting strategy that achieves up to 19.4% accuracy improvements on long-video datasets by restructuring how frames are presented to VLMs.

Our design synthesises ideas from MovieChat's memory mechanism, TimeChat's temporal grounding, TrafficVILA's domain-specific adaptation, and Video-LLaVA's instruction tuning, updated to leverage Qwen3-VL's native temporal capabilities.

---

## 3. Challenges in Long Video Understanding

### 3.1 Token Explosion

A single image processed by ViT-bigG generates approximately 256 patch tokens. At 2 fps, a 30-minute video produces:

```
30 min × 60 s/min × 2 frames/s × 256 tokens/frame = 921,600 tokens
```

This far exceeds the context window of any current LLM. Even Gemini 1.5 Pro's 1M-token limit is consumed by a single video, leaving no room for instructions or generation.

### 3.2 Temporal Reasoning

Traffic events are inherently temporal. Congestion does not appear instantaneously; it builds over minutes as traffic density increases. An accident at timestamp 02:15 may cause a secondary incident at 04:30 due to rubbernecking. The model must:

- Track object persistence across frames (the same truck visible at 01:00 and 01:45)
- Understand causal sequences (brake lights → deceleration → collision)
- Localise events to precise temporal boundaries

### 3.3 Memory Limits

GPU VRAM constrains both the number of tokens in the context and the activation memory during backpropagation. Fine-tuning a 72B-parameter model on long sequences requires:

- **Gradient checkpointing** to trade compute for memory
- **Mixed-precision training** (FP16/BF16) to halve activation size
- **Segment-level processing** with a compact memory representation

### 3.4 Domain-Specific Visual Complexity

Traffic surveillance presents unique visual challenges:

| Challenge | Description | Example |
|---|---|---|
| Small objects at distance | Vehicles occupy < 1% of pixels at far-field cameras | Highway overhead cameras |
| Occlusion | Trucks occlude cars; infrastructure blocks sightlines | Multi-lane intersections |
| Lighting variation | Day/night transitions, tunnel entry/exit, headlight glare | 24-hour continuous feeds |
| Weather degradation | Rain, fog, snow reduce visibility and alter appearance | Winter highway monitoring |
| Camera motion | Wind-induced vibration, PTZ camera panning | Pole-mounted cameras |
| Repetitive backgrounds | Long stretches of identical highway create visual aliasing | Interstate corridors |

### 3.5 Summary of Challenges and Proposed Solutions

| Challenge | Proposed Solution | Section |
|---|---|---|
| Token explosion | Adaptive segmentation + key-frame sampling | §6 |
| Temporal reasoning | Hierarchical memory bank with cross-attention | §8 |
| Memory limits | LoRA + gradient checkpointing + curriculum learning | §5 |
| Domain complexity | Traffic-specific annotations + instruction diversity | §4, §7 |

---

## 4. Dataset Design

### 4.1 Data Sources

The training dataset is constructed from three categories of traffic video:

1. **Public benchmarks:** UA-DETRAC (Wen et al., 2020), AI City Challenge datasets, BDD100K driving video subset.
2. **Synthetic long sequences:** Short clips stitched with synthetic transitions to create 10–30 minute continuous feeds with known ground-truth events.
3. **Proprietary surveillance feeds:** Anonymised footage from partner transportation agencies (subject to IRB approval and data use agreements).

Target dataset size: **~2,000 videos, 500+ hours total**, with full multi-level annotations.

### 4.2 Annotation Granularity

Annotations are provided at three levels, each serving different training tasks:

#### 4.2.1 Frame-Level Annotations

| Field | Type | Description | Example |
|---|---|---|---|
| `frame_idx` | int | Frame index in the decoded sequence | 1204 |
| `timestamp_s` | float | Absolute timestamp in seconds | 602.0 |
| `objects` | list\[dict\] | Detected objects with bounding boxes | `[{"class": "vehicle", "bbox": [120, 80, 210, 160], "attributes": {"type": "sedan", "color": "red"}}]` |
| `traffic_signal_state` | str | Current signal phase | "red" |
| `weather_condition` | str | Observed weather | "rain" |
| `visibility` | str | Visibility quality | "moderate" |

#### 4.2.2 Segment-Level Annotations

| Field | Type | Description | Example |
|---|---|---|---|
| `segment_id` | int | Unique segment identifier | 7 |
| `start_s` | float | Event start time | 135.0 |
| `end_s` | float | Event end time | 210.0 |
| `event_type` | str | Event category | "accident" |
| `severity` | str | Impact level | "high" |
| `description` | str | Free-text narrative | "A white truck rear-ends a sedan in lane 3, causing a three-vehicle pile-up." |
| `involved_objects` | list\[int\] | Indices into frame-level objects | [12, 15, 18] |
| `causal_chain` | str | Cause-effect explanation | "Truck driver distracted → late braking → rear-end collision → lane 2 vehicle swerves" |

#### 4.2.3 Video-Level Annotations

| Field | Type | Description | Example |
|---|---|---|---|
| `video_id` | str | Unique video identifier | "HWY95-N-20260310-0800" |
| `duration_s` | float | Total video length | 1800.0 |
| `location` | str | Camera location | "Highway I-95 N, mile marker 42" |
| `overall_congestion_score` | float | 0.0 (free-flow) to 1.0 (gridlock) | 0.72 |
| `summary` | str | 2-3 sentence narrative | "Morning rush-hour footage showing a gradual build-up of congestion from 05:00. An accident at 12:15 in lane 3 causes a full stop lasting 8 minutes." |
| `risk_assessment` | str | Overall risk level | "high" |

### 4.3 Instruction-Response Pairing

Each annotated video produces multiple training examples across six task types. The pairing logic is implemented in `dataset_builder.py`.

| Task Type | Instruction Example | Response Example |
|---|---|---|
| **Temporal Grounding** | "At what time does the first accident occur?" | "The first accident occurs between 135.0s and 210.0s. A white truck rear-ends a sedan in lane 3." |
| **Event Description** | "Describe what happens between 135s and 210s." | "A distracted truck driver fails to brake in time, rear-ending a red sedan. The impact pushes the sedan into a neighbouring lane, causing a three-vehicle pile-up." |
| **Video Summary** | "Summarise the key events in this video." | "This 30-minute highway surveillance video captures morning rush hour. Congestion builds gradually until an accident at 02:15 causes a full stop lasting 8 minutes…" |
| **Object Counting** | "How many vehicles are visible at 10:02?" | "At timestamp 602s, there are 14 vehicles visible: 9 sedans, 3 trucks, 1 bus, and 1 motorcycle." |
| **Causal Reasoning** | "What caused the traffic jam after 02:15?" | "The congestion was triggered by a three-vehicle accident in lane 3 at 02:15. Rubbernecking in adjacent lanes compounded the delay." |
| **Risk Assessment** | "Rate the traffic risk level in this video." | "Overall congestion score: 0.72/1.0. Risk level: high. The combination of heavy volume and a major accident creates sustained delays and elevated collision risk." |

### 4.4 Dataset Statistics (Target)

| Split | Videos | Hours | Instruction Pairs | Avg. Duration |
|---|---|---|---|---|
| Train | 1,600 | 400 | ~48,000 | 15 min |
| Validation | 200 | 50 | ~6,000 | 15 min |
| Test | 200 | 50 | ~6,000 | 15 min |
| **Total** | **2,000** | **500** | **~60,000** | **15 min** |

---

## 5. Fine-Tuning Strategy

### 5.1 Full Fine-Tuning vs. Parameter-Efficient Tuning

The table below compares tuning strategies for the selected Qwen3-VL-32B backbone. For reference, estimates for the original Qwen-VL-72B are shown in parentheses where they differ significantly.

| Approach | Trainable Params | VRAM (32B model) | Training Time | Risk of Catastrophic Forgetting |
|---|---|---|---|---|
| Full fine-tuning | 32B (100%) | ~160 GB (4×A100) | ~1 week | High |
| LoRA (rank 64) | ~180M (~0.56%) | ~48 GB (2×A100) | ~2 days | Low |
| QLoRA (4-bit + LoRA) | ~180M | ~28 GB (1×A100) | ~3 days | Low |
| Adapter layers only | ~50M (~0.16%) | ~40 GB | ~1.5 days | Very low |

**Selected approach: LoRA (rank 64, alpha 128)** — matching Qwen3-VL's official fine-tuning defaults — with additional fully trainable temporal components. The higher rank (64 vs. 16 in the original design) is justified by the richer temporal adaptation required, and is feasible because the 32B model has a substantially smaller memory footprint than the 72B alternative.

### 5.2 Component Freezing Strategy

The following table details which Qwen3-VL-32B components are frozen, adapted via LoRA, or fully trainable:

| Component | Parameters | Strategy | Rationale |
|---|---|---|---|
| DeepStack vision encoder | ~2B | **FROZEN** | Multi-level ViT features already capture fine-grained spatial detail; pre-trained on diverse visual data |
| Patch embedding layer | ~5M | **FROZEN** | Low-level spatial features transfer well |
| Visual projection MLP | ~100M | **LoRA** (rank 64) | Needs domain adaptation for traffic-specific visual tokens |
| LLM self-attention (Q, V) | ~8B | **LoRA** (rank 64) | Adapts attention patterns for temporal traffic reasoning |
| LLM feed-forward layers | ~16B | **FROZEN** | General language capabilities preserved |
| Interleaved-MRoPE temporal encoding | ~10M | **FROZEN** | Qwen3-VL's native temporal position encoding is already well-trained; freezing preserves timestamp alignment |
| Temporal cross-attention (new) | ~50M | **FULLY TRAINABLE** | New component for cross-segment reasoning; no pre-trained weights |
| Memory compressor (Perceiver) | ~30M | **FULLY TRAINABLE** | New component for segment summarisation; no pre-trained weights |
| LM head | ~100M | **FROZEN** | Vocabulary unchanged |

**Total trainable parameters: ~260M out of 32B (~0.81%)**

*Note:* Compared to the original Qwen-VL-72B design (0.28% trainable), the Qwen3-VL-32B configuration trains a higher fraction of parameters (0.81%) while using fewer total FLOPs, because the base model is less than half the size. The net result is faster training, lower VRAM, and stronger domain adaptation.

### 5.3 Curriculum Learning

Training proceeds in three stages with progressively increasing video duration. This prevents the model from being overwhelmed by long sequences before it has learned basic traffic vocabulary.

```
STAGE 1: Warm-Up (short clips)
├── Max duration: 60 seconds
├── Epochs: 3
├── Learning rate: 2 × 10⁻⁴
├── Focus: Basic traffic vocabulary, object recognition, signal detection
└── Examples: Single-event clips (one accident, one congestion onset)

STAGE 2: Medium (multi-event segments)
├── Max duration: 300 seconds (5 minutes)
├── Epochs: 5
├── Learning rate: 1 × 10⁻⁴
├── Focus: Multi-event sequencing, temporal boundary detection
└── Examples: Clips containing 2-4 sequential events

STAGE 3: Long-Form (full surveillance feeds)
├── Max duration: 1800 seconds (30 minutes)
├── Epochs: 8
├── Learning rate: 5 × 10⁻⁵
├── Focus: Cross-segment reasoning, video-level summarisation
└── Examples: Full-length surveillance recordings
```

At each stage boundary, the learning rate is warm-restarted with cosine annealing. The data-loader filters examples by duration, ensuring the model only sees appropriately long videos at each stage.

### 5.4 Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Batch size (per GPU) | 4 |
| Gradient accumulation steps | 8 |
| Effective batch size | 32 |
| Optimiser | AdamW |
| Weight decay | 0.01 |
| Max gradient norm | 1.0 |
| Precision | FP16 (mixed precision) |
| Gradient checkpointing | Enabled |
| Warmup ratio | 5% of each stage |
| Evaluation frequency | Every 500 steps |

---

## 6. Pseudo-Code Pipeline

> **All pseudo-code is implemented as documented Python modules in the `vlm_ft/` directory.** Each file contains detailed docstrings, step-by-step logic, and inline explanations. Below is a summary of the pipeline architecture and pointers to each module.

### 6.1 Pipeline Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAW TRAFFIC VIDEO                          │
│                   (10–60 min, 30 fps)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              VIDEO SEGMENTATION (video_segmentation.py)         │
│  ┌──────────────────┐   ┌────────────────────┐                  │
│  │ Uniform Slicing   │ + │ Scene-Change Det.  │                  │
│  │ (30 s intervals)  │   │ (cosine threshold) │                  │
│  └────────┬─────────┘   └────────┬───────────┘                  │
│           └───────────┬──────────┘                               │
│                       ▼                                          │
│            Merged Boundary List                                  │
│                       │                                          │
│                       ▼                                          │
│            Key-Frame Sampling                                    │
│         (≤ 64 frames per segment)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           VISION ENCODING (frozen ViT-bigG)                     │
│                                                                  │
│     frames → patch tokens (N_frames × 256 × D)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         TEMPORAL MEMORY BANK (temporal_memory.py)               │
│                                                                  │
│  Layer 0: Raw segment tokens                                    │
│  Layer 1: Compressed summaries (Perceiver resampler)            │
│  Layer 2: Global memory slots (gated write, LRU eviction)       │
│                                                                  │
│  Output: [GLOBAL_MEM | RECENT_WINDOW | CURRENT_SEGMENT]         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          INSTRUCTION TOKENISATION                               │
│                                                                  │
│   [SYSTEM] traffic analysis prompt                               │
│   [USER]   "What caused the jam at 02:15?"                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│        LANGUAGE DECODER (Qwen-72B with LoRA)                    │
│                                                                  │
│   Input:  [context_tokens | instruction_tokens]                  │
│   Output: generated response tokens                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          MULTI-TASK LOSS (training_pipeline.py)                  │
│                                                                  │
│   L = w1·L_caption + w2·L_temporal + w3·L_class + w4·L_halluc   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Module-by-Module Summary

#### 6.2.1 Configuration (`pipeline_config.py`)

Centralises all hyperparameters as nested dataclasses: segmentation, temporal memory, LoRA, curriculum, training, loss weights, dataset paths, and evaluation settings. Every parameter is documented and has a sensible default.

#### 6.2.2 Video Segmentation (`video_segmentation.py`)

**Purpose:** Convert a long video into bounded segments with sampled key-frames.

**Steps:**

1. **Load video** at the configured FPS (default 2 fps) using a lazy memory-mapped reader.
2. **Detect scene boundaries** by computing cosine similarity between consecutive lightweight frame embeddings. Boundaries are inserted when similarity drops below a threshold (default 0.35).
3. **Merge boundaries** from both the uniform slicer (every 30 s) and the scene detector. Cuts within 2 s of each other are deduplicated.
4. **Sample key-frames** uniformly within each segment, capped at 64 per segment.
5. **Return** an ordered list of `VideoSegment` objects containing indices and metadata.

**Why two strategies?** Uniform slicing ensures regular coverage, while scene detection captures visually meaningful transitions (e.g., camera switches, tunnel entries). Their union provides robust segmentation across diverse traffic footage.

#### 6.2.3 Temporal Memory (`temporal_memory.py`)

**Purpose:** Maintain a compressed representation of the entire video history.

**Steps:**

1. **Compress** each segment's raw token sequence (potentially 1,000+ tokens) into a fixed-budget summary (256 tokens) using a Perceiver-style cross-attention resampler.
2. **Update global memory** via a gated write: the least-relevant slot is overwritten with a blend of old and new content, controlled by a learned sigmoid gate.
3. **Build context** for the decoder by concatenating three components:
   - Global memory slots (128 × D) — long-term important events
   - Recent window summaries (last 4 segments) — medium-term context
   - Current segment tokens (full resolution) — immediate visual input

#### 6.2.4 Dataset Builder (`dataset_builder.py`)

**Purpose:** Transform multi-level annotations into instruction-response training pairs.

**Steps:**

1. **Load** JSONL annotations with frame-, segment-, and video-level labels.
2. **Generate** multiple instruction-response pairs per video across six task types (temporal grounding, event description, video summary, counting, causal reasoning, risk assessment).
3. **Randomise** template selection to increase instruction diversity.
4. **Shuffle** the final dataset and export as JSONL.

#### 6.2.5 Training Pipeline (`training_pipeline.py`)

**Purpose:** Orchestrate the full fine-tuning run with curriculum learning.

**Steps:**

1. **Initialise model:** Load pre-trained weights, freeze all, inject LoRA, add temporal components, enable gradient checkpointing.
2. **Build dataset:** Load annotations and construct instruction pairs.
3. **Curriculum loop:** For each stage (warm-up → medium → long-form):
   - Filter dataset by maximum duration.
   - Warm-restart learning rate with cosine annealing.
   - For each epoch and mini-batch:
     - Reset the temporal memory bank.
     - Run `process_example()` to segment, encode, ingest, and decode.
     - Compute multi-task loss.
     - Accumulate gradients; step optimiser every N batches.
   - Evaluate on validation set; save best checkpoint.
4. **Merge LoRA weights** into the base model for efficient deployment.

#### 6.2.6 Evaluation (`evaluation.py`)

**Purpose:** Comprehensive automatic and human evaluation.

Detailed in Section 9 below.

#### 6.2.7 Inference (`inference.py`)

**Purpose:** End-to-end inference for a deployed model.

The inference pipeline mirrors training but without backpropagation: segment → encode → ingest into memory → build context → decode with beam search → post-process structured events from free-text output.

---

## 7. Instruction-Following Design

### 7.1 Prompt Format

Each training and inference input follows a structured three-part prompt format:

```
[SYSTEM] You are a traffic surveillance analysis assistant. Analyse the
provided video footage and answer the user's question using precise
timestamps (MM:SS format), event descriptions, severity assessments,
and causal explanations. If you are uncertain about any detail, state
your uncertainty rather than fabricating information.

[VIDEO_CONTEXT] <visual tokens from temporal memory bank>

[USER] <natural-language instruction>
```

The system prompt is fixed during training but can be customised at inference time to adjust the model's behaviour (e.g., requesting only timestamps, or requesting a formal incident report).

### 7.2 Multi-Task Learning

The model is trained jointly on all six task types within each mini-batch. Task diversity within a batch is enforced by the data-loader's sampling strategy:

| Task Type | Sampling Weight | Rationale |
|---|---|---|
| Temporal grounding | 25% | Core capability; most operationally valuable |
| Event description | 25% | Drives narrative quality |
| Video summary | 15% | Requires end-to-end video understanding |
| Causal reasoning | 15% | Tests deep temporal reasoning |
| Object counting | 10% | Anchors the model to precise visual evidence |
| Risk assessment | 10% | Higher-order judgement |

The loss function (Section 6.2.5) combines task-specific losses with configurable weights, allowing the training emphasis to be shifted post-hoc.

### 7.3 Instruction Diversity

To prevent the model from overfitting to specific phrasings, instructions are generated from a template bank with:

- **6 task categories** (as listed above)
- **3–5 templates per category** (see `dataset_builder.py` for the full list)
- **Randomised slot filling** for timestamps, event types, and object references
- **Paraphrased variants** generated via back-translation (EN → ZH → EN)

This yields approximately **30 unique instruction patterns**, each instantiated with video-specific values. At training time, the template for each example is selected uniformly at random.

### 7.4 Negative Examples

To train the model to refuse gracefully when the answer is not in the video, 10% of training examples are intentionally constructed with **unanswerable questions**:

- "Describe the accident at 45:00" for a 30-minute video (out-of-range timestamp)
- "How many bicycles are visible?" when the footage shows only motorised vehicles
- "What is the speed limit on this road?" (information not visually available)

Expected response: *"The requested information is not available in the provided video footage."*

---

## 8. Long-Video Memory Architecture

### 8.1 Design Rationale

Processing a 30-minute video at 2 fps with 256 tokens per frame produces ~921,600 tokens. No current LLM can attend to this directly. The memory architecture must:

1. **Preserve** globally important information (major incidents)
2. **Prioritise** recent context (current traffic state)
3. **Compress** redundant information (long stretches of normal flow)
4. **Fit** within a 4,096–8,192 token context window

### 8.2 Three-Layer Hierarchy

```
┌────────────────────────────────────────────────────────┐
│                  LAYER 2: GLOBAL MEMORY                │
│                                                        │
│  128 external memory slots (128 × D)                   │
│  Gated write: sigmoid(W · s_new) * s_new               │
│              + (1 - gate) * s_old                       │
│  LRU eviction: overwrite slot with lowest relevance    │
│  Relevance decay: r_t = r_{t-1} × 0.75                │
│                                                        │
│  Stores: Major accidents, sustained congestion,        │
│          camera/weather changes                        │
├────────────────────────────────────────────────────────┤
│              LAYER 1: SEGMENT SUMMARIES                │
│                                                        │
│  Each segment compressed to 256 tokens via Perceiver   │
│  Sliding window: last 4 summaries retained             │
│  Total: 4 × 256 = 1,024 tokens                        │
│                                                        │
│  Stores: Recent event descriptions, traffic state      │
│          transitions, short-term object tracks         │
├────────────────────────────────────────────────────────┤
│            LAYER 0: CURRENT SEGMENT                    │
│                                                        │
│  Full-resolution vision tokens for active segment      │
│  Up to 64 frames × 256 tokens = 16,384 tokens         │
│  (truncated or pooled to fit budget)                   │
│                                                        │
│  Stores: Detailed spatial information for the          │
│          segment being actively queried                 │
└────────────────────────────────────────────────────────┘
```

### 8.3 Context Budget

| Component | Tokens | Percentage |
|---|---|---|
| Global memory (Layer 2) | 128 | 2.4% |
| Recent window (Layer 1) | 1,024 | 19.2% |
| Current segment (Layer 0) | 3,072 | 57.6% |
| Instruction + system prompt | 512 | 9.6% |
| Generation headroom | 600 | 11.2% |
| **Total** | **5,336** | **100%** |

This fits comfortably within a 8,192-token context window.

### 8.4 Approach Trade-Offs

| Approach | Pros | Cons |
|---|---|---|
| **Sliding window only** | Simple; no compression artefacts | Cannot reason about events older than the window |
| **Hierarchical summarisation only** | Good long-range coverage | Summary compression loses spatial detail |
| **External memory only** | Fixed memory footprint | Requires careful gating; training instability |
| **Our hybrid (all three)** | Balances detail, recency, and long-range coverage | Increased architectural complexity; more hyperparameters |

### 8.5 Memory Compression Details

The Perceiver-style resampler uses **learnable query tokens** (256 × D) that attend over the segment's raw token sequence via cross-attention:

```
Input:  segment_tokens  (S × D),  where S ≈ 16,384
Queries: learned_queries (256 × D)

MultiHeadCrossAttention:
    Q = learned_queries × W_Q
    K = segment_tokens  × W_K
    V = segment_tokens  × W_V

    Output = softmax(Q K^T / √d_k) V    →  (256 × D)
```

This reduces 16,384 tokens to 256 tokens — a **64× compression ratio** — while preserving the most attended-to information via the learned query mechanism.

---

## 9. Evaluation Plan

### 9.1 Automatic Metrics

| Metric | Target Task | Computation | Threshold/Target |
|---|---|---|---|
| **BLEU-4** | Caption quality | n-gram overlap with references | ≥ 0.25 |
| **CIDEr** | Caption quality | TF-IDF weighted n-gram consensus | ≥ 1.0 |
| **METEOR** | Caption quality | Alignment-based with synonym matching | ≥ 0.30 |
| **R@IoU=0.3** | Temporal grounding | Recall at IoU threshold 0.3 | ≥ 0.70 |
| **R@IoU=0.5** | Temporal grounding | Recall at IoU threshold 0.5 | ≥ 0.55 |
| **R@IoU=0.7** | Temporal grounding | Recall at IoU threshold 0.7 | ≥ 0.35 |
| **Macro-F1** | Event classification | Per-class F1, macro-averaged | ≥ 0.65 |
| **Hallucination Rate** | Factual accuracy | Fraction of generated entities absent from GT | ≤ 0.10 |

### 9.2 Composite Score

For checkpoint selection, a single composite score is computed:

```
Score = 0.3 × CIDEr + 0.3 × R@IoU=0.5 + 0.2 × F1 + 0.2 × (1 - HallucinationRate)
```

This weighting prioritises caption quality and temporal grounding equally, with event classification and hallucination avoidance as secondary objectives.

### 9.3 Human Evaluation Protocol

| Criterion | Scale | Description | Inter-Annotator Agreement |
|---|---|---|---|
| Factual accuracy | 1–5 Likert | Are all stated facts verifiable in the video? | Cohen's κ ≥ 0.7 |
| Temporal precision | Binary | Are reported timestamps within ±5 s of GT? | Cohen's κ ≥ 0.8 |
| Coherence & fluency | 1–5 Likert | Is the response well-structured and grammatical? | Cohen's κ ≥ 0.6 |

**Protocol details:**

- **Sample size:** 200 examples from the test set
- **Annotators:** 3 trained annotators per example (majority vote)
- **Compensation:** Standard hourly rate; no per-item incentive (to avoid speed bias)
- **Calibration:** 20 pilot examples with adjudicated gold labels before the main study
- **Presentation:** Annotators see the video, the instruction, and the model response side-by-side; ground-truth responses are hidden

### 9.4 Ablation Studies

| Ablation | Purpose |
|---|---|
| No temporal memory (frame-bag baseline) | Quantify the contribution of hierarchical memory |
| No curriculum (direct long-form training) | Measure the benefit of progressive duration |
| No LoRA (full fine-tuning) | Compare parameter efficiency vs. adaptation quality |
| No hallucination penalty | Assess the effect of the penalty term on factual accuracy |
| Window size = 1 vs. 4 vs. 8 | Optimise the sliding window hyperparameter |

---

## 10. Failure Cases & Limitations

### 10.1 Known Failure Modes

| Failure Mode | Description | Example | Mitigation |
|---|---|---|---|
| **Temporal aliasing** | Model confuses similar events at different timestamps | Two congestion events 10 min apart reported as one | Increase temporal position encoding resolution |
| **Small object blindness** | Distant vehicles or pedestrians missed by the vision encoder | Motorcycle at far-field not detected | Fine-tune vision encoder on traffic-specific crops |
| **Hallucinated timestamps** | Model generates plausible but incorrect timestamps | Reports accident at 03:15 when it actually occurred at 03:45 | Stronger hallucination penalty; timestamp verification post-processing |
| **Weather confusion** | Fog vs. camera blur vs. low-light misclassified | Night scene described as "foggy" | Augment training data with diverse weather conditions |
| **Causal over-attribution** | Model invents causal links not supported by visual evidence | "The accident was caused by a red-light runner" when signal state is not visible | Train on negative causal examples; calibrate uncertainty |
| **Repetitive generation** | Long responses loop or repeat phrases | Same sentence restated three times in a summary | Repetition penalty in decoding; length-based stopping |
| **Cross-segment information loss** | Important detail in early segment forgotten by the time the model generates a response | Vehicle involved in early incident not referenced in final summary | Increase global memory slots; tune relevance decay |

### 10.2 Fundamental Limitations

1. **Annotation bottleneck:** High-quality temporal annotations for long traffic videos are expensive and time-consuming. Our dataset of 2,000 videos may not capture the full diversity of traffic scenarios globally.

2. **Camera dependency:** The model is trained on fixed-angle surveillance cameras. Performance may degrade on dashcam footage, drone video, or fish-eye lenses without additional fine-tuning.

3. **Real-time constraint:** The current architecture processes videos offline. Real-time inference (< 1 s latency per query on a live stream) is not feasible with a 72B-parameter model.

4. **Privacy concerns:** Traffic surveillance inevitably captures personally identifiable information (licence plates, faces). The model may inadvertently memorise and reproduce such information.

5. **Single-language limitation:** The current design targets English-language instructions and responses. Multi-lingual support requires additional training data and evaluation.

---

## 11. Future Improvements

### 11.1 Short-Term (3–6 months)

- **Streaming inference:** Adapt the memory architecture for real-time processing by ingesting segments as they arrive from a live camera feed, rather than processing the full video post-hoc.
- **Multi-camera fusion:** Extend the temporal memory to accept segments from multiple synchronised cameras at the same intersection, enabling cross-view reasoning.
- **Active learning:** Identify low-confidence predictions and route them to human annotators for targeted labelling, improving dataset quality efficiently.

### 11.2 Medium-Term (6–12 months)

- **Vision encoder adaptation:** Unfreeze the last few ViT layers and fine-tune on traffic-specific object detection (small vehicles, traffic signs, lane markings) to improve frame-level representation quality.
- **Retrieval-augmented generation:** Integrate an external knowledge base of traffic regulations and road geometry to ground the model's responses in domain knowledge beyond visual evidence.
- **Multi-lingual support:** Extend instruction templates to Chinese, Spanish, and Arabic to serve global transportation agencies.

### 11.3 Long-Term (12+ months)

- **End-to-end differentiable segmentation:** Replace the rule-based segmentation (uniform + scene-change) with a learned segmentation module trained jointly with the VLM, optimising segment boundaries for downstream task performance.
- **Model distillation:** Distill the 32B model into a 4B or 8B variant (leveraging Qwen3-VL's smaller model family) suitable for edge deployment on traffic management centre workstations without GPU clusters.
- **Sim-to-real transfer:** Pre-train on large-scale synthetic traffic data (CARLA, SUMO) before fine-tuning on real footage to reduce annotation requirements.
- **Autonomous incident response:** Integrate the VLM into a closed-loop system that not only detects and describes incidents but also triggers automated responses (signal timing changes, dynamic message sign updates, emergency dispatch).

---

## 12. Conclusion

This proposal presents a complete system design for fine-tuning a Vision-Language Model to understand long-duration traffic surveillance videos. The recommended backbone — **Qwen3-VL-32B** (updated from the original Qwen-VL-Chat-72B) — was selected after a comprehensive evaluation of the 2026 VLM landscape including GPT-5.4, Gemini 2.5 Pro, Claude 4.6, Gemma 3, and InternVL3. While proprietary models like Gemini 2.5 Pro offer impressive native long-video support (up to 6 hours), they lack fine-tuning access essential for domain-specific traffic adaptation. Qwen3-VL uniquely combines open weights, native temporal grounding, hours-long video support, and a production-ready LoRA fine-tuning framework.

The design addresses the fundamental challenges of token explosion, temporal reasoning, and memory limits through three interlocking innovations:

1. **Adaptive video segmentation** that combines uniform temporal slicing with content-aware scene detection to partition long videos into bounded, overlapping segments.

2. **A hierarchical temporal memory bank** with three layers — global external memory, compressed segment summaries, and full-resolution current-segment tokens — that enables cross-segment reasoning within a fixed context budget.

3. **Parameter-efficient fine-tuning with curriculum learning** that adapts 0.81% of the model's parameters via LoRA (rank 64) and newly introduced temporal components, trained progressively from short clips to full-length surveillance feeds.

The gap analysis in Section 2.4 demonstrates that even with 2026-era million-token context windows, our segmentation and memory architecture remains necessary for production surveillance — due to fine-tuning requirements, spatial-detail preservation, and deployment cost considerations.

The accompanying pseudo-code pipeline (implemented in the `vlm_ft/` directory) provides a concrete, reproducible specification of every system component: segmentation, memory management, dataset construction, multi-task training, and evaluation. The evaluation plan combines standard NLG metrics, temporal grounding IoU, hallucination detection, and a rigorous human evaluation protocol to ensure that the system meets the operational requirements of traffic management.

While limitations remain — particularly around real-time processing, annotation scalability, and privacy — the proposed architecture establishes a principled foundation for deploying VLMs in intelligent transportation systems. The modular design allows individual components to be improved independently, and the curriculum learning strategy ensures that the model's capabilities grow gracefully with the complexity of the input.

---

## 13. References

1. Alayrac, J.-B., et al. (2022). Flamingo: A visual language model for few-shot learning. *NeurIPS 2022*.

2. Bai, J., et al. (2023). Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. *arXiv:2308.12966*.

3. Chen, Z., et al. (2025). InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. *arXiv:2504.10479*.

4. Google DeepMind. (2024). Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv:2403.05530*.

5. Google DeepMind. (2025). Gemini 2.5 Pro technical report. *Google AI Technical Report*.

6. Google DeepMind. (2025). Gemma 3 report. *Google DeepMind Technical Report*.

7. He, B., et al. (2024). MA-LMM: Memory-augmented large multimodal model for long-term video understanding. *CVPR 2024*.

8. Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

9. Jaegle, A., et al. (2021). Perceiver: General perception with iterative attention. *ICML 2021*.

10. Lin, B., et al. (2023). Video-LLaVA: Learning united visual representation by alignment before projection. *arXiv:2311.10122*.

11. Lin, B., et al. (2024). LLaVA-NeXT-Video: Improved video understanding with AnyRes and length generalisation. *arXiv:2407.15841*.

12. OpenAI. (2023). GPT-4V(ision) system card. *OpenAI Technical Report*.

13. OpenAI. (2026). Introducing GPT-5.4. *OpenAI Technical Report*.

14. Papalampidi, P., et al. (2024). A simple recipe for contrastively pre-training video-first encoders beyond 16 frames. *CVPR 2024*.

15. Pervaiz, U., et al. (2025). TrafficVILA: Scaling vision-language models to high-resolution video understanding for traffic surveillance. *ICCV 2025 Workshop (AI City Challenge)*.

16. Qwen Team. (2025). Qwen3-VL technical report. *arXiv:2511.21631*.

17. Ren, S., et al. (2024). TimeChat: A time-sensitive multimodal large language model for long video understanding. *CVPR 2024*.

18. Song, E., et al. (2024). MovieChat: From dense token to sparse memory for long video understanding. *CVPR 2024*.

19. SurveillanceVQA Authors. (2025). SurveillanceVQA-589K: A large-scale video question-answering benchmark for surveillance. *arXiv:2505.12589*.

20. Wen, L., et al. (2020). UA-DETRAC: A benchmark suite for multi-object detection and tracking in unconstrained environments. *International Journal of Computer Vision*, 128(2), 319–338.

21. Yu, F., et al. (2020). BDD100K: A diverse driving dataset for heterogeneous multitask learning. *CVPR 2020*.

---

*End of Document*
