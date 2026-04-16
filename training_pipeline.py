"""
training_pipeline.py
====================
PSEUDO-CODE — End-to-End Fine-Tuning Pipeline for Long Traffic Video VLM.

This is the central orchestration module.  It ties together:
  • Video segmentation   (video_segmentation.py)
  • Temporal memory      (temporal_memory.py)
  • Dataset construction (dataset_builder.py)
  • LoRA injection & freezing strategy
  • Multi-task loss computation
  • Curriculum learning scheduler
  • Training loop with gradient accumulation
"""

from typing import Dict, List, Tuple

from pipeline_config import PipelineConfig
from video_segmentation import segment_video, VideoSegment
from temporal_memory import TemporalMemoryBank
from dataset_builder import build_dataset


# ═══════════════════════════════════════════════════════════════════════
# STEP 1  Model initialisation & LoRA injection
# ═══════════════════════════════════════════════════════════════════════

def initialise_model(cfg: PipelineConfig):
    """
    1. Load the pre-trained VLM (e.g. Qwen-VL-Chat-72B) with the
       vision encoder, projection layer, and language decoder.
    2. Freeze the entire model.
    3. Inject LoRA adapters into the target modules.
    4. Unfreeze the temporal cross-attention layers and the memory
       compressor (these are new parameters, not pre-trained).
    5. Enable gradient checkpointing to reduce VRAM.

    Frozen vs. trainable breakdown
    ──────────────────────────────
    Component                 | Status
    ─────────────────────────-+─────────
    Vision encoder (ViT)      | FROZEN
    Patch embedding           | FROZEN
    Visual projection MLP     | LoRA adapters (trainable)
    LLM self-attention Q, V   | LoRA adapters (trainable)
    LLM FFN layers            | FROZEN
    Temporal cross-attention   | FULLY TRAINABLE (new)
    Memory compressor          | FULLY TRAINABLE (new)
    LM head                   | FROZEN
    """
    model = load_pretrained_vlm(cfg.model_name)

    # freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # inject LoRA into specified modules
    lora_modules = inject_lora(
        model,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
    )

    # add and unfreeze new temporal components
    model.temporal_cross_attention = TemporalCrossAttention(hidden_dim=model.hidden_dim)
    model.memory_compressor = PerceiverResampler(
        num_queries=cfg.temporal_memory.summary_token_budget,
        hidden_dim=model.hidden_dim,
    )

    for param in model.temporal_cross_attention.parameters():
        param.requires_grad = True
    for param in model.memory_compressor.parameters():
        param.requires_grad = True

    if cfg.training.grad_checkpoint:
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
          f"({100 * trainable / total:.2f}%)")

    return model


# ═══════════════════════════════════════════════════════════════════════
# STEP 2  Multi-task loss function
# ═══════════════════════════════════════════════════════════════════════

def compute_loss(
    predictions: Dict[str, "Tensor"],
    targets: Dict[str, "Tensor"],
    cfg: PipelineConfig,
) -> "Tensor":
    """
    Weighted sum of task-specific losses:

    L_total = w1·L_caption + w2·L_temporal + w3·L_class + w4·L_halluc

    Losses
    ------
    L_caption  : next-token cross-entropy on the generated caption tokens
    L_temporal : smooth-L1 loss on predicted (start, end) timestamps
    L_class    : cross-entropy on event-type classification logits
    L_halluc   : penalty term that discourages generating entity names
                 or timestamps not present in the visual evidence
    """
    L_caption = cross_entropy_loss(
        predictions["caption_logits"],
        targets["caption_token_ids"],
    )

    L_temporal = smooth_l1_loss(
        predictions["temporal_boundaries"],  # (B, 2) predicted start/end
        targets["temporal_boundaries"],      # (B, 2) ground-truth
    )

    L_class = cross_entropy_loss(
        predictions["event_class_logits"],
        targets["event_class_labels"],
    )

    L_halluc = hallucination_penalty(
        generated_tokens=predictions["caption_logits"].argmax(-1),
        visual_evidence=predictions["attended_visual_tokens"],
    )

    total_loss = (
        cfg.loss.caption_loss_weight * L_caption
        + cfg.loss.temporal_grounding_loss_weight * L_temporal
        + cfg.loss.classification_loss_weight * L_class
        + cfg.loss.hallucination_penalty_weight * L_halluc
    )

    return total_loss


# ═══════════════════════════════════════════════════════════════════════
# STEP 3  Process a single training example
# ═══════════════════════════════════════════════════════════════════════

def process_example(
    model,
    memory_bank: TemporalMemoryBank,
    video_path: str,
    instruction: str,
    cfg: PipelineConfig,
) -> Dict[str, "Tensor"]:
    """
    Forward pass for one training sample:

    1. Segment the video.
    2. Encode each segment through the frozen vision encoder.
    3. Feed segments into the temporal memory bank sequentially.
    4. For the current (target) segment, build the full context
       via the memory bank.
    5. Concatenate [context_tokens, instruction_tokens] and pass
       through the language decoder to obtain logits.
    """
    # segment the video
    segments: List[VideoSegment] = segment_video(video_path, cfg.segmentation)

    # encode each segment and feed into memory
    for seg in segments:
        frame_pixels = load_frames(video_path, seg.keyframe_indices)
        vision_tokens = model.vision_encoder(frame_pixels)  # (N_frames, T_patch, D)
        vision_tokens = vision_tokens.flatten(0, 1)          # (N_frames*T_patch, D)
        seg.keyframe_embeddings = vision_tokens

        memory_bank.ingest_segment(
            segment_id=seg.segment_id,
            segment_tokens=vision_tokens,
            time_range=(seg.start_time_s, seg.end_time_s),
        )

    # build context for the last segment (target)
    current_tokens = segments[-1].keyframe_embeddings
    context = memory_bank.build_context(current_tokens)

    # tokenise the instruction
    instruction_tokens = model.tokenizer.encode(instruction)

    # forward through the language decoder
    input_embeds = concatenate([context, instruction_tokens], dim=0)
    logits = model.language_decoder(input_embeds)

    return logits


# ═══════════════════════════════════════════════════════════════════════
# STEP 4  Curriculum learning scheduler
# ═══════════════════════════════════════════════════════════════════════

def curriculum_schedule(cfg: PipelineConfig, dataset):
    """
    Progressively increase video duration across training stages:

    Stage 1 (warm-up)  : only clips ≤  60 s  |  3 epochs  |  lr = 2e-4
    Stage 2 (medium)   : clips ≤ 300 s       |  5 epochs  |  lr = 1e-4
    Stage 3 (long-form): clips ≤ 1800 s      |  8 epochs  |  lr = 5e-5

    Within each stage the data-loader filters examples by duration.
    The learning rate is warm-restarted at each stage boundary.
    """
    for stage in cfg.curriculum.stages:
        stage_data = [
            ex for ex in dataset
            if ex["duration_s"] <= stage["max_duration_s"]
        ]
        yield {
            "name": stage["name"],
            "data": stage_data,
            "epochs": stage["epochs"],
            "lr": stage["lr"],
        }


# ═══════════════════════════════════════════════════════════════════════
# STEP 5  Main training loop
# ═══════════════════════════════════════════════════════════════════════

def train(cfg: PipelineConfig):
    """
    Orchestrates the full fine-tuning run.

    Pseudo-code walkthrough
    -----------------------
    1. Initialise model with LoRA + temporal layers.
    2. Build the instruction-response dataset.
    3. For each curriculum stage:
       a. Filter dataset by max duration.
       b. Create data-loader with dynamic batching.
       c. For each epoch:
          i.   For each mini-batch:
               - Reset the temporal memory bank.
               - For each example in the batch:
                 • Run process_example() to get logits.
                 • Compute multi-task loss.
               - Accumulate gradients.
               - Every *gradient_accumulation_steps* batches, step the optimiser.
          ii.  Run validation and log metrics.
       d. Save the best checkpoint per stage.
    4. Merge LoRA weights into the base model for deployment.
    """
    # 1. Model init
    model = initialise_model(cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.curriculum.stages[0]["lr"],
        weight_decay=cfg.training.weight_decay,
    )

    # 2. Dataset
    dataset = build_dataset(cfg.dataset.train_annotation_path)
    val_dataset = build_dataset(cfg.dataset.val_annotation_path)

    best_metric = float("-inf")

    # 3. Curriculum stages
    for stage in curriculum_schedule(cfg, dataset):
        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE: {stage['name']}  "
              f"(max {stage['data'][0]['duration_s'] if stage['data'] else 0}s, "
              f"lr={stage['lr']})")
        print(f"{'='*60}")

        # warm-restart learning rate
        set_learning_rate(optimizer, stage["lr"])
        scheduler = CosineAnnealingLR(optimizer, T_max=stage["epochs"])

        dataloader = DataLoader(
            stage["data"],
            batch_size=cfg.training.batch_size,
            shuffle=True,
            collate_fn=traffic_video_collate,
        )

        for epoch in range(stage["epochs"]):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(dataloader):
                # reset temporal memory per video
                memory_bank = TemporalMemoryBank(cfg.temporal_memory)

                # forward
                predictions = process_example(
                    model,
                    memory_bank,
                    video_path=batch["video_path"],
                    instruction=batch["instruction"],
                    cfg=cfg,
                )

                # loss
                loss = compute_loss(predictions, batch["targets"], cfg)
                loss = loss / cfg.training.gradient_accumulation_steps
                loss.backward()
                epoch_loss += loss.item()

                # gradient accumulation step
                if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                # periodic evaluation
                if (step + 1) % cfg.training.eval_every_n_steps == 0:
                    val_metrics = evaluate(model, val_dataset, cfg)
                    log_metrics(val_metrics, step=step, epoch=epoch)

                    if val_metrics["composite_score"] > best_metric:
                        best_metric = val_metrics["composite_score"]
                        save_checkpoint(model, cfg.output_dir, tag="best")

            scheduler.step()
            print(f"  Epoch {epoch+1}/{stage['epochs']}  "
                  f"avg_loss={epoch_loss / len(dataloader):.4f}")

        save_checkpoint(model, cfg.output_dir, tag=f"stage-{stage['name']}")

    # 4. Merge LoRA weights for deployment
    model = merge_lora_weights(model)
    save_checkpoint(model, cfg.output_dir, tag="final-merged")
    print("\nTraining complete.")
