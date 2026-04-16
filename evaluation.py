"""
evaluation.py
=============
PSEUDO-CODE — Evaluation Pipeline for Long Traffic Video VLM.

Covers both automatic metrics and human evaluation protocols.

Automatic metrics
-----------------
  • Caption quality   : BLEU-4, CIDEr, METEOR
  • Temporal grounding: IoU at thresholds [0.3, 0.5, 0.7]
  • Event classification: macro-F1
  • Hallucination rate: fraction of generated entities absent from GT

Human evaluation
----------------
  • Factual accuracy  (1-5 Likert scale)
  • Temporal precision (binary: correct or off by > 5 s)
  • Coherence & fluency (1-5 Likert scale)
"""

from dataclasses import dataclass
from typing import Dict, List

from pipeline_config import EvaluationConfig


@dataclass
class EvalResult:
    metric_name: str
    value: float
    details: Dict = None


# ═══════════════════════════════════════════════════════════════════════
# STEP 1  Caption quality metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_caption_metrics(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """
    Standard NLG metrics between predicted captions and ground-truth
    reference captions.

    Returns dict with keys: bleu4, cider, meteor.

    Example
    -------
    prediction : "A white truck rear-ends a sedan at 02:15 causing a
                  three-vehicle pile-up on the highway."
    reference  : "At 2 minutes 15 seconds a truck collides with a car,
                  triggering a chain reaction involving three vehicles."
    """
    bleu4 = corpus_bleu(predictions, references, max_n=4)
    cider = corpus_cider(predictions, references)
    meteor = corpus_meteor(predictions, references)

    return {"bleu4": bleu4, "cider": cider, "meteor": meteor}


# ═══════════════════════════════════════════════════════════════════════
# STEP 2  Temporal grounding IoU
# ═══════════════════════════════════════════════════════════════════════

def compute_temporal_iou(
    pred_intervals: List[tuple],   # [(start, end), ...]
    gt_intervals: List[tuple],
    thresholds: List[float],
) -> Dict[str, float]:
    """
    For each (predicted, ground-truth) interval pair compute IoU.
    Report Recall@IoU for each threshold.

    IoU = intersection / union of the two time intervals.

    Example
    -------
    pred = (120.0, 180.0)   # model says event is 2:00–3:00
    gt   = (125.0, 175.0)   # ground-truth is 2:05–2:55
    IoU  = 50 / 60 = 0.833  →  passes thresholds 0.3, 0.5, 0.7
    """
    results = {}
    for thr in thresholds:
        hits = 0
        for pred, gt in zip(pred_intervals, gt_intervals):
            inter_start = max(pred[0], gt[0])
            inter_end = min(pred[1], gt[1])
            intersection = max(0, inter_end - inter_start)

            union = (pred[1] - pred[0]) + (gt[1] - gt[0]) - intersection
            iou = intersection / union if union > 0 else 0.0

            if iou >= thr:
                hits += 1

        recall = hits / len(gt_intervals) if gt_intervals else 0.0
        results[f"R@IoU={thr}"] = recall

    return results


# ═══════════════════════════════════════════════════════════════════════
# STEP 3  Event classification F1
# ═══════════════════════════════════════════════════════════════════════

def compute_classification_f1(
    pred_labels: List[str],
    gt_labels: List[str],
) -> float:
    """
    Macro-averaged F1 across event types:
      {congestion, accident, violation, normal_flow}

    Handles class imbalance common in traffic data (normal_flow dominates).
    """
    classes = sorted(set(gt_labels))
    f1_scores = []

    for cls in classes:
        tp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(pred_labels, gt_labels) if p != cls and g == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


# ═══════════════════════════════════════════════════════════════════════
# STEP 4  Hallucination detection
# ═══════════════════════════════════════════════════════════════════════

def compute_hallucination_rate(
    generated_texts: List[str],
    ground_truth_entities: List[List[str]],
) -> float:
    """
    Extract named entities and timestamps from the generated text.
    Any entity or timestamp NOT present in the ground-truth annotations
    counts as a hallucination.

    hallucination_rate = n_hallucinated / n_total_entities

    Traffic-specific hallucinations include:
      • Mentioning a vehicle type not visible (e.g. "bus" when only cars)
      • Citing a timestamp outside the video's duration
      • Describing weather conditions inconsistent with visual evidence
    """
    total_entities = 0
    hallucinated = 0

    for text, gt_ents in zip(generated_texts, ground_truth_entities):
        extracted = extract_entities_and_timestamps(text)
        total_entities += len(extracted)
        for ent in extracted:
            if ent not in gt_ents:
                hallucinated += 1

    return hallucinated / total_entities if total_entities > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
# STEP 5  Composite evaluation entry point
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model, val_dataset, cfg) -> Dict[str, float]:
    """
    Run the full evaluation suite on the validation set.

    1. Generate predictions for every example (greedy decoding).
    2. Compute each metric.
    3. Return a combined dict + a single composite score for
       checkpoint selection.

    Composite score = 0.3*CIDEr + 0.3*R@IoU=0.5 + 0.2*F1 + 0.2*(1 - halluc_rate)
    """
    predictions = {"captions": [], "intervals": [], "event_labels": []}
    targets = {"captions": [], "intervals": [], "event_labels": [], "entities": []}

    for example in val_dataset:
        pred = model.generate(example["video_context"], example["instruction"])
        predictions["captions"].append(pred["caption"])
        predictions["intervals"].append(pred["temporal_boundary"])
        predictions["event_labels"].append(pred["event_class"])

        targets["captions"].append(example["reference_captions"])
        targets["intervals"].append(example["temporal_boundary"])
        targets["event_labels"].append(example["event_class"])
        targets["entities"].append(example["entities"])

    caption_metrics = compute_caption_metrics(
        predictions["captions"], targets["captions"]
    )
    temporal_metrics = compute_temporal_iou(
        predictions["intervals"],
        targets["intervals"],
        cfg.evaluation.temporal_iou_thresholds,
    )
    f1 = compute_classification_f1(
        predictions["event_labels"], targets["event_labels"]
    )
    halluc = compute_hallucination_rate(
        predictions["captions"], targets["entities"]
    )

    all_metrics = {
        **caption_metrics,
        **temporal_metrics,
        "event_classification_f1": f1,
        "hallucination_rate": halluc,
    }

    all_metrics["composite_score"] = (
        0.3 * caption_metrics["cider"]
        + 0.3 * temporal_metrics.get("R@IoU=0.5", 0.0)
        + 0.2 * f1
        + 0.2 * (1.0 - halluc)
    )

    return all_metrics


# ═══════════════════════════════════════════════════════════════════════
# STEP 6  Human evaluation protocol
# ═══════════════════════════════════════════════════════════════════════

def run_human_evaluation(
    model,
    sample_set: List[dict],
    cfg: EvaluationConfig,
) -> Dict[str, float]:
    """
    Protocol for human annotators:

    1. Randomly sample *human_eval_sample_size* examples.
    2. For each example, present:
       - The original video segment
       - The instruction
       - The model's generated response
    3. Annotators rate on three axes (1-5 Likert scale):
       a. Factual accuracy — are all stated facts visible in the video?
       b. Temporal precision — are timestamps within 5 s of ground truth?
       c. Coherence & fluency — is the response well-structured?
    4. Inter-annotator agreement is measured via Cohen's κ.
    5. Return average scores per axis.
    """
    samples = random_sample(sample_set, k=cfg.human_eval_sample_size)
    scores = {"factual_accuracy": [], "temporal_precision": [], "coherence": []}

    for sample in samples:
        pred = model.generate(sample["video_context"], sample["instruction"])

        # collect ratings from annotators (simulated here)
        rating = human_annotator_interface(
            video=sample["video_path"],
            instruction=sample["instruction"],
            response=pred["caption"],
        )
        scores["factual_accuracy"].append(rating["factual_accuracy"])
        scores["temporal_precision"].append(rating["temporal_precision"])
        scores["coherence"].append(rating["coherence"])

    return {k: mean(v) for k, v in scores.items()}
