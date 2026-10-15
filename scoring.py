"""BVB pilot matching rules. See licenses/BVB-MIT.txt and NOTICE.md."""
import re
from typing import Any
from difflib import SequenceMatcher

NUMERIC_TYPES = {
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
    "object_abs_distance",
}

def normalize_text(value: Any) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"^[a-d][\).:\-]\s*", "", text)
    text = re.sub(r"[^a-z0-9.\s,\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .,")

def extract_number(value: Any) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None

def semantic_correct(answer: Any, qa: dict[str, Any]) -> bool:
    """Match a model answer against VSI-Bench ground truth (pilot rules)."""
    ground_truth = str(qa["ground_truth"]).strip()
    question_type = str(qa["question_type"])
    if question_type in NUMERIC_TYPES:
        predicted = extract_number(answer)
        expected = extract_number(ground_truth)
        if predicted is None or expected is None:
            return False
        if question_type == "object_counting":
            return abs(predicted - expected) < 0.5
        if expected == 0:
            return abs(predicted) < 1e-6
        return abs(predicted - expected) / abs(expected) <= 0.20
    letter = re.search(
        r"(?:^|\b)(?:option\s*)?([A-Da-d])(?:\b|$)", str(answer).strip()
    )
    if letter and len(str(answer)) < 40:
        return letter.group(1).upper() == ground_truth.upper()
    target = None
    for option in qa.get("options", []):
        match = re.match(r"([A-D])\.\s*(.*)", str(option), re.I)
        if match and match.group(1).upper() == ground_truth.upper():
            target = normalize_text(match.group(2))
            break
    if not target:
        return False
    predicted_text = normalize_text(answer)
    if predicted_text == target or target in predicted_text or predicted_text in target:
        return True
    return SequenceMatcher(None, predicted_text, target).ratio() >= 0.88
