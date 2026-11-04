"""Run a paired prompt/threshold pilot using the senior detector's postprocessing."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForZeroShotObjectDetection, AutoProcessor

from detect_cached_frames import merge_detections, read, save, sha


ALIASES = {
    "built-in refrigerator": "refrigerator",
    "undercounter refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "bedspread": "blanket",
    "duvet": "blanket",
    "bed blanket": "blanket",
}
THRESHOLDS = {"normal": (0.20, 0.25), "loose": (0.12, 0.17)}


def audit_weights(model, weights):
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    result = {"sha256": digest.hexdigest()}
    if model.config.model_type == "mm-grounding-dino":
        from safetensors import safe_open

        names = ("bbox_embed.", "class_embed.")
        prefixes = (*names, *("model.decoder." + name for name in names))
        with safe_open(str(weights), framework="pt", device="cpu") as source:
            keys = [key for key in source.keys() if key.startswith(prefixes)]
            state = model.state_dict()
            if any(not torch.equal(state[key].cpu(), source.get_tensor(key)) for key in keys):
                assert not model.config.decoder_bbox_embed_share and not model.config.decoder_cls_embed_share
                for name in ("bbox_embed", "class_embed"):
                    old = getattr(model, name)
                    heads = torch.nn.ModuleList([copy.deepcopy(head) for head in old])
                    tensors = {}
                    for prefix in (f"model.decoder.{name}.", f"{name}."):
                        tensors.update({key[len(prefix):]: source.get_tensor(key)
                                        for key in keys if key.startswith(prefix)})
                    heads.load_state_dict(tensors, strict=True)
                    setattr(model, name, heads)
                    setattr(model.model.decoder, name, heads)
                result["restored_unshared_heads_from_checkpoint"] = True
                state = model.state_dict()
            assert keys and all(torch.equal(state[key].cpu(), source.get_tensor(key)) for key in keys)
        for heads in (model.bbox_embed, model.class_embed):
            assert len({next(head.parameters()).data_ptr() for head in heads}) == len(heads)
        result["verified_decoder_head_tensors"] = len(keys)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--output", required=True, type=Path)
    prompt_mode = ap.add_mutually_exclusive_group()
    prompt_mode.add_argument("--base-only", action="store_true")
    prompt_mode.add_argument("--aliases-only", action="store_true")
    ap.add_argument("--alias", action="append", choices=ALIASES,
                    help="Use only these aliases instead of the full default set.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.80 if torch.cuda.get_device_properties(device).total_memory < 6 * 1024**3 else .30,
                                                   device=device)
    config = read(args.cache / "config.json")
    candidates = read(args.cache / "candidates.json")
    for row in candidates:
        assert sha(args.cache / row["frame"]) == row["frame_sha256"]
    model_config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.checkpoint, config=model_config, local_files_only=True
    ).eval()
    weights = audit_weights(model, args.checkpoint / "model.safetensors")
    model.to(device)
    aliases = {label: ALIASES[label] for label in (args.alias or ALIASES)}
    prompts = {} if args.aliases_only else {"base": config["objects"]}
    if not args.base_only:
        prompts["aliases"] = [*config["objects"], *aliases]
    w, h = config["width"], config["height"]
    args.output.mkdir(parents=True)
    save(args.output / "provenance.json", {
        "model": args.model, "revision": args.revision,
        "weights": weights,
        "source_candidates_sha256": sha(args.cache / "candidates.json"),
        "thresholds_text_box": THRESHOLDS, "prompts": prompts, "aliases": aliases,
        "device": str(device),
    })
    print(args.model, type(model).__name__, len(candidates), flush=True)
    for prompt_name, labels in prompts.items():
        started = time.monotonic()
        caption = ""
        spans = []
        for label in labels:
            spans.append((len(caption), len(caption) + len(label)))
            caption += label + " . "
        caption = caption.strip()
        encoded = processor.tokenizer(caption, return_offsets_mapping=True)
        token_indices = [
            [i for i, (x, y) in enumerate(encoded["offset_mapping"]) if y > x and x < end and y > begin]
            for begin, end in spans
        ]
        assert all(token_indices)
        mapping = aliases if prompt_name == "aliases" else {}
        with (args.output / f"{prompt_name}.jsonl").open("x", encoding="utf-8") as output:
            for i, row in enumerate(candidates):
                with Image.open(args.cache / row["frame"]) as src:
                    im = src.convert("RGB")
                assert im.size == (w, h)
                inputs = processor(images=im, text=caption, return_tensors="pt",
                                   size={"shortest_edge": 640, "longest_edge": 1024}).to(device)
                assert inputs["input_ids"][0].tolist() == encoded["input_ids"]
                with torch.inference_mode():
                    pred = model(**inputs)
                probs = pred.logits[0].sigmoid().float().cpu().numpy()
                raw = pred.pred_boxes[0].float().cpu().numpy()
                boxes = np.column_stack((raw[:, :2] - raw[:, 2:] / 2,
                                         raw[:, :2] + raw[:, 2:] / 2)) * [w, h, w, h]
                means = np.stack([probs[:, idx].mean(1) for idx in token_indices], 1)
                peaks = np.stack([probs[:, idx].max(1) for idx in token_indices], 1)
                output.write(json.dumps({
                    "source_frame_index": row["source_frame_index"],
                    "timestamp_s": row["timestamp_s"],
                    **{key: merge_detections(boxes, means, peaks, labels, mapping, w, h, text, box)
                       for key, (text, box) in THRESHOLDS.items()},
                }) + "\n")
                if i % 20 == 0:
                    output.flush()
                    print(prompt_name, i + 1, "/", len(candidates), flush=True)
        print(prompt_name, "seconds", round(time.monotonic() - started, 2), flush=True)


if __name__ == "__main__":
    main()
