"""Complete missing paired detector trials; reuse frozen original detections."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig, AutoModelForZeroShotObjectDetection, AutoProcessor

HERE = Path(__file__).resolve().parent
DETECTOR = HERE.parent
sys.path.insert(0, str(DETECTOR))
from pilot_detect import audit_weights, THRESHOLDS
from detect_cached_frames import merge_detections
from evaluate_pilot import REFERENCES, NEGATIVES


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.8)
    rows = read(HERE / "panel/candidates.json")
    delta = read(HERE / "delta/candidates.json")
    labels = read(HERE / "panel/config.json")["objects"]
    fridge_ids = set(REFERENCES["refrigerator"]) | set(NEGATIVES["refrigerator"])
    fridge_rows = [r for r in rows if r["source_frame_index"] in fridge_ids]
    jobs = [
        ("base-delta", labels, {}, delta),
        ("built-in", labels + ["built-in refrigerator"], {"built-in refrigerator": "refrigerator"}, rows),
        ("focus-refrigerator", ["refrigerator"], {}, fridge_rows),
        ("focus-built-in", ["built-in refrigerator"], {"built-in refrigerator": "refrigerator"}, fridge_rows),
    ]
    models = [
        ("base", "base-iwr", "IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"),
        ("mm", "mm-base-iwr", "openmmlab-community/mm_grounding_dino_base_all", "b32293f50a56f7218d9f73ff940ffb9eba84d63c"),
    ]
    for tag, checkpoint, name, revision in models:
        folder = HERE / tag
        folder.mkdir(exist_ok=True)
        checkpoint = DETECTOR / "models" / checkpoint
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint, config=config,
                    local_files_only=True).eval()
        weights = audit_weights(model, checkpoint / "model.safetensors")
        model.to("cuda:0")
        provenance = dict(model=name, revision=revision, weights=weights, thresholds=THRESHOLDS,
            input_size=dict(shortest_edge=640, longest_edge=1024), jobs={},
            selection="Frozen diagnostic panel and existing 1758 selected frames; focused prompts use the same 25 fridge diagnostic frames.")
        for mode, words, aliases, frames in jobs:
            output = folder / (mode + ".jsonl")
            if output.exists():
                raise FileExistsError(output)
            caption = ""
            spans = []
            for word in words:
                spans.append((len(caption), len(caption) + len(word)))
                caption += word + " . "
            caption = caption.strip()
            encoded = processor.tokenizer(caption, return_offsets_mapping=True)
            tokens = [[i for i, (x, y) in enumerate(encoded["offset_mapping"])
                       if y > x and x < end and y > begin] for begin, end in spans]
            assert all(tokens)
            started = time.monotonic()
            with output.open("x", encoding="utf-8") as handle:
                for i, row in enumerate(frames):
                    path = Path(row["frame"])
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["frame_sha256"]
                    im = Image.open(path).convert("RGB")
                    assert im.size == (640, 480)
                    inputs = processor(images=im, text=caption, return_tensors="pt",
                        size={"shortest_edge":640, "longest_edge":1024}).to("cuda:0")
                    assert inputs["input_ids"][0].tolist() == encoded["input_ids"]
                    with torch.inference_mode():
                        prediction = model(**inputs)
                    probs = prediction.logits[0].sigmoid().float().cpu().numpy()
                    raw = prediction.pred_boxes[0].float().cpu().numpy()
                    boxes = np.column_stack((raw[:,:2] - raw[:,2:]/2,
                                             raw[:,:2] + raw[:,2:]/2)) * [640,480,640,480]
                    means = np.stack([probs[:,idx].mean(1) for idx in tokens], 1)
                    peaks = np.stack([probs[:,idx].max(1) for idx in tokens], 1)
                    record = dict(source_frame_index=row["source_frame_index"],timestamp_s=row["timestamp_s"],
                        **{key:merge_detections(boxes,means,peaks,words,aliases,640,480,text,box)
                           for key,(text,box) in THRESHOLDS.items()})
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    if i % 20 == 0:
                        print(tag,mode,i+1,"/",len(frames),flush=True)
            provenance["jobs"][mode] = dict(caption=caption, frames=len(frames),
                seconds=round(time.monotonic()-started,2),source_frames=[r["source_frame_index"] for r in frames])
            (folder/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
            print("DONE",tag,mode,provenance["jobs"][mode]["seconds"],flush=True)
        del model
        torch.cuda.empty_cache()
    print("SEARCH_DONE",flush=True)


if __name__ == "__main__":
    main()
