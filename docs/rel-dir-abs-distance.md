# Relative direction and absolute distance

These are **view selection and input preparation**, downstream of reconstruction.
Supply a `.blend` and explicit target-to-mesh mappings using the existing question
schema. No AI, video inference, reconstruction or QA is performed by these commands.
All meshes belonging to each physical target must be included. Do not map a pot to
`bowl` merely because the detector proposed that label.

## Relative direction

1. Continuously solve a camera beside the standing object, looking toward the facing
   object. Both facing and queried objects must pass joint framing/visibility gates.
   The lens range is 12–35mm. Camera heading must stay within 25 degrees of the
   standing-to-facing direction.
2. Solve an overview showing all three objects. Output order is standing view first,
   overview second. No third relation-support view silently substitutes for failure.
3. Red = standing object; blue = facing object; green = queried object. The prompt
   describes the actual image order and asks for the question's reference frame,
   not screen-space left/right.

The default rejects incomplete direction inputs. A deliberately approved overview-only
experiment can use `--allow-overview-only`; its manifest preserves
`geometry_qa_ready=false` and labels the protocol as a fallback. This is not the
complete two-view method. The earlier ten-question pilot used this fallback on five
questions; its result is not a uniform two-image experiment.

## Absolute distance

The existing continuous joint-endpoint solver is retained. Both endpoints must be
visible together and pass the relation-projection and object-readability gates.
Red/blue identify the two endpoints; source positions, dimensions and scales are not
changed. The prompt follows the original question's distance definition and units.
No metric distance, GT or object coordinates are inserted in the model input.

## Run

In the question JSON, set `room.remove_walls: true`, enumerate `room.walls` separately
from `room.floor`, and provide the role fields (`standing/facing/query` or `endpoints`).
Walls are excluded before BVH construction/optimization and hidden during rendering.
Floors and support objects remain. The source blend is never saved over.

```bash
python scripts/select_views.py --blend scene.blend --video source.mp4 \
  --question question.json --out output --blender /path/to/blender --role-colors
python scripts/direct_views/relation_inputs.py --question question.json \
  --renders output/render_manifest.json --out output/model-input.json
```

Use the original public question file for the second command, including its options
and `qa_id`. The resulting JSON contains image paths, hashes, roles and the prompt;
it does not call a model. Role prompts require a manifest rendered with `--role-colors`.

## Evidence and limits

The November 9–10 pilot reused V8 scenes for 21 absolute-distance and ten direction
questions. Absolute-distance images were prepared for review, not evaluated in this
delivery. Direction QA and later fresh V8 attempts are separately packaged outside Git.
In a fresh 1889 rerun, different local reconstructions produced inconsistent floor
slabs that obscured part of the bin despite selector acceptance. In 1892, detector
peaks retained cooking-pot candidates while a lower-score pale candidate did not enter
the selected groups. These are unresolved upstream/geometry limits, not validated
accuracy improvements. Removing walls does not repair wrong identity, object placement,
overlapping floors or uncalibrated LingBot scale.

No videos, blends, experiment answers, subscription credentials or account identifiers
are shipped in this code update.
