"""Lossy VLM transport only; full-precision camera references remain authoritative."""
import hashlib
import json
from pathlib import Path


def build_transport(reference, decimals=3):
    if not 0 <= decimals <= 8:
        raise ValueError('Invalid pose precision')
    compact = []
    centers = []
    for row in reference['rows']:
        c, k = row['c2w_blender'], row['intrinsics_640x480']
        values = ([c[i][3] for i in range(3)] + [c[i][1] for i in range(3)]
                  + [-c[i][2] for i in range(3)] + [k[0][0], k[1][1]])
        compact.append([round(float(v), decimals) + 0.0 for v in values])
        centers.append([round(float(k[0][2]), decimals), round(float(k[1][2]), decimals)])
    if not compact:
        raise ValueError('Empty pose reference')
    shared = all(c == centers[0] for c in centers)
    fields = ['px', 'py', 'pz', 'ux', 'uy', 'uz', 'fx', 'fy', 'fz', 'focal_x', 'focal_y']
    if not shared:
        fields += ['cx', 'cy']
        compact = [r + c for r, c in zip(compact, centers)]
    explanation = ('Camera row follows each image: [' + ','.join(fields)
                   + ']. p=world position; u=up; f=forward; right=f cross u. '
                   'Focal lengths and principal point are in pixels. ')
    if shared:
        explanation += 'Shared [cx,cy]=' + json.dumps(centers[0], separators=(',', ':')) + '. '
    explanation += 'Values are rounded to ' + str(decimals) + ' decimals for input only.'
    return dict(format='compact-camera-v1', decimals=decimals, explanation=explanation,
                fields=fields, shared_principal_point=centers[0] if shared else None,
                rows=compact,
                frame_mapping=[dict(image_index=i, timestamp_s=r['timestamp_s'],
                                    source_frame_index=r['source_frame_index'])
                               for i, r in enumerate(reference['rows'])])


def write_transport(source, destination):
    source = Path(source)
    if source.resolve() == Path(destination).resolve():
        raise ValueError('Never overwrite the full-precision source')
    result = build_transport(json.loads(source.read_text()))
    result.update(source_reference_path=str(source.resolve()),
                  source_reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  purpose='VLM input only; use source full-precision rows for rendering/comparison')
    Path(destination).write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        raise ValueError('Never overwrite the full-precision source')
    write_transport(args.source, args.destination)
