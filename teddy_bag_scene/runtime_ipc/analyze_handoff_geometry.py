"""Read-only material-labelled geometry of an actual native handoff probe.

This never substitutes a PCA axis for a material width direction and never
modifies a saved state. Projections are diagnostics, not insertion certificates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validate import boundary_loops, proper_segment_crossings


def line(a, b):
    v = b - a
    return dict(from_m=a.tolist(), to_m=b.tolist(), vector_m=v.tolist(),
                length_m=float(np.linalg.norm(v)),
                xy_yaw_deg=float(np.degrees(np.arctan2(v[1], v[0]))))


def analyze(run):
    config = json.loads((run / 'config.json').read_text())
    rest = np.load(run / 'rest.npz')
    source_path = run / 'source/assets' / config['bag']['npz_file']
    source = np.load(source_path)
    error, mapping = cKDTree(rest['bag']).query(
        source['points'] + np.asarray(config['bag']['position']))
    if len(source['points']) != len(rest['bag']) or error.max() > 1e-7 or len(np.unique(mapping)) != len(mapping):
        raise ValueError('Non-bijective source-to-native cloth mapping')
    mapped_faces = {tuple(row) for row in np.sort(mapping[source['faces']], axis=1)}
    native_faces = {tuple(row) for row in np.sort(rest['bag_faces'], axis=1)}
    if (len(mapped_faces) != len(source['faces'])
            or len(native_faces) != len(rest['bag_faces']) or mapped_faces != native_faces):
        raise ValueError('Source and native cloth triangles differ')
    faces = mapping[source['faces'][source['body_face_indices']]]
    loops, manifold = boundary_loops(faces)
    if not manifold or len(loops) != 1 or set(loops[0]) != set(mapping[source['rim']]):
        raise ValueError('Expected the physical two-panel body mouth')
    rim_ids = loops[0]
    phases = ['close_both_handles', 'drape_under_gravity', 'spread_bag',
              'orient_held_bag', 'release_right_handle', 'right_clear_handle',
              'settle_handoff']
    initial = np.load(run / 'initial.npz')
    records = [('initial', initial)]
    records += [(name, np.load(run / (name + '.npz')))
                for name in phases if (run / (name + '.npz')).exists()]
    rows = []
    for name, state in records:
        bag = state['bag']
        rim = bag[rim_ids]
        left = bag[mapping[source['seam_left']]]
        right = bag[mapping[source['seam_right']]]
        centered = rim - rim.mean(0)
        _, singular, basis = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ basis[:2].T
        xy = rim[:, :2]
        row = dict(phase=name, time_s=float(state['time']),
                   rim_bbox_min_m=rim.min(0).tolist(), rim_bbox_max_m=rim.max(0).tolist(),
                   rim_span_xyz_m=np.ptp(rim, axis=0).tolist(),
                   material_width_lines={k: line(left[i], right[i])
                       for k, i in [('bottom', 0), ('middle', len(left)//2), ('top', -1)]},
                   pca_projection=dict(dimensions_m=np.ptp(projected, axis=0).tolist(),
                       singular_values=singular.tolist(), basis_world=basis.tolist(),
                       crossings=int(proper_segment_crossings(projected)),
                       absolute_signed_area_m2=float(abs(np.sum(projected[:, 0]*np.roll(projected[:, 1], -1)
                           - projected[:, 1]*np.roll(projected[:, 0], -1)))/2)),
                   xy_projection=dict(crossings=int(proper_segment_crossings(xy)),
                       absolute_signed_area_m2=float(abs(np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)
                           - xy[:, 1]*np.roll(xy[:, 0], -1)))/2)),
                   handle_roots={})
        for panel in ('front', 'back'):
            a = bag[mapping[source[f'handle_{panel}_root_left']]].mean(0)
            b = bag[mapping[source[f'handle_{panel}_root_right']]].mean(0)
            row['handle_roots'][panel] = line(a, b)
        if 'tcp_left' in state:
            group = config['task']['handle_grasps']['left']['group']
            distances = np.linalg.norm(bag[mapping[source[group]]] - state['tcp_left'], axis=1)
            row['left_named_material_distance_m'] = dict(min=float(distances.min()), max=float(distances.max()))
        rows.append(row)
    result = dict(kind='actual_native_handoff_material_geometry',
                  source_to_native_mapping_max_error_m=float(error.max()),
                  source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  simulation_state_modified=False, insertion_success_inferred=False,
                  initial_to_rest_max_distance_m=float(np.linalg.norm(initial['bag']-rest['bag'], axis=1).max()),
                  scope='Saved actual phase endpoints only. Material line lengths are endpoint chords, not fabric arc lengths or free aperture widths. PCA and projection area do not establish a free insertion passage.',
                  phases=rows)
    out = run / 'handoff-material-geometry.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(out)
    for row in rows:
        print(row['phase'], 't=', round(row['time_s'], 3),
              'top seam width mm=', round(row['material_width_lines']['top']['length_m']*1000, 2),
              'rim xyz mm=', np.round(np.asarray(row['rim_span_xyz_m'])*1000, 2).tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    analyze(parser.parse_args().run.resolve())
