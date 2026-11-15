"""Offline mathematical review only; no scene/engine/GPU or validation changes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import trimesh
SCENE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCENE))
from validate import winding, mesh_edges, mouth_structure, proper_segment_crossings
from runtime_ipc.staged_spatial_loop_geometry_regression import independent_minimum


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cap(vertices, directed_boundary, apex):
    v = np.vstack([vertices, apex])
    f = np.column_stack([directed_boundary[:, 1], directed_boundary[:, 0],
                         np.full(len(directed_boundary), len(vertices))])
    return v, f


def closed(vertices, body_faces, apex):
    _, _, boundary = mesh_edges(body_faces)
    v, f = cap(vertices, boundary, apex)
    return v, np.vstack([body_faces, f])


def jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x))


def run():
    output = {"scope": "Independent offline mathematical/synthetic review, not an executed insertion or replacement validator", "runtime_changed": False, "cases": []}
    cases = output["cases"]
    cube = trimesh.creation.box(extents=(2, 2, 2))
    v = np.asarray(cube.vertices)
    body = np.asarray(cube.faces)[~np.all(v[cube.faces, 2] == 1., axis=1)]
    queries = np.array([[0., 0., 0.], [2., 0., 0.], [0., 0., -2.], [.9, .9, .8]])
    apices = [np.array([0., 0., 1.]), np.array([.25, .1, 3.]), np.array([-3., 2., 1.25])]
    values = [winding(queries, *closed(v, body, a)) for a in apices]
    delta = float(np.max(np.abs(values - np.asarray(values)[0])))
    assert delta < 1e-12 and np.allclose(np.abs(values[0]), [1., 0., 0., 1.], atol=1e-12)
    cases.append({"name": "closed_cup_low_queries_equal_for_three_high_caps", "pass": True, "query_winding": values, "maximum_difference": delta, "all_queries_below_rim": True})

    bow = np.array([[-1., -1., 1.], [1., 1., 1.], [-1., 1., 1.1], [1., -1., 1.1]])
    edges = np.column_stack([np.arange(4), np.roll(np.arange(4), -1)])
    q = np.array([[0., 0., 0.], [.2, .2, .9], [-.7, .3, -.5], [3., 0., .99]])
    high_apices = [bow.mean(0), bow.mean(0) + [0., 0., 2.], np.array([4., -3., 1.2])]
    cap_values = np.asarray([winding(q, *cap(bow, edges, a)) for a in high_apices])
    delta = float(np.max(np.abs(cap_values - cap_values[0])))
    _, _, basis = np.linalg.svd(bow-bow.mean(0), full_matrices=False)
    crossing = proper_segment_crossings((bow-bow.mean(0)) @ basis[:2].T)
    separation = independent_minimum(bow)
    assert crossing == 1 and separation > .09 and delta < 1e-12
    cases.append({"name": "simple_spatial_rim_with_crossed_svd_projection", "pass": True, "svd_projection_crossings": crossing, "minimum_nonadjacent_3d_edge_separation": separation, "cap_only_winding": cap_values, "maximum_difference": delta, "body_embedding_not_claimed": True})

    q_above = np.array([[0., 0., 2.]])
    w_flat = winding(q_above, *closed(v, body, apices[0]))
    w_high = winding(q_above, *closed(v, body, np.array([0., 0., 3.])))
    assert np.isclose(abs(float(w_high[0]-w_flat[0])), 1.)
    cases.append({"name": "counterexample_query_above_lowest_rim", "pass": True, "flat_cap_winding": w_flat, "high_cap_winding": w_high, "conclusion": "Cap choice can change membership when the query-height condition fails"})

    q_low = np.array([[0., 0., .3]])
    w_flat = winding(q_low, *closed(v, body, apices[0]))
    w_bad = winding(q_low, *closed(v, body, np.array([0., 0., -.5])))
    assert np.isclose(abs(float(w_flat[0]-w_bad[0])), 1.)
    cases.append({"name": "counterexample_cap_dips_below_rim", "pass": True, "high_cap_winding": w_flat, "low_apex_cap_winding": w_bad, "conclusion": "Both caps, including every triangle interior, must remain above the separating plane"})

    vv, ff = closed(v, body, apices[0])
    broken = ff.copy(); broken[-1] = broken[-1, ::-1]
    mesh = trimesh.Trimesh(vv, broken, process=False)
    bad_w = float(winding(np.array([[.1, .2, 0.]]), vv, broken)[0])
    assert mesh.is_watertight and not mesh.is_winding_consistent and abs(bad_w-round(bad_w)) > 1e-3
    cases.append({"name": "counterexample_reversed_single_cap_face", "pass": True, "is_watertight": mesh.is_watertight, "is_winding_consistent": mesh.is_winding_consistent, "query_winding": bad_w})

    a = trimesh.creation.box(extents=(2, 2, 2)); b = a.copy(); b.apply_translation([.5, 0., 0.])
    overlap = trimesh.util.concatenate([a, b]); multi = float(winding(np.array([[.2, 0., 0.]]), overlap.vertices, overlap.faces)[0])
    assert overlap.is_watertight and overlap.is_winding_consistent and np.isclose(abs(multi), 2.)
    cases.append({"name": "counterexample_combinatorial_closedness_does_not_prove_embedding", "pass": True, "is_watertight": overlap.is_watertight, "is_winding_consistent": overlap.is_winding_consistent, "query_winding": multi, "fixture_components": 2, "conclusion": "A closed coherently oriented but geometrically crossing surface may have degree two; abs(w)>0.5 alone is insufficient evidence of a simple physical interior"})

    q_near = np.array([[.27, .19, 1.-2e-3], [.27, .19, 1.-1e-6], [.27, .19, 1.-1e-9]])
    near = np.asarray([winding(q_near, *closed(v, body, aa)) for aa in apices])
    delta = float(np.max(np.abs(near-near[0])))
    assert delta < 1e-7 and np.allclose(near, 1., atol=1e-7)
    cases.append({"name": "numerical_low_plane_gaps_on_small_fixture", "pass": True, "gaps": [2e-3, 1e-6, 1e-9], "maximum_difference": delta, "production_tolerance_not_established_by_fixture": True})

    run_dir = SCENE / 'runs/two_panel_full_14'
    config = json.loads((run_dir/'config.json').read_text())
    rest = np.load(run_dir/'rest.npz'); frame = np.load(run_dir/'lower_teddy_to_rim.npz')
    body_faces, rim, topology = mouth_structure(rest['bag'], rest['bag_faces'], config)
    bag, bear = frame['bag'], frame['bear']; rim_points = bag[rim]
    h = float(rim_points[:, 2].min()); gap = float(h-bear[:, 2].max())
    low_mask = bear[:, 2] <= h-.002
    _, _, boundary = mesh_edges(body_faces)
    aa = rim_points.mean(0); bb = aa + np.array([.05, .01, .15])
    actual_cap = np.asarray([winding(bear, *cap(bag, boundary, x)) for x in [aa, bb]])
    diff = np.abs(actual_cap[1]-actual_cap[0])
    low_delta = float(diff[low_mask].max()) if low_mask.any() else None
    assert low_mask.any() and low_delta < 1e-12 and gap < 0.
    _, _, basis = np.linalg.svd(rim_points-aa, full_matrices=False)
    output['actual_full14_applicability'] = {
        'time_s': float(frame['time']), 'bear_vertex_count': len(bear),
        'rim_min_z_m': h, 'bear_max_z_m': float(bear[:, 2].max()),
        'minimum_whole_bear_height_gap_m': gap, 'all_bear_queries_below_rim': False,
        'svd_projection_crossings': proper_segment_crossings((rim_points-aa) @ basis[:2].T),
        'queries_at_least_2mm_below_rim': int(low_mask.sum()),
        'maximum_cap_contribution_difference_on_low_subset': low_delta,
        'maximum_cap_contribution_difference_on_all_bear_vertices': float(diff.max()),
        'theorem_certifies_full14_containment': False,
        'topology': topology,
        'not_an_insertion_success': True,
        'input_sha256': {str(x.relative_to(SCENE)): digest(x) for x in [SCENE/'validate.py', run_dir/'rest.npz', run_dir/'lower_teddy_to_rim.npz', run_dir/'config.json']}}
    output['all_synthetic_cases_passed'] = all(x['pass'] for x in cases)
    output['synthetic_case_count'] = len(cases)
    output['physical_runtime_tests_performed'] = 0
    output['theorem_scope'] = 'Pointwise cap independence strictly below all cap vertices; physical interior requires additional embedding/orientation and separation conditions.'
    target = SCENE/'runtime_ipc/low-rim-cap-independence-review.json'
    target.write_text(json.dumps(output, indent=2, default=jsonable)+'\n')
    print(json.dumps({'output': str(target), 'cases': len(cases), 'all_pass': output['all_synthetic_cases_passed'], 'actual_full14_height_gap_m': gap, 'actual_full14_low_subset_count': int(low_mask.sum()), 'actual_full14_low_subset_cap_difference': low_delta}, indent=2))

if __name__ == '__main__':
    run()
