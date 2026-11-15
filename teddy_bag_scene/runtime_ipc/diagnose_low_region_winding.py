"""Read-only saved-state low-region winding diagnostic, never a task validator.

No Genesis import, scene construction, GPU, input mutation, or validation override.
The optional --output writes only an explicitly requested standalone report.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import trimesh

SCENE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCENE))
from validate import mesh_edges, mouth_structure, proper_segment_crossings, signed_volumes, winding

HEIGHT_MARGIN_M = .002
RIM_NUMERICAL_GUARD_M = 1e-8
BODY_AMBIGUITY_DISTANCE_M = 1e-6
INTEGER_RESIDUAL_TOLERANCE = 1e-6
CAP_AGREEMENT_TOLERANCE = 1e-8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serialize(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value))


def segment_distance(a, b, c, d):
    """Independent constrained quadratic minimum over two finite segments."""
    u, v, w = b-a, d-c, a-c
    aa, bb, ee, cc, ff = (float(u@u), float(u@v), float(v@v), float(u@w), float(v@w))
    candidates = []
    for s in (0., 1.):
        candidates.append((s, float(np.clip((ff+bb*s)/ee, 0., 1.)) if ee else 0.))
    for t in (0., 1.):
        candidates.append((float(np.clip((bb*t-cc)/aa, 0., 1.)) if aa else 0., t))
    det = aa*ee-bb*bb
    if det > 16*np.finfo(float).eps*aa*ee:
        s, t = (bb*ff-cc*ee)/det, (aa*ff-bb*cc)/det
        if 0. <= s <= 1. and 0. <= t <= 1.:
            candidates.append((s, t))
    return min(float(np.linalg.norm(a+s*u-c-t*v)) for s, t in candidates)


def spatial_rim(points, rest_points, radius):
    lengths = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
    reference_lengths = np.linalg.norm(np.roll(rest_points, -1, axis=0)-rest_points, axis=1)
    minimum, pair, pair_count = float('inf'), None, 0
    for i in range(len(points)):
        for j in range(i+2, len(points)):
            if i == 0 and j == len(points)-1: continue
            value = segment_distance(points[i], points[(i+1)%len(points)], points[j], points[(j+1)%len(points)])
            pair_count += 1
            if value < minimum: minimum, pair = value, [i, j]
    required = max(.0006, 2*radius)
    return {
        'vertex_count': len(points), 'nonadjacent_edge_pair_count': pair_count,
        'minimum_current_edge_length_m': float(lengths.min()),
        'minimum_rest_edge_length_m': float(reference_lengths.min()),
        'minimum_nonadjacent_3d_edge_distance_m': minimum,
        'minimum_nonadjacent_3d_edge_pair': pair,
        'edge_distance_numerical_guard_m': RIM_NUMERICAL_GUARD_M,
        'required_nonadjacent_3d_edge_separation_m': required,
        'minimum_nonadjacent_3d_edge_distance_lower_bound_m': minimum-RIM_NUMERICAL_GUARD_M,
        'passed': bool(min(lengths.min(), reference_lengths.min()) > 1e-8 and
                       minimum-RIM_NUMERICAL_GUARD_M > required)}


def capped_surface(bag, body_faces, directed_boundary, apex):
    vertices = np.vstack([bag, apex])
    caps = np.column_stack([directed_boundary[:, 1], directed_boundary[:, 0],
                            np.full(len(directed_boundary), len(bag))])
    return vertices, np.vstack([body_faces, caps])


def diagnose(run, state=None, index=-1):
    run = Path(run).expanduser().resolve()
    state_path = run/'states.npz' if state is None else Path(state).expanduser()
    if not state_path.is_absolute(): state_path = run/state_path
    state_path = state_path.resolve()
    config_path, rest_path = run/'config.json', run/'rest.npz'
    config = json.loads(config_path.read_text())
    with np.load(rest_path, allow_pickle=False) as archive:
        rest = {key: archive[key] for key in archive.files}
    with np.load(state_path, allow_pickle=False) as archive:
        batch = archive['bag'].ndim == 3
        bag = np.asarray(archive['bag'][index] if batch else archive['bag'], float)
        bear = np.asarray(archive['bear'][index] if batch else archive['bear'], float)
        timestamp = float(archive['time'][index] if batch else archive['time'])
    report = {
        'status': 'input_rejected', 'scope': 'Pointwise cap-independent low-region winding diagnostic only',
        'run': str(run), 'state': str(state_path), 'state_index': index if batch else None,
        'time_s': timestamp, 'physical_task_success_certified': False,
        'does_not_replace_original_validator': True,
        'does_not_certify_physical_body_nonintersection': True,
        'does_not_replace_final_contact_release_or_stability_checks': True,
        'continuous_whole_bear_containment_certified': False,
        'all_bear_vertices_inside_by_guarded_unit_winding': None,
        'bear_com_inside_by_guarded_unit_winding': None,
        'all_query_points_inside_by_guarded_unit_winding': None,
        'thresholds': {'minimum_low_region_height_margin_m': HEIGHT_MARGIN_M,
            'physical_body_near_surface_ambiguity_distance_m': BODY_AMBIGUITY_DISTANCE_M,
            'integer_residual_tolerance': INTEGER_RESIDUAL_TOLERANCE,
            'cap_agreement_tolerance': CAP_AGREEMENT_TOLERANCE,
            'tolerance_scope': 'Explicit diagnostic numerical guards, not a relaxation of physical contact or penetration limits'},
        'input_sha256': {str(p): sha256(p) for p in [config_path, rest_path, state_path, SCENE/'validate.py', Path(__file__).resolve()]},
        'reasons': []}
    if (bag.shape != rest['bag'].shape or bear.shape != rest['bear'].shape or
            not np.isfinite(bag).all() or not np.isfinite(bear).all() or not np.isfinite(timestamp)):
        report['reasons'].append('Invalid finite matching actual state geometry')
        return report
    body_faces, rim, topology = mouth_structure(rest['bag'], rest['bag_faces'], config)
    report['original_body_topology'] = topology
    if len(rim) < 4 or len(np.unique(rim)) != len(rim):
        report['reasons'].append('No unique original physical body rim')
        return report
    tets = np.asarray(rest['bear_tets'], int)
    weights = np.abs(signed_volumes(rest['bear'], tets))
    if not np.isfinite(weights).all() or weights.min() <= 0:
        report['reasons'].append('Invalid positive rest mass weights')
        return report
    com = np.sum(bear[tets].mean(1)*weights[:, None], axis=0)/weights.sum()
    queries = np.vstack([bear, com])
    points, reference = bag[rim], rest['bag'][rim]
    h = float(points[:, 2].min()); gap = float(h-queries[:, 2].max())
    report['height_precondition'] = {
        'actual_rim_min_z_m': h, 'actual_rim_max_z_m': float(points[:, 2].max()),
        'actual_bear_max_z_m': float(bear[:, 2].max()), 'actual_com_m': com.tolist(),
        'minimum_query_height_gap_m': gap,
        'required_height_gap_m': HEIGHT_MARGIN_M,
        'all_bear_vertices_and_com_at_least_required_margin_below_rim': bool(gap >= HEIGHT_MARGIN_M),
        'query_count': len(queries),
        'queries_below_rim_with_required_margin': int((h-queries[:, 2] >= HEIGHT_MARGIN_M).sum())}
    if gap < HEIGHT_MARGIN_M:
        report['status'] = 'not_applicable'
        report['reasons'].append('At least one actual teddy vertex or COM is not 2 mm below the lowest actual rim')
        report['winding_computed'] = False
        return report
    report['winding_computed'] = True
    _, _, boundary = mesh_edges(body_faces)
    body_mesh = trimesh.Trimesh(bag, body_faces, process=False)
    cloth_mesh = trimesh.Trimesh(bag, rest['bag_faces'], process=False)
    area2 = np.linalg.norm(np.cross(bag[body_faces[:, 1]]-bag[body_faces[:, 0]], bag[body_faces[:, 2]]-bag[body_faces[:, 0]]), axis=1)
    boundary_matches = set(boundary.ravel()) == set(rim) and len(boundary) == len(rim)
    topology_ok = bool(topology.get('two_panels_directly_sewn_along_u') and
        topology.get('actual_open_body_boundary') and topology.get('connected_components') == 1 and
        topology.get('nonmanifold_edges') == 0 and body_mesh.is_winding_consistent and
        cloth_mesh.is_winding_consistent and boundary_matches and area2.min() > 1e-14)
    report['actual_topology_orientation'] = {
        'passed': topology_ok, 'body_winding_consistent': bool(body_mesh.is_winding_consistent),
        'full_cloth_winding_consistent': bool(cloth_mesh.is_winding_consistent),
        'body_directed_boundary_matches_rim': boundary_matches,
        'minimum_actual_body_triangle_twice_area_m2': float(area2.min()),
        'geometric_embedding_not_proven_by_these_flags': True}
    if not topology_ok: report['reasons'].append('Original body topology, orientation, boundary cancellation, or nondegeneracy check failed')
    geometry = spatial_rim(points, reference, float(config['bag']['thickness']))
    report['actual_spatial_rim'] = geometry
    if not geometry['passed']: report['reasons'].append('Actual nonadjacent 3D rim edge separation or nondegeneracy check failed')
    center = points.mean(0)
    _, _, basis = np.linalg.svd(points-center, full_matrices=False)
    projected = (points-center)@basis[:2].T
    report['projection_diagnostics_only'] = {
        'svd_projection_crossings': proper_segment_crossings(projected),
        'xy_projection_crossings': proper_segment_crossings(points[:, :2]),
        'svd_projected_signed_area_abs_m2': float(abs(np.sum(projected[:,0]*np.roll(projected[:,1],-1)-projected[:,1]*np.roll(projected[:,0],-1)))/2),
        'svd_plane_max_nonplanarity_m': float(np.abs((points-center)@basis[2]).max()),
        'not_used_as_low_region_rejection': True}
    apices = [center, center+np.array([.031, -.023, .15])]
    values, closure_flags = [], []
    for apex in apices:
        vv, ff = capped_surface(bag, body_faces, boundary, apex)
        mesh = trimesh.Trimesh(vv, ff, process=False)
        closure_flags.append({'is_watertight': bool(mesh.is_watertight),
            'is_winding_consistent': bool(mesh.is_winding_consistent),
            'cap_apex_m': apex.tolist(), 'cap_minimum_vertex_z_m': float(min(h, apex[2])),
            'analysis_cap_not_simulated': True})
        values.append(winding(queries, vv, ff))
    mean_w, high_w = values
    residuals = np.maximum(np.abs(mean_w-np.rint(mean_w)), np.abs(high_w-np.rint(high_w)))
    difference = np.abs(mean_w-high_w)
    finite_w = bool(np.isfinite(mean_w).all() and np.isfinite(high_w).all())
    multiple = (np.abs(np.rint(mean_w)) > 1) | (np.abs(np.rint(high_w)) > 1)
    unit = ((np.abs(np.abs(mean_w)-1.) <= INTEGER_RESIDUAL_TOLERANCE) &
            (np.abs(np.abs(high_w)-1.) <= INTEGER_RESIDUAL_TOLERANCE))
    closed_ok = all(x['is_watertight'] and x['is_winding_consistent'] for x in closure_flags)
    agreement_ok = bool(finite_w and difference.max() <= CAP_AGREEMENT_TOLERANCE)
    integer_ok = bool(finite_w and residuals.max() <= INTEGER_RESIDUAL_TOLERANCE)
    if not closed_ok: report['reasons'].append('Capped directed triangle chain is not coherently closed')
    if not agreement_ok: report['reasons'].append('Two high caps disagree or yield nonfinite winding')
    if not integer_ok: report['reasons'].append('Winding integer residual exceeds explicit numerical tolerance')
    if multiple.any(): report['reasons'].append('At least one query has multiple winding magnitude greater than one')
    report['cap_winding'] = {
        'query_order': 'All actual bear vertices in saved order, followed by actual mass COM',
        'mean_apex_surface': closure_flags[0], 'high_apex_surface': closure_flags[1],
        'mean_apex_raw_winding': mean_w.tolist(), 'high_apex_raw_winding': high_w.tolist(),
        'absolute_cap_difference': difference.tolist(),
        'maximum_absolute_cap_difference': float(difference.max()), 'caps_agree': agreement_ok,
        'per_query_maximum_integer_residual': residuals.tolist(),
        'maximum_integer_residual': float(residuals.max()), 'integer_residual_passed': integer_ok,
        'multiple_winding_query_indices': np.flatnonzero(multiple).tolist(),
        'non_unit_winding_query_indices': np.flatnonzero(~unit).tolist(),
        'raw_all_bear_vertices_have_unit_winding': bool(unit[:-1].all()),
        'raw_com_has_unit_winding': bool(unit[-1])}
    body_closest, body_distance, body_triangles = trimesh.proximity.closest_point(body_mesh, queries)
    _, cloth_distance, cloth_triangles = trimesh.proximity.closest_point(cloth_mesh, queries)
    nearest = int(np.argmin(body_distance))
    separated = bool(np.isfinite(body_distance).all() and body_distance.min() > BODY_AMBIGUITY_DISTANCE_M)
    report['physical_cloth_query_distance'] = {
        'method': 'Exact closest triangle point for each queried bear vertex and COM; no continuous whole-bear certificate',
        'body_distance_m': body_distance.tolist(), 'full_cloth_distance_m': cloth_distance.tolist(),
        'minimum_body_distance_m': float(body_distance.min()),
        'minimum_full_cloth_distance_m': float(cloth_distance.min()),
        'minimum_body_distance_query_index': nearest,
        'minimum_body_distance_body_face_index': int(body_triangles[nearest]),
        'minimum_body_distance_closest_point_m': body_closest[nearest].tolist(),
        'minimum_full_cloth_distance_face_index': int(cloth_triangles[np.argmin(cloth_distance)]),
        'near_physical_body_ambiguity_query_indices': np.flatnonzero(body_distance <= BODY_AMBIGUITY_DISTANCE_M).tolist(),
        'queries_separated_from_body_at_numerical_guard': separated,
        'does_not_replace_cloth_into_teddy_penetration_audit': True}
    if not separated: report['reasons'].append('One or more query points are numerically too close to actual physical body triangles')
    guards = topology_ok and geometry['passed'] and closed_ok and agreement_ok and integer_ok and not multiple.any() and separated
    report['all_bear_vertices_inside_by_guarded_unit_winding'] = bool(guards and unit[:-1].all())
    report['bear_com_inside_by_guarded_unit_winding'] = bool(guards and unit[-1])
    report['all_query_points_inside_by_guarded_unit_winding'] = bool(guards and unit.all())
    if not unit.all(): report['reasons'].append('At least one teddy vertex or COM does not have unit winding in both caps')
    report['status'] = 'passed_point_diagnostic' if guards and unit.all() else 'failed_point_diagnostic'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--state', type=Path, help='NPZ absolute path or path relative to run; default states.npz')
    parser.add_argument('--index', type=int, default=-1, help='Frame index when state NPZ contains a batch')
    parser.add_argument('--output', type=Path, help='Explicit report destination; inputs are never modified')
    args = parser.parse_args()
    report = diagnose(args.run, args.state, args.index)
    if args.output:
        if args.output.resolve() in {Path(p).resolve() for p in report['input_sha256']}:
            raise ValueError('Output must not replace any input or source file')
        if args.output.suffix.lower() != '.json':
            raise ValueError('Explicit diagnostic output must be a JSON report')
        args.output.write_text(json.dumps(report, indent=2, default=serialize)+'\n')
        print(json.dumps({k:report.get(k) for k in ['status', 'time_s', 'height_precondition', 'all_query_points_inside_by_guarded_unit_winding', 'reasons']}, indent=2))
        print('Report: '+str(args.output.resolve()))
    else:
        print(json.dumps(report, indent=2, default=serialize))

if __name__ == '__main__':
    main()
