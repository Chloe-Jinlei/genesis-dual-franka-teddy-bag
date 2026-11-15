"""Exactly one +25deg frozen post-lift yaw and two declared COM candidates.

Saved full25 native lift supplies the initial geometry. Everything after yaw
is explicitly hypothetical frozen rigid grasp, not a native observation.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time

import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/two_panel_full_25'
OUT = RUN/'independent-plus25-two-offset-review'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(RUN/'source'))
import postlift_yaw
from contact_insertion import plan_guarded_contact_trial, reobserve_guarded_contact_trial, _serial, _mouth, _section_envelopes
from contact_slice_geometry import convex_hull_metrics
from motion import Planner

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = [RUN/n for n in ['config.json', 'rest.npz', 'lift_teddy.npz', 'postlift_yaw_plan.npz', 'postlift_yaw_candidate.json', 'input_sha256.json']]
inputs += sorted((RUN/'source').glob('*.py')) + sorted((RUN/'source/assets').iterdir())
before = {str(p): sha(p) for p in inputs if p.is_file()}
production = {str(p): sha(p) for p in [*sorted(ROOT.glob('*.py')), ROOT/'config.json', ROOT/'config_gravity_seating_trial.json']}
cfg = json.loads((RUN/'config.json').read_text())
rest = dict(np.load(RUN/'rest.npz', allow_pickle=False))
actual = dict(np.load(RUN/'lift_teddy.npz', allow_pickle=False))
saved_yaw = json.loads((RUN/'postlift_yaw_candidate.json').read_text())
archive = dict(np.load(RUN/'postlift_yaw_plan.npz', allow_pickle=False))
assert abs(float(actual['time'])-float(saved_yaw['observed_time_s'])) < 1e-9
command = {s: PPoly(archive[s+'_c'], archive[s+'_x'])(0.) for s in ('left', 'right')}
assert all(np.array_equal(command[s], saved_yaw['command_source_poses'][s]['q']) for s in command)
config = copy.deepcopy(cfg)
config['task']['contact_trial']['postlift_yaw'] = {'world_z_degrees': 25., 'duration_s': 6.}
config['robots']['mjcf_path'] = str(RUN/'source/assets/panda_tactile.xml')
for name in ('npz_file', 'metadata'):
    config['bag'][name] = str(RUN/'source/assets'/Path(config['bag'][name]).name)
planner = Planner(config)
p0, R0 = planner.fk('right', actual['q_right'][:7])
ideal_rotation = Rotation.from_euler('z', 25., degrees=True)
target_R = ideal_rotation*R0
ideal_bear = ideal_rotation.apply(actual['bear']-actual['tcp_right'])+actual['tcp_right']
polygon, _, mouth = _mouth(rest, actual, config)
eig, vectors = np.linalg.eigh(np.cov(polygon.T)); long = vectors[:, eig.argmax()]
if long[1] < 0: long = -long
across = np.array([long[1], -long[0]])


def dimensions(points):
    hull = convex_hull_metrics(points[:, :2])
    return {'axis_aligned_extent_m': np.ptp(points, axis=0),
            'whole_bear_footprint_hull_area_m2': hull['area'], 'whole_bear_footprint_hull_perimeter_m': hull['perimeter'],
            'extent_across_rim_principal_axis_m': float(np.ptp(points[:, :2]@across)),
            'extent_along_rim_principal_axis_m': float(np.ptp(points[:, :2]@long))}


sections, envelopes = _section_envelopes(ideal_bear, rest['bear_tets'])
report = {'kind': 'bounded_full25_actual_lift_plus25_two_offset_frozen_review', 'physical_success': False,
          'native_execution_performed': False, 'source_native_lift_time_s': float(actual['time']),
          'exact_input_sha256': before, 'production_sha256_before': production,
          'yaw_degrees': 25., 'declared_only_offsets_xy_m': [[0., 0.], [0., .03075]],
          'command_source': 'Exact first-knot motor command from full25 archived native postlift curve; cross-checked to saved candidate command_source_poses.',
          'source_actual_rotation_xyzw': R0.as_quat(), 'target_full_rotation_xyzw': target_R.as_quat(),
          'actual_lift_geometry': dimensions(actual['bear']), 'ideal_plus25_geometry': dimensions(ideal_bear),
          'mouth': mouth,
          'rim_principal_axis_geometry': {'long_axis_xy': long, 'across_axis_xy': across,
             'long_axis_heading_degrees_mod180': float(np.degrees(np.arctan2(long[1], long[0])) % 180),
             'projected_extent_across_axis_m': float(np.ptp(polygon@across)),
             'projected_extent_along_axis_m': float(np.ptp(polygon@long)),
             'scope': 'Geometric projected rim principal direction and extents, not material cloth width or a free passage certificate.'},
          'ideal_all_tetra_section_statistics': {'tetrahedron_count': len(rest['bear_tets']), 'vertex_height_event_count': len(sections),
             'enclosing_slab_count': len(envelopes), 'maximum_event_perimeter_m': max(s['perimeter'] for s in sections),
             'maximum_slab_perimeter_m': max(s['perimeter'] for s in envelopes),
             'maximum_slab_across_axis_extent_m': max(float(np.ptp(s['hull_xy']@across)) for s in envelopes)},
          'status': 'evaluating_original_yaw_gates', 'candidates': [],
          'limitations': ['Only the lift source is an actual native observation. All post-yaw/transport/fresh states are frozen rigid-grasp hypotheses.',
             'Rotating a saved teddy does not predict deformable grasp settling, slip, contact forces, or the final dynamic orientation.',
             'Geometry extents and all-tetra section envelopes do not certify passage through deforming cloth or handles.',
             'No source, default or trial config is changed; no additional angles or offsets are searched.',
             'Native force, torque, J, strain, held material, release eligibility and final whole-teddy containment remain required.']}


def save():
    report['original_inputs_unchanged'] = all(sha(p) == h for p, h in before.items())
    report['production_unchanged'] = all(sha(p) == h for p, h in production.items())
    (OUT/'review.json').write_text(json.dumps(_serial(report), indent=2, allow_nan=False)+'\n')


save()
started = time.monotonic()
yaw, yaw_plan = postlift_yaw.plan_postlift_yaw(rest, actual, config, command)
(OUT/'yaw_candidate.json').write_text(json.dumps(yaw, indent=2, allow_nan=False)+'\n')
report['yaw_preflight'] = yaw; report['yaw_cpu_wall_seconds'] = time.monotonic()-started
if yaw_plan is None:
    report['status'] = 'rejected_original_yaw_preflight'; save()
    print(json.dumps({'status': report['status'], 'rejections': yaw['rejections']}, indent=2), flush=True)
    raise SystemExit(0)
np.savez_compressed(OUT/'hypothetical_yaw_plan.npz', duration=yaw_plan.duration,
                    **{s+'_x': yaw_plan.curves[s].x for s in command}, **{s+'_c': yaw_plan.curves[s].c for s in command})
endq = yaw_plan.end_q(); p1, R1 = planner.fk('right', endq['right'])
hyp = {key: np.array(actual[key], copy=True) for key in ['bag', 'bear', 'q_left', 'q_right', 'tcp_left', 'tcp_right', 'time']}
hyp['q_left'][:7] = endq['left']; hyp['q_right'][:7] = endq['right']
hyp['tcp_left'] = planner.fk('left', endq['left'])[0]; hyp['tcp_right'] = p1
hyp['bear'] = (R1*R0.inv()).apply(actual['bear']-actual['tcp_right'])+p1
steps = int(np.ceil(yaw_plan.duration/cfg['simulation']['dt']))
hyp['time'] = np.array(float(actual['time'])+steps*cfg['simulation']['dt'])
np.savez_compressed(OUT/'hypothetical_after_yaw.npz', **hyp)
report['hypothetical_construction'] = {'formula': '(endpoint_FK_R * inverse(actual_source_FK_R)) * (actual_bear - actual_source_TCP) + endpoint_FK_TCP',
    'clock_increment_steps': steps, 'clock_increment_s': steps*cfg['simulation']['dt'], 'hypothetical_time_s': float(hyp['time']),
    'max_point_difference_from_exact_ideal_plus25_m': float(np.linalg.norm(hyp['bear']-ideal_bear, axis=1).max()),
    'endpoint_TCP_error_m': float(np.linalg.norm(p1-actual['tcp_right'])),
    'endpoint_R_error_rad': float((R1*target_R.inv()).magnitude()),
    'bag_positions_unchanged': np.array_equal(hyp['bag'], actual['bag']),
    'finger_positions_unchanged': all(np.array_equal(hyp['q_'+s][7:], actual['q_'+s][7:]) for s in command),
    'left_arm_pose_scope': 'Saved commanded endpoint; original measured left fingers retained. Not an actual native endpoint.',
    'no_inherited_force_arrays': set(hyp) == {'bag', 'bear', 'q_left', 'q_right', 'tcp_left', 'tcp_right', 'time'}}
report['status'] = 'yaw_preflight_passed_evaluating_two_offsets'; save()
print('YAW_PASSED', yaw_plan.duration, 'rigid', yaw['robot_preflight']['minimum_sampled_rigid_clearance_m'], flush=True)
for name, offset in [('zero', [0., 0.]), ('positive_y30p75', [0., .03075])]:
    candidate_config = copy.deepcopy(config)
    candidate_config['task']['contact_trial']['center_offsets_xy_m'] = [offset]
    (OUT/(name+'_diagnostic_config.json')).write_text(json.dumps(candidate_config, indent=2)+'\n')
    started = time.monotonic()
    choice = plan_guarded_contact_trial(rest, hyp, candidate_config, output=OUT/(name+'_preparation.json'))
    summary = {'name': name, 'offset_xy_m': offset, 'preparation_status': choice['status'],
        'preparation_rejections': choice['report'].get('rejections', []), 'cpu_wall_seconds': time.monotonic()-started,
        'section_centers': [{key: center[key] for key in ['center_of_mass_xy_target_m', 'section_guard_rejections',
             'minimum_section_slab_overlap_fraction', 'maximum_outside_excursion_bound_m', 'active_perimeter_margin_m', 'perimeter_margin_m']}
              for center in choice['report'].get('centering_candidates', [])],
        'robot_attempts': [{key: row.get(key) for key in ['center_of_mass_xy_target_m', 'rejection', 'stages_completed']}
              for row in choice['report'].get('robot_centering_attempts', [])], 'physical_success': False}
    report['candidates'].append(summary); save()
    print('PREP', name, choice['status'], choice['report'].get('rejections', []), summary['robot_attempts'], flush=True)
    if choice['status'] != 'candidate': continue
    # Only this selected preparation route is replayed geometrically, with its
    # already reported endpoint q values; no alternative path is optimized.
    prepared = {key: value.copy() for key, value in hyp.items()}
    time_delta = 0.; motion_clock = []
    for stage in choice['stages_to_execute']:
        time_steps = int(np.ceil(stage['duration_s']/cfg['simulation']['dt']))
        time_delta += time_steps*cfg['simulation']['dt']
        motion_clock.append({'stage': stage['name'], 'reported_duration_s': stage['duration_s'], 'integer_steps': time_steps})
        prepared['q_right'][:7] = stage['right_q_end']
    pend, Rend = planner.fk('right', prepared['q_right'][:7])
    prepared['bear'] = (Rend*R1.inv()).apply(hyp['bear']-hyp['tcp_right'])+pend
    prepared['tcp_right'] = pend; prepared['time'] = np.array(float(hyp['time'])+time_delta)
    np.savez_compressed(OUT/(name+'_hypothetical_transport_endpoint.npz'), **prepared)
    fresh = reobserve_guarded_contact_trial(rest, prepared, candidate_config, choice, output=OUT/(name+'_hypothetical_fresh.json'))
    summary.update(hypothetical_fresh_status=fresh['status'], hypothetical_fresh_rejections=fresh['report'].get('rejections', []),
        hypothetical_transport_time_s=float(prepared['time']), preparation_clock=motion_clock,
        preparation_clock_scope='Integer clock based on planner-reported provisional durations, not native runtime-replanned command timing.',
        transport_hypothetical_FK_rotation_error_rad=float((Rend*R1.inv()).magnitude()),
        hypothetical_fresh_robot_attempts=[{key: row.get(key) for key in ['center_of_mass_xy_target_m', 'rejection', 'stages_completed']}
              for row in fresh['report'].get('robot_centering_attempts', [])])
    save(); print('FRESH', name, fresh['status'], fresh['report'].get('rejections', []), flush=True)
report['status'] = 'bounded_two_offset_review_complete'
report['feasible_original_gate_candidates'] = [row['name'] for row in report['candidates'] if row['preparation_status'] == 'candidate' and row.get('hypothetical_fresh_status') == 'candidate']
save(); print(json.dumps({'path': str(OUT/'review.json'), 'sha256': sha(OUT/'review.json'),
    'status': report['status'], 'feasible_original_gate_candidates': report['feasible_original_gate_candidates'],
    'inputs_unchanged': report['original_inputs_unchanged'], 'production_unchanged': report['production_unchanged']}, indent=2), flush=True)
