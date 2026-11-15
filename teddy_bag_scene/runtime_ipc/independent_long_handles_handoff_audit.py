"""Independent CPU-only completed handoff audit; no simulation or source mutation."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/two_panel_long_handles_handoff_01'
# Only execute on a completed native evidence set. Missing files must not create
# an incomplete-looking final verdict or overwrite any historical run report.
_REQUIRED = ('report.json','validation.json','events.json','rest.npz','states.npz',
             'joint_trace.npy','soft_metrics.npy','handle_grip_trace.json',
             'coupling_trace.json','config.json','input_sha256.json',
             'source/run.py','source/validate.py','settle_handoff.npz')
_missing = [name for name in _REQUIRED if not (RUN/name).is_file()]
if _missing:
    raise SystemExit('Native handoff final audit pending; missing: '+', '.join(_missing))
sys.path[:0] = [str(RUN / 'source'), str(ROOT / 'runtime_ipc')]
import validate
from audit_input_manifest import audit as audit_inputs
from audit_richer_contact_traces import audit_handle_trace

def read(name):
    return json.loads((RUN / name).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def scalar(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)

def save(name, value):
    path = RUN / name
    path.write_text(json.dumps(value, indent=2, default=scalar, allow_nan=False) + '\n')
    return dict(path=str(path), sha256=sha(path))

cfg, report, events = read('config.json'), read('report.json'), read('events.json')
rest, frames = dict(np.load(RUN / 'rest.npz')), dict(np.load(RUN / 'states.npz'))
joint, soft = np.load(RUN / 'joint_trace.npy'), np.load(RUN / 'soft_metrics.npy')
dt = float(cfg['simulation']['dt'])
inputs = audit_inputs(RUN)
input_output = save('independent-final-input-manifest-review.json', inputs)
audit = validate.audit_arrays(rest, frames, cfg, task_completed=False,
    joint_trace=joint, events=events, soft_metrics=soft)
audit_output = save('independent-handoff-geometry-soft-audit.json', audit)
handles = audit_handle_trace(read('handle_grip_trace.json'), events, joint, dt)
handle_output = save('independent-handoff-native-reactions.json', handles)

physical_keys = (
    'finite_saved_states', 'per_step_soft_metrics_valid_when_supplied',
    'actual_open_bag_topology', 'two_cloth_panels_with_direct_u_seam_and_no_bottom_or_side_panels',
    'no_tetrahedron_inversion', 'no_tetrahedron_volume_below_5_percent',
    'table_penetration_below_2mm', 'cloth_edge_stretch_below_15_percent',
    'no_final_sampled_bag_bear_penetration_over_1mm', 'joint_motion_within_configured_limits',
    'motion_sample_interval_at_most_20ms', 'left_gripper_holds_pose_within_15mm_after_handoff',
    'named_left_handle_material_within_18mm_after_handoff', 'both_left_fingers_command_clamping_after_handoff')
expected_phases = ['close_both_handles', 'drape_under_gravity', 'spread_bag',
                   'release_right_handle', 'right_clear_handle', 'settle_handoff']
manifest = read('input_sha256.json')
live_mismatches = [k for k, v in manifest.items() if k.startswith('source/') and
                  sha(ROOT / k.removeprefix('source/')) != v]
checks = {k: bool(audit['checks'].get(k, False)) for k in physical_keys}
checks.update(
    frozen_manifest_matches=inputs['all_declared_hashes_match'] and inputs['all_snapshot_files_in_manifest'],
    all_live_source_assets_match_frozen=not live_mismatches,
    exact_reviewed_serialization_only_runner=sha(RUN / 'source/run.py') == '3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9',
    explicitly_requested_handoff_scope=report.get('requested_task') == 'handoff',
    source_config_matches_frozen=json.loads((ROOT / 'config_long_handles_gravity_trial.json').read_text()) == cfg,
    original_config_sha_matches=sha(ROOT / 'config_long_handles_gravity_trial.json') == 'c8a3a0c01b1cc107cbf3f9317b0cd9d5b956dde21c7e4d9fa28cb9c950fc40c4',
    handoff_exact_six_phases_in_order=[e['phase'] for e in events] == expected_phases,
    events_match_report=events == report['events'],
    native_finished_without_exception=report['status'] == 'completed' and not report.get('error') and 'validation_error' not in report,
    every_step_joint_timing=np.allclose(joint[:, 0], np.arange(1, len(joint)+1)*dt, rtol=0, atol=2e-9),
    every_step_soft_timing=np.allclose(soft[:, 0], np.arange(len(soft))*dt, rtol=0, atol=2e-9),
    soft_has_initial_plus_all_joint_rows=len(soft) == len(joint)+1,
    every_joint_and_soft_value_finite=np.isfinite(joint).all() and np.isfinite(soft).all(),
    four_native_finger_reactions_during_both_handle_hold=handles['passed'],
    final_right_handle_fingers_open=np.all(joint[-1, 17:19] >= .038),
    no_teddy_handling_phase_executed=not any(e['phase'] in ('teddy_pregrasp','close_teddy_gripper','lift_teddy','insert_teddy','release_teddy') for e in events),
    full_task_explicitly_false=report['task_passed'] is False and not report.get('sequence_completed', False),
)

source_path = RUN / 'source/assets' / cfg['bag']['npz_file']
source = dict(np.load(source_path))
error, mapping = cKDTree(rest['bag']).query(source['points'] + cfg['bag']['position'])
checks['material_vertex_bijection'] = len(mapping) == len(rest['bag']) == len(np.unique(mapping)) and error.max() <= 1e-7
checks['material_faces_match_native'] = np.array_equal(mapping[source['faces']], rest['bag_faces'])
groups = {s: mapping[source[cfg['task']['handle_grasps'][s]['group']]] for s in ('left','right')}
phase_rows = []
for i, event in enumerate(events):
    state = dict(np.load(RUN / (event['phase'] + '.npz')))
    start = float(event['start']); steps = int(np.ceil(event['duration'] / dt)); end = float(state['time'])
    indices = np.flatnonzero(np.abs(frames['time'] - end) <= 1e-9)
    exact_saved = bool(len(indices) and all(np.array_equal(v, frames[k][indices[-1]]) for k,v in state.items() if k in frames))
    jindices = np.flatnonzero(np.abs(joint[:,0]-end)<=1e-9)
    exact_q = bool(len(jindices) and np.array_equal(state['q_left'],joint[jindices[-1],1:10]) and np.array_equal(state['q_right'],joint[jindices[-1],10:19]))
    seam_a = state['bag'][mapping[source['seam_left'][-1]]]
    seam_b = state['bag'][mapping[source['seam_right'][-1]]]
    rim = state['bag'][mapping[source['rim']]]
    phase_rows.append(dict(phase=event['phase'],start_s=start,end_s=end,planned_duration_s=event['duration'],
        actual_duration_s=end-start,actual_steps=steps,clock_matches_integer_steps=abs(end-start-steps*dt)<2e-9,
        next_phase_contiguous=i==len(events)-1 or abs(float(events[i+1]['start'])-end)<1e-9,
        full_frame_exact=exact_saved,actual_joint_trace_endpoint_exact=exact_q,
        left_named_material_distance_m=float(np.linalg.norm(state['bag'][groups['left']]-state['tcp_left'],axis=1).min()),
        right_named_material_distance_m=float(np.linalg.norm(state['bag'][groups['right']]-state['tcp_right'],axis=1).min()),
        finger_pd_n={s:state['force_'+s][7:] for s in ('left','right')},
        finger_q_m={s:state['q_'+s][7:] for s in ('left','right')},
        top_seam_chord_m=float(np.linalg.norm(seam_b-seam_a)),top_seam_vector_m=seam_b-seam_a,
        actual_rim_span_xyz_m=np.ptp(rim,axis=0),all_state_values_finite=all(np.isfinite(v).all() for v in state.values())))
checks['all_phase_endpoints_and_clocks_reconstruct'] = all(all(row[k] for k in ('clock_matches_integer_steps','next_phase_contiguous','full_frame_exact','actual_joint_trace_endpoint_exact','all_state_values_finite')) for row in phase_rows)
hold = phase_rows[-1]
hold_j = joint[joint[:,0]>hold['start_s']+1e-9]
checks['ten_second_hold_has_exact_2000_native_steps'] = hold['phase']=='settle_handoff' and hold['actual_steps']==2000 and len(hold_j)==2000 and abs(hold['actual_duration_s']-10)<2e-9
checks['reported_final_time_matches_all_streams'] = all(abs(report['simulated_seconds']-x)<1e-9 for x in (joint[-1,0],soft[-1,0],frames['time'][-1],hold['end_s']))

native = report['shell_conversion']['entities'][0]
regions = native['material_region_mapping']['regions']
expected_regions = {'body':np.asarray(source['body_face_indices']),
                    'handle_front':np.flatnonzero(np.all(np.isin(source['faces'],source['handle_front']),axis=1)),
                    'handle_back':np.flatnonzero(np.all(np.isin(source['faces'],source['handle_back']),axis=1))}
for name, ids in expected_regions.items():
    checks['native_'+name+'_material_faces_match'] = np.array_equal(ids,regions[name]['native_face_indices'])
    r = native['region_coefficients'][name]
    checks['native_'+name+'_effective_shear_readback_matches'] = r['all_faces_equal'] and r['face_count']==len(ids) and r['shear_stiffness_n_per_m']==(50. if name=='body' else 250.)
checks['material_source_hash_matches'] = sha(source_path) == native['material_region_mapping']['source_sha256']
checks['native_protected_shell_attributes_unchanged'] = all(native['protected_attributes_unchanged'].values())
checks['native_reference_mass_matches_geometry'] = abs(native['native_rest_mass_kg']-audit['mass']['bag_kg'])<1e-8

coupling = read('coupling_trace.json')
metrics = ('translation_error_m','rotation_error_rad','affine_singular_value_deviation')
coupling_maxima = {key:max(float(link[key]) for row in coupling for link in row['links']) for key in metrics}
checks['coupling_every_saved_frame_timestamp'] = len(coupling)==len(frames['time']) and np.allclose([r['time'] for r in coupling],frames['time'],rtol=0,atol=1e-9)
checks['coupling_raw_link_diagnostics_finite'] = all(np.isfinite(float(link[key])) for row in coupling for link in row['links'] for key in metrics)
checks['runtime_left_held_distance_matches_available_samples'] = report['maximum_left_held_material_distance_m'] >= audit['left_handle_hold_evidence']['maximum_nearest_group_distance_m']-1e-12 and report['maximum_left_held_material_distance_m']<=.018
checks['runtime_right_held_distance_matches_spread_trace'] = abs(report['maximum_right_held_handle_material_distance_m']-handles['maximum_designated_material_distance_m'][1])<1e-12
native_scope = bool(report.get('handoff_probe_completed') and all(checks[k] for k in physical_keys) and checks['final_right_handle_fingers_open'] and checks['native_finished_without_exception'])
checks['native_declared_probe_and_requested_task_flags_agree'] = bool(report['handoff_probe_passed'])==native_scope and bool(report['requested_task_passed'])==native_scope
checks['recomputed_all_validation_checks_equal_native'] = audit['checks'] == read('validation.json')['checks']
passed = bool(all(checks.values()))
summary = dict(kind='independent_completed_handoff_scope_audit',run=str(RUN),
    status='declared_handoff_probe_passed' if passed else 'independent_handoff_audit_failed',
    independent_requested_handoff_probe_passed=passed,full_task_success=False,
    continuous_native_left_finger_reaction_hold_certificate=False,
    native_requested_task=report['requested_task'],native_requested_task_passed=report['requested_task_passed'],
    native_handoff_probe_passed=report['handoff_probe_passed'],native_full_task_passed=report['task_passed'],
    checks=checks,failed_checks=[k for k,v in checks.items() if not v],live_source_mismatches=live_mismatches,
    full_task_incomplete_checks=[k for k,v in audit['checks'].items() if not v],
    simulated_seconds=report['simulated_seconds'],joint_trace_rows=len(joint),soft_trace_rows=len(soft),saved_full_frames=len(frames['time']),
    physical_metrics={k:audit[k] for k in ('minimum_tetrahedron_volume_ratio','maximum_bag_edge_stretch_ratio','maximum_table_penetration_m','motion','motion_thresholds','final_contact','mass')},
    phase_endpoints=phase_rows,native_two_handle_reaction_evidence=handles,
    left_hold_saved_frame_evidence=audit['left_handle_hold_evidence'],
    maximum_left_tcp_displacement_after_handoff_m=audit['sequence_evidence']['maximum_left_tcp_displacement_after_handoff_m'],
    native_per_step_left_material_distance_max_m=report['maximum_left_held_material_distance_m'],
    native_per_step_left_material_distance_scope='Source checks this named group each step from spread onward; only maximum aggregate is retained after right release. Saved full frames independently recomputed.',
    native_material_readback=dict(region_coefficients=native['region_coefficients'],protected_attributes_unchanged=native['protected_attributes_unchanged'],native_rest_mass_kg=native['native_rest_mass_kg'],scope='Initialization native IPC material readback persisted into final report; no separate final-time material-buffer readback.'),
    coupling_diagnostic_maxima=coupling_maxima,
    scope='Both fabric handles held during spread, right handle released and right arm cleared, left holds designated fabric for a 10-second native settle. No teddy grasp/lift/insertion/release attempted.',
    limitations=[
        'Four native finger opening-axis reaction traces cover the actual recorded two-handle hold interval through spread, not the later ten-second one-hand hold. After right release, saved PD forces are motor control evidence, not measured contact forces; no continuous native left-finger reaction certificate is asserted.',
        'No full-task contact guard is initialized in handoff mode. Native force/torque limits of the later teddy contact trial are therefore not certified or applicable to this scoped probe.',
        'Named material proximity and opening-axis reactions do not identify individual cloth-contact pairs or certify force closure.',
        'Every-step state summaries and saved-frame geometry do not certify continuous-time nonpenetration between samples; final bag-bear contact is a sampled final geometry check.',
        'TCP/IPC link coupling diagnostic maxima are observations, not a newly introduced pass threshold.',
        'Actual joint rates are derivatives of actual per-step q under validation limits 1.5 rad/s and 10 rad/s^2; they are not guarantees of exact continuous servo rates.',
        'Top seam width is a chord between material seam corners, not intrinsic cloth width or guaranteed free opening.',
        'Uncalibrated effective 50/250 N/m cloth shear and homogeneous teddy FEM remain model assumptions. This probe does not establish full packing success.'],
    outputs=dict(geometry_soft=audit_output,native_reactions=handle_output,input_manifest=input_output),
    provenance_sha256={str(RUN/name):sha(RUN/name) for name in ['report.json','validation.json','events.json','states.npz','rest.npz','joint_trace.npy','soft_metrics.npy','handle_grip_trace.json','coupling_trace.json','config.json','input_sha256.json','source/run.py','source/motion.py','source/validate.py']},
    audit_script_sha256=sha(Path(__file__)))
output=save('independent-handoff-final-review.json',summary)
print(json.dumps(dict(output=output,status=summary['status'],failed_checks=summary['failed_checks'],physical_metrics=summary['physical_metrics'],left_tcp_drift_m=summary['maximum_left_tcp_displacement_after_handoff_m'],final_phase=phase_rows[-1]),indent=2,default=scalar,allow_nan=False))
