"""Read-only full23 stopped-insertion audit supplement. No native simulation."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/two_panel_full_23'
sys.path.insert(0, str(ROOT / 'runtime_ipc'))
import audit_guarded_run as auditor
sys.path.insert(0, str(RUN / 'source'))
from validate import signed_volumes, mesh_edges
from physics_config import _map_shell_material_regions

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda n: json.loads((RUN / n).read_text())
report, cfg, events = read('report.json'), read('config.json'), read('events.json')
final, audit = read('independent-final-review.json'), read('independent-audit.json')
guard, rows = read('independent-guard-reactions.json'), read('contact_guard_trace.json')
rich = read('independent-richer-contact-review.json')
yaw = read('independent-postlift-yaw-sequence.json')
rim = read('independent-rim-perimeter-policy-review.json')
manifest = read('input_sha256.json')
launch = json.loads((ROOT / 'runtime_ipc/full23-launch.json').read_text())
rest = dict(np.load(RUN / 'rest.npz'))
states = dict(np.load(RUN / 'states.npz'))
failed = dict(np.load(RUN / 'failed_soft_state.npz'))
joint = np.load(RUN / 'joint_trace.npy')
soft = np.load(RUN / 'soft_metrics.npy')
dt = float(cfg['simulation']['dt'])
protected = [RUN / n for n in manifest]
protected += [RUN / n for n in ('report.json', 'events.json', 'states.npz', 'rest.npz',
    'joint_trace.npy', 'soft_metrics.npy', 'contact_guard_trace.json', 'coupling_trace.json',
    'handle_grip_trace.json', 'failed_soft_state.npz', 'postlift_yaw_plan.npz',
    'postlift_yaw_execution.json', 'independent-final-review.json',
    'independent-postlift-yaw-sequence.json', 'independent-guard-reactions.json')]
protected += [ROOT / 'runtime_ipc/audit_guarded_run.py', Path(__file__)]
before = {str(p): sha(p) for p in protected}
checks = {}
checks['all_frozen_input_manifest_hashes_match'] = all(sha(RUN / n) == h for n,h in manifest.items())
checks['frozen_production_sources_assets_equal_live'] = all(sha(ROOT / n.removeprefix('source/')) == h for n,h in manifest.items() if n.startswith('source/'))
config_path = ROOT / 'config_guarded_deformable_rim_trial.json'
checks['live_trial_config_matches_launch_and_frozen_semantics'] = sha(config_path) == launch['config_sha256'] and json.loads(config_path.read_text()) == cfg
checks['frozen_helper_matches_launch'] = sha(RUN / 'source/contact_insertion.py') == launch['contact_helper_sha256']
checks['per_step_actual_arrays_finite'] = np.isfinite(joint).all() and np.isfinite(soft).all()
checks['every_step_joint_trace_coverage'] = len(joint) == round(report['simulated_seconds']/dt) and abs(joint[0,0]-dt)<1e-8 and abs(joint[-1,0]-report['simulated_seconds'])<1e-8 and np.all(np.abs(np.diff(joint[:,0])-dt)<1e-8)
checks['every_step_soft_trace_coverage'] = len(soft)==len(joint)+1 and abs(soft[0,0])<1e-10 and np.allclose(soft[1:,0],joint[:,0],rtol=0,atol=1e-8)
checks['soft_and_actual_rate_gates_pass'] = soft[:,1].min()>=.05 and soft[:,4].max()<=1.15 and soft[:,2:4].max()<=.002 and all(x['within_configured_limits'] for x in audit['motion'].values())
checks['only_final_tcp_torque_guard_trips'] = [k for k,v in guard['checks'].items() if not v] == ['raw_tcp_torque_below_limit','no_recorded_stop_reason']
checks['material_and_per_link_wrench_audits_pass'] = rich['held_handle_monitor']['passed'] and rich['per_link_wrench_reconstruction']['passed']
checks['recenter_disabled_and_no_corrections'] = cfg['task']['contact_trial']['recenter_enabled'] is False and not report['guarded_contact_corrections'] and not any(e['phase'].startswith('recenter_') for e in events)
checks['active_rim_policy_matches_frozen_config_and_keeps_union_diagnostic'] = rim['passed'] and rim['mode']=='deformable_rim_contact'

# The main yaw audit deliberately depends on the COMPLETE RUN reaction audit.
# It returned early after the later insertion torque trip. Complete its phase-only
# evidence with a separately derived yaw-interval condition, without rewriting
# the main audit or claiming that the whole-run force condition passed.
exe = read('postlift_yaw_execution.json')
yr = [r for r in rows if exe['start_time_s']-1e-8 <= r['time_s'] <= exe['end_time_s']+1e-8]
limits = guard['raw_limits']
yaw_interval_checks = dict(
    global_raw_trace_reconstruction_coverage_and_hold_checks_pass=all(v for k,v in guard['checks'].items() if k not in ('raw_tcp_torque_below_limit','no_recorded_stop_reason')),
    exactly_1205_steps_and_source_row=len(yr)==exe['actual_step_count']+1,
    yaw_interval_tcp_torque_below_original_limit=all(r['tcp_torque_norm_nm']<=limits['max_tcp_torque_nm'] for r in yr),
    yaw_interval_has_no_stop_reason=not any(r['stop_reason'] for r in yr))
phase_yaw = auditor.audit_postlift_yaw(RUN, report, states, joint, cfg, {'passed': all(yaw_interval_checks.values())})
phase_yaw['checks']['yaw_interval_native_reaction_audit_passed'] = phase_yaw['checks'].pop('complete_run_native_reaction_audit_passed')
phase_yaw['kind'] = 'independent_full23_yaw_interval_evidence_supplement'
phase_yaw['scope_override'] = 'Only the supplied reaction dependency is scoped to the actual yaw interval. All original saved-source/archive/endpoint/preparation checks are unchanged. The complete-run reaction audit remains FAILED and its original report is untouched.'
phase_yaw['reaction_interval_checks'] = yaw_interval_checks
phase_yaw['complete_run_native_reaction_audit_passed'] = False
phase_yaw['full_task_success'] = False
phase_yaw['main_yaw_aggregate_status'] = yaw['status']
phase_yaw['main_yaw_aggregate_failed_checks'] = [k for k,v in yaw['checks'].items() if not v]
phase_yaw['native_interval'] = dict(start_s=yr[0]['time_s'],end_s=yr[-1]['time_s'],rows=len(yr),max_torque_nm=max(r['tcp_torque_norm_nm'] for r in yr),max_force_n=max(r['net_force_norm_n'] for r in yr))
yaw_path = RUN / 'independent-postlift-yaw-interval-review.json'
yaw_path.write_text(json.dumps(phase_yaw,indent=2,default=auditor._json_scalar,allow_nan=False)+'\n')
checks['supplemental_yaw_interval_and_actual_endpoint_evidence_pass'] = phase_yaw['passed'] and all(yaw_interval_checks.values())

tets = rest['bear_tets']; v0 = signed_volumes(rest['bear'], tets)
edges = mesh_edges(rest['bag_faces'])[0]
l0 = np.linalg.norm(rest['bag'][edges[:,1]]-rest['bag'][edges[:,0]],axis=1)
failed_j = float((signed_volumes(failed['bear'],tets)/v0).min())
failed_stretch = float((np.linalg.norm(failed['bag'][edges[:,1]]-failed['bag'][edges[:,0]],axis=1)/l0).max())
checks['failed_soft_snapshot_is_actual_last_step_and_recomputes'] = abs(float(failed['time'])-joint[-1,0])<1e-9 and np.array_equal(failed['metrics'],soft[-1]) and abs(failed_j-soft[-1,1])<1e-12 and abs(failed_stretch-soft[-1,4])<1e-12
rates = {}
for side,col in [('left',1),('right',10)]:
    vel = np.gradient(joint[:,col:col+7],joint[:,0],axis=0)
    rates[side] = (vel,np.gradient(vel,joint[:,0],axis=0))
phases = []
for i,e in enumerate(events):
    complete = (RUN/(e['phase']+'.npz')).exists()
    end = dict(np.load(RUN/(e['phase']+'.npz'))) if complete else None
    t = float(end['time']) if complete else float(failed['time'])
    mask = (joint[:,0]>=e['start']-1e-9)&(joint[:,0]<=t+1e-9)
    sm = (soft[:,0]>=e['start']-1e-9)&(soft[:,0]<=t+1e-9)
    row = dict(phase=e['phase'],completed=complete,start_s=e['start'],actual_end_or_stop_s=t,planned_duration_s=e['duration'],actual_elapsed_s=t-e['start'],actual_step_count=round((t-e['start'])/dt),per_step_soft=dict(rows=int(sm.sum()),min_J=float(soft[sm,1].min()),max_cloth_stretch=float(soft[sm,4].max()),max_table_penetration_m=float(soft[sm,2:4].max())),actual_sampled_rates={s:dict(velocity_rad_s=float(np.abs(v[mask]).max()),acceleration_rad_s2=float(np.abs(a[mask]).max())) for s,(v,a) in rates.items()})
    row['next_start_or_final_matches'] = abs(t-(events[i+1]['start'] if i+1<len(events) else report['simulated_seconds']))<1e-8
    if complete:
        si = np.flatnonzero(np.abs(states['time']-t)<1e-9); ji=np.flatnonzero(np.abs(joint[:,0]-t)<1e-9)
        row['saved_full_frame_and_actual_joint_trace_match'] = bool(len(si) and len(ji) and all(np.array_equal(x,states[k][si[-1]]) for k,x in end.items()) and np.array_equal(end['q_left'],joint[ji[-1],1:10]) and np.array_equal(end['q_right'],joint[ji[-1],10:19]))
        row['integer_step_end_matches_plan'] = abs(t-e['start']-int(np.ceil(e['duration']/dt))*dt)<1e-8
    phases.append(row)
checks['thirteen_complete_phases_then_partial_insertion_consistent'] = len(phases)==14 and all(p['completed'] and p['integer_step_end_matches_plan'] and p['saved_full_frame_and_actual_joint_trace_match'] for p in phases[:-1]) and not phases[-1]['completed'] and phases[-1]['phase']=='insert_teddy' and all(p['next_start_or_final_matches'] for p in phases)
checks['no_release_withdraw_or_final_settle'] = not any(e['phase'] in ('release_teddy','withdraw_right_arm','final_settle') for e in events)

entity = report['shell_conversion']['entities'][0]
asset = RUN/'source/assets'/Path(cfg['bag']['npz_file']).name
regions,_ = _map_shell_material_regions(asset,cfg['bag']['position'],rest['bag'],rest['bag_faces'])
checks['native_50_250_material_readback_and_source_mapping'] = all(entity['region_coefficients'][n]['shear_stiffness_n_per_m']==mu and entity['region_coefficients'][n]['all_faces_equal'] and np.array_equal(regions[n],entity['material_region_mapping']['regions'][n]['native_face_indices']) for n,mu in [('body',50.),('handle_front',250.),('handle_back',250.)]) and sha(asset)==entity['material_region_mapping']['source_sha256'] and all(entity['protected_attributes_unchanged'].values())

times = np.array([r['time_s'] for r in rows]); torque = np.array([r['tcp_torque_norm_nm'] for r in rows])
last = rows[-1]
checks['unique_torque_exceedance_is_last_recorded_step'] = np.flatnonzero(torque>limits['max_tcp_torque_nm']).tolist()==[len(rows)-1] and abs(times[-1]-report['simulated_seconds'])<1e-9
direc = np.array(last['torque_at_aim_tcp_world_nm']) / last['tcp_torque_norm_nm']
perlink=[]
for i,name in enumerate(last['right_link_names']):
    f=np.array(last['right_link_force_world_n'][i]);tau=np.array(last['right_link_torque_at_aim_tcp_world_nm'][i])
    perlink.append(dict(link=name,force_world_n=f,force_norm_n=np.linalg.norm(f),torque_at_aim_tcp_world_nm=tau,torque_norm_nm=np.linalg.norm(tau),projection_onto_total_torque_direction_nm=float(tau@direc),scope='Projection is additive; individual vector norms are not additive.'))
sample_times = [40.28,41,42,43,43.5,44,44.07,44.2,44.3,44.4,44.45,44.5,44.505,44.51]
chronology=[]
for t in sample_times:
    k=int(np.argmin(abs(times-t))); r=rows[k]; si=int(np.argmin(abs(soft[:,0]-times[k])))
    chronology.append(dict(time_s=times[k],tcp_torque_norm_nm=r['tcp_torque_norm_nm'],net_force_norm_n=r['net_force_norm_n'],right_finger_force_norm_n=r['right_finger_force_norm_n'],min_J=soft[si,1],max_cloth_stretch=soft[si,4],stop_reason=r['stop_reason']))
first_crossings=[]
for limit in [.025,.05,.075,.1,.125,.14,.15]:
    ix=np.flatnonzero(torque>limit)
    first_crossings.append(dict(diagnostic_level_nm=limit,first_time_s=float(times[ix[0]]) if len(ix) else None))
joint_limits=np.array([87,87,87,87,12,12,12.])
pd = {side:dict(per_joint_peak_abs_control_torque_nm=np.max(abs(states['force_'+side][:,:7]),axis=0),per_finger_peak_abs_control_force_n=np.max(abs(states['force_'+side][:,7:]),axis=0),recorded_control_torques_within_native_clamp=bool(np.all(abs(states['force_'+side][:,:7])<=joint_limits+1e-9))) for side in ('left','right')}
checks['saved_pd_joint_controls_respect_native_joint_clamp'] = all(v['recorded_control_torques_within_native_clamp'] for v in pd.values())
fresh = read('guarded_contact_reobserved.json')
section = fresh['report']['selected_sections']
checks['native_stop_and_independent_task_failure_agree'] = report['status']=='failed' and report['error']=='GateFailure: IPC distal-hand torque at TCP exceeded engineering trial limit' and not final['independent_task_success']
checks['protected_inputs_unchanged_during_summary'] = all(sha(p)==h for p,h in before.items())
out = dict(kind='independent_full23_stopped_insertion_evidence_review',status='stop_evidence_consistent_task_incomplete' if all(checks.values()) else 'evidence_review_failed',physical_task_success=False,all_consistency_checks_passed=all(checks.values()),checks=checks,failed_checks=[k for k,v in checks.items() if not v],native_status=report['status'],native_error=report['error'],final_time_s=report['simulated_seconds'],actual_insertion_executed=True,insertion_completed=False,release_executed=False,actual_phases=phases,
    stop=dict(reason='Uncalibrated engineering TCP-moment norm threshold',last_raw_row=last,maximum_and_last_moment_nm=last['tcp_torque_norm_nm'],limit_nm=limits['max_tcp_torque_nm'],excess_nm=last['tcp_torque_norm_nm']-limits['max_tcp_torque_nm'],excess_percent=100*(last['tcp_torque_norm_nm']/limits['max_tcp_torque_nm']-1),exceeding_rows=int(np.count_nonzero(torque>limits['max_tcp_torque_nm'])),right_per_link_at_stop=perlink,chronology=chronology,first_diagnostic_level_crossings=first_crossings),
    force_and_hold=guard,native_initial_handle_grip=rich['held_handle_monitor'],per_link_wrench_reconstruction=rich['per_link_wrench_reconstruction'],every_step_soft=audit['soft_metric_sampling'],last_soft_metrics=dict(time_s=float(failed['time']),min_J=failed_j,max_cloth_stretch=failed_stretch,bag_table_penetration_m=float(soft[-1,2]),bear_table_penetration_m=float(soft[-1,3])),actual_joint_rates=audit['motion'],actual_joint_rate_thresholds=audit['motion_thresholds'],saved_pd_control_torques=pd,
    yaw=dict(main_aggregate_status=yaw['status'],main_aggregate_failed_checks=[k for k,v in yaw['checks'].items() if not v],supplemental_interval_status=phase_yaw['status'],supplemental_report=str(yaw_path),supplemental_sha256=sha(yaw_path),actual_endpoint_position_error_m=phase_yaw.get('actual_endpoint_position_error_m'),actual_endpoint_rotation_error_rad=phase_yaw.get('actual_endpoint_rotation_error_rad'),scope='The original full-run dependency failure remains. Supplement completes actual yaw endpoint/fresh-preparation/401 sample evidence under a verified yaw-interval reaction condition, never a complete-run pass.'),
    active_rim_policy=dict(mode=rim['mode'],criterion=rim['criterion'],audit_passed=rim['passed'],fresh_time_s=fresh['observed_time_s'],fresh_status=fresh['status'],available_rim_m=section['available_rim_perimeter_m'],legacy_union_required_m=section['maximum_required_enclosing_rim_perimeter_m'],legacy_union_margin_m=section['perimeter_margin_m'],active_required_m=section['active_required_enclosing_rim_perimeter_m'],active_margin_m=section['active_perimeter_margin_m'],scope=rim['scope']),
    native_material_readback=dict(regions=entity['region_coefficients'],all_protected_attributes_unchanged=entity['protected_attributes_unchanged'],scope='Native initialization readback persisted in final report, not a new end-time buffer sample.'),
    franka_comparison=dict(native_model_and_official_panda_joint_torque_limits_nm=joint_limits,engineering_tcp_moment_norm_limit_nm=.15,quantities_not_interchangeable=True,conclusion='The TCP moment exceeded this experiment engineering threshold. It does not establish a manufacturer joint-torque-limit violation, structural failure, or violation of physical laws. Joint torque requires a configuration-dependent wrench mapping plus other contacts/dynamic contributions. Saved control forces are PD commands, not all actual load torques.',sources=[dict(url='https://frankarobotics.github.io/docs/robot_specifications.html',claim='FER/Panda maximum joint torques: 87,87,87,87,12,12,12 Nm.'),dict(url='https://frankarobotics.github.io/libfranka/latest/classfranka_1_1Robot.html',claim='setCollisionBehavior configures separate joint-torque and Cartesian force/moment per-axis thresholds.'),dict(path=str(RUN/'source/scene.py'),line=114,claim='Native joint actuator force clamp uses the same seven torque limits.')]),
    unmet_full_task_checks=final['failed_geometry_or_soft_checks'],last_complete_geometry_and_robot_frame_s=float(states['time'][-1]),failure_snapshot_scope='failed_soft_state has actual bag and bear geometry at 44.510s but no robot q/TCP; actual joint trace reaches 44.510s. Full geometry/robot saved frame ends at 44.480s.',
    limitations=['Force feedback is IPC soft-coupling STC and excludes native Genesis rigid contacts. Per-link reactions do not identify the contacting material or object.','Positive opening reactions and material proximity support holding, not force closure.','Actual rate estimates use finite differences at integration steps; guarded continuous reference .35/.5 differs from original actual validation 1.5/10.','Sampled control torques do not certify continuous actuator saturation or physical hardware safety.','No release, withdrawal or final retention occurred. Final sampled containment diagnostics are not a completed insertion success test.','Native soft state summaries cover every integration step, not a continuous collision or material calibration proof.','Geometry/contact root cause requires the separate actual-state review; torque chronology alone does not prove rear-handle causation.'],input_sha256=before)
path=RUN/'independent-executed-prefix-review.json'
path.write_text(json.dumps(out,indent=2,default=auditor._json_scalar,allow_nan=False)+'\n')
print(json.dumps(dict(path=str(path),sha256=sha(path),consistency_checks=len(checks),all_passed=all(checks.values()),failed=out['failed_checks'],yaw_interval=phase_yaw['status'],yaw_failed=[k for k,v in phase_yaw['checks'].items() if not v]),indent=2))
