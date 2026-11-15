"""Read only the stopped full28 release sequence; no planning or native stepping."""
from pathlib import Path
import hashlib, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_28'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((RUN/n).read_text())
names=['report.json','events.json','config.json','input_sha256.json','gravity_release_actual_review.json','independent-gravity-execution-review.json','joint_trace.npy','soft_metrics.npy','states.npz','insert_teddy.npz','failed_soft_state.npz','contact_guard_trace.json','source/run.py','source/motion.py']
hashes={n:sha(RUN/n) for n in names}
report=read('report.json');events=read('events.json');cfg=read('config.json');audit=read('independent-gravity-execution-review.json');decision=read('gravity_release_actual_review.json');guards=read('contact_guard_trace.json')
j=np.load(RUN/'joint_trace.npy');soft=np.load(RUN/'soft_metrics.npy');states=dict(np.load(RUN/'states.npz'));endpoint=dict(np.load(RUN/'insert_teddy.npz'))
event=next(e for e in events if e['phase']=='release_teddy');start=float(event['start']);duration=float(event['duration']);end=float(report['simulated_seconds']);dt=cfg['simulation']['dt'];force_limit=cfg['task']['bear_finger_force_limit']
sel=(j[:,0]>=start-1e-8)&(j[:,0]<=end+1e-8);jr=j[sel];gr=[g for g in guards if start-1e-8<=g['time_s']<=end+1e-8];sm=soft[(soft[:,0]>=start-1e-8)&(soft[:,0]<=end+1e-8)];ss=(states['time']>=start-1e-8)&(states['time']<=end+1e-8)
def target(t):
    u=float(np.clip((t-start)/duration,0,1));return .04*(10*u**3-15*u**4+6*u**5)
def closest_row(t):
    q=j[np.argmin(abs(j[:,0]-t))];g=min(guards,key=lambda x:abs(x['time_s']-q[0]));m=soft[np.argmin(abs(soft[:,0]-q[0]))]
    return dict(time_s=float(q[0]),elapsed_release_s=float(q[0]-start),source_reconstructed_motor_target_per_finger_m=target(q[0]),actual_right_finger_positions_m=q[17:19].tolist(),nominal_joint_separation_m=float(q[17:19].sum()),right_opening_reactions_n=g['right_finger_opening_projection_n'],right_finger_contact_force_norms_n=g['right_finger_force_norm_n'],net_force_n=g['net_force_norm_n'],tcp_torque_nm=g['tcp_torque_norm_nm'],minimum_tetra_volume_ratio=float(m[1]),right_two_sided_contact_required=g['right_two_sided_contact_required'])
def first_and_persistent(mask):
    ix=np.flatnonzero(mask)
    return None if not len(ix) else {'time_s':gr[int(ix[0])]['time_s'],'elapsed_s':gr[int(ix[0])]['time_s']-start,'remains_true_through_last_row':bool(np.all(mask[int(ix[0]):]))}
openings=np.asarray([g['right_finger_opening_projection_n'] for g in gr]);finger_norms=np.asarray([g['right_finger_force_norm_n'] for g in gr])
first_unload=first_and_persistent(np.max(openings,axis=1)<=.1)
first_no_contact=first_and_persistent(np.max(finger_norms,axis=1)<=1e-6)
minidx=np.argmin(jr[:,17:19],axis=0)
sampled_controls=[{'time_s':float(t),'actual_motor_control_force_n':f[7:].tolist(),'actual_finger_positions_m':q[7:].tolist()} for t,f,q in zip(states['time'][ss],states['force_right'][ss],states['q_right'][ss])]
badJ=np.flatnonzero(soft[:,1]<.05)
checks={
 'reviewed_runner_3454_and_manifest':hashes['source/run.py']=='3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9'==read('input_sha256.json')['source/run.py'],
 'actual_original_finite_force_18252N_zero_close_target':report['teddy_grip_limits']=={'force_per_finger_n':1.8252,'target_per_finger_m':0.0} and force_limit==1.8252,
 'admissible_actual_capture_before_intent':decision['status']=='admissible_release' and all(v is True for v in decision['checks'].values()) and abs(decision['actual_time_s']-start)<1e-8,
 'complete_fixed_step_release_joint_and_guard_rows':len(jr)==101 and len(gr)==101 and np.max(abs(np.diff(jr[:,0])-dt))<1e-9 and np.max(abs(np.asarray([g['time_s'] for g in gr])-jr[:,0]))<1e-9,
 'right_hold_required_at_endpoint_then_intentionally_disabled':gr[0]['right_two_sided_contact_required'] is True and all(g['right_two_sided_contact_required'] is False for g in gr[1:]),
 'finite_native_joint_force_and_soft_records':np.isfinite(jr).all() and np.isfinite(sm).all() and np.isfinite(states['force_right'][ss]).all(),
 'saved_motor_forces_within_unchanged_finite_limit':np.max(abs(states['force_right'][ss,7:]))<=force_limit+1e-12,
 'actual_fingers_reached_open_endpoint':np.max(abs(jr[-1,17:19]-.04))<.00002,
 'right_finger_contact_vanished_before_failure':first_no_contact is not None and first_no_contact['remains_true_through_last_row'] and first_no_contact['time_s']<end,
 'native_failed_soft_gate_not_grip_or_wrench_gate':report['status']=='failed' and 'Teddy element excessively compressed' in report['error'] and len(badJ)>0 and abs(soft[badJ[0],0]-end)<1e-8 and audit['every_step_reaction_audit']['passed'],
 'withdrawal_and_settle_not_executed':all(e['phase'] not in ['withdraw_right_arm','final_settle'] for e in events) and not (RUN/'gravity_withdraw_plan.npz').exists(),
 'existing_auditor_did_not_pass_incomplete_task':audit['passed'] is False and audit['status']=='incomplete' and audit['incomplete_evidence']==['No withdrawal curve archived'],
 'all_reviewed_evidence_unchanged':all(sha(RUN/n)==h for n,h in hashes.items()),
}
checks={k:bool(v) for k,v in checks.items()}
out=dict(kind='independent_full28_release_sequence_review',diagnostic_checks_passed=all(checks.values()),checks=checks,physical_task_success=False,
 actual_run={'status':report['status'],'error':report['error'],'failure_time_s':end,'release_start_s':start,'release_duration_s':duration,'integrated_release_steps':int(round((end-start)/dt)),'release_phase_completed':False,'failure_scope':'Final release integration step was recorded, then original J guard raised before the normal release endpoint capture/return; withdrawal and settle were not executed.'},
 existing_gravity_audit={'path':'independent-gravity-execution-review.json','sha256':hashes['independent-gravity-execution-review.json'],'status':audit['status'],'passed':audit['passed'],'passed_prefix_checks':sum(audit['checks'].values()),'incomplete_evidence':audit['incomplete_evidence'],'scope':'The existing auditor returns at missing withdrawal archive. It certifies listed executed-prefix checks, not final physics, final containment, or all later auditor sections.'},
 admissible_release_containment=decision['actual_containment'],
 source_sequence={'runner_sha256':hashes['source/run.py'],'motor_force_cap_per_finger_n':force_limit,'command_start_per_finger_m':0.,'command_end_per_finger_m':.04,'interpolation':'0.5 s minimum-jerk position and analytic velocity feed-forward; both arm command positions held constant','minimum_jerk_target_at_02s_m':target(start+.2),'actual_start_per_finger_m':jr[0,17:19].tolist(),'minimum_actual_positions_m':[float(jr[minidx[k],17+k]) for k in range(2)],'minimum_actual_position_times_s':[float(jr[minidx[k],0]) for k in range(2)],'actual_final_per_finger_m':jr[-1,17:19].tolist(),'interpretation':'The historical closed motor target is zero even though actual fingers surround the bear at about18.86mm. The release ramp therefore initially continues closing and only later opens. This is source-consistent existing behavior, not proof of a cause for the later local J failure.'},
 first_both_opening_reactions_at_most_01N=first_unload,first_both_contact_force_norms_at_most_1uN=first_no_contact,
 selected_release_rows=[closest_row(start+x) for x in [0,.05,.10,.15,.20,.25,.28,.285,.30,.35,.40,.45,.495,.50]],
 sampled_motor_control_forces=sampled_controls,
 release_metrics={'minimum_J':float(sm[:,1].min()),'maximum_cloth_edge_ratio':float(sm[:,4].max()),'maximum_table_penetration_m':float(sm[:,2:4].max()),'maximum_right_net_force_n':max(g['net_force_norm_n'] for g in gr),'maximum_right_TCP_torque_nm':max(g['tcp_torque_norm_nm'] for g in gr),'first_J_failure_time_s':float(soft[badJ[0],0]),'last_complete_geometry_time_s':float(states['time'][-1]),'final_guard_finger_force_norms_n':gr[-1]['right_finger_force_norm_n']},
 input_sha256=hashes,limitations=['Finger displacement sum is nominal joint separation, not an exact deformed pad-to-pad geometric distance.','Command sequence is reconstructed from pinned source and recorded event; there is no separate per-step target log. Actual positions and coupling reactions are observed at5ms; motor control force is saved with complete geometry approximately every50ms.','A negative PD force while fingers move outward can be braking; after48.080s the measured finger coupling force remains below1microN.','The terminal local compression occurs after fingers have unloaded. Its contact-object/tetra geometry cause is outside this bounded sequencing review.','No native stepping, planning, source/config/asset/default edit, or final-success assertion.'])
dest=RUN/'independent-release-sequence-review.json'
if dest.exists():raise FileExistsError(dest)
dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(dest),'sha256':sha(dest),'checks':sum(checks.values()),'total':len(checks),'failed':[k for k,v in checks.items() if not v],'first_unload':first_unload,'first_no_contact':first_no_contact,'metrics':out['release_metrics']},indent=2))
