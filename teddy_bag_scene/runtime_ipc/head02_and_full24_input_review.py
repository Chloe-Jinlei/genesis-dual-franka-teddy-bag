"""Bounded saved-result/head02 and startup/full24 checks. No native calls."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'runs/two_panel_head_center_probe_02'
OLD=ROOT/'runs/two_panel_head_center_probe_01'
sys.path.insert(0,str(R/'source'))
from motion import Planner
from validate import signed_volumes
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return sum((differences(a.get(k),b.get(k),path+'.'+k) for k in sorted(a.keys()|b.keys())),[])
    return [] if a==b else [dict(path=path,old=a,new=b)]
def write(p,x):
    p.write_text(json.dumps(x,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(p),sha256=sha(p),checks=len(x['checks']),all_passed=all(x['checks'].values()),failed=[k for k,v in x['checks'].items() if not v]),indent=2))
c=read(R/'config.json');old=read(OLD/'config.json');report=read(R/'report.json');manifest=read(R/'input_sha256.json')
rest=dict(np.load(R/'rest.npz'));states=dict(np.load(R/'states.npz'));failed=dict(np.load(R/'failed_soft_state.npz'));joint=np.load(R/'joint_trace.npy');soft=np.load(R/'soft_metrics.npy')
coupling=read(R/'coupling_trace.json');protected=[R/n for n in manifest]+[R/n for n in ('report.json','rest.npz','initial.npz','states.npz','failed_soft_state.npz','joint_trace.npy','soft_metrics.npy','coupling_trace.json')];before={str(p):sha(p) for p in protected}
pc=json.loads(json.dumps(c));pc['robots']['mjcf_path']=str(R/'source/assets/panda_tactile.xml');planner=Planner(pc)
v0=signed_volumes(rest['bear'],rest['bear_tets']);weights=abs(v0);ref=np.array(c['bear']['position'])+c['task']['bear_grasp_local'];roi=np.flatnonzero(np.linalg.norm(rest['bear']-ref,axis=1)<c['task']['bear_grasp_tracking_radius_m'])
trajectory=[]
for i,t in enumerate(states['time']):
    p,rot=planner.fk('right',states['q_right'][i,:7]);b=states['bear'][i]
    row=min(coupling,key=lambda x:abs(x['time']-t));fingers=[x for x in row['links'] if x['entity']=='right_franka' and x['link'].endswith('_finger')]
    raw=None
    if all(x.get('last_feedback_force_world_n') is not None for x in fingers):
        forces=np.array([x['last_feedback_force_world_n'] for x in fingers]);axes=np.array([x['aim_opening_axis_world'] for x in fingers]);raw=dict(time_s=row['time'],forces_world_n=forces,opening_reactions_n=np.einsum('ij,ij->i',forces,axes))
    trajectory.append(dict(time_s=float(t),actual_TCP_world_m=p,finger_half_openings_m=states['q_right'][i,7:],nearest_teddy_vertex_to_TCP_m=float(np.linalg.norm(b-p,axis=1).min()),ROI_hand_local_m=(b[roi].mean(0)-p)@rot.as_matrix(),mass_COM_world_m=np.average(b[rest['bear_tets']].mean(1),weights=weights,axis=0),bear_minimum_z_m=float(b[:,2].min()),sampled_reaction=raw))
fp,fr=planner.fk('right',joint[-1,10:17]);distance=float(np.linalg.norm(failed['bear']-fp,axis=1).min());lastJ=float((signed_volumes(failed['bear'],rest['bear_tets'])/v0).min())
after=[x for x in trajectory if x['time_s']>2.66 and x['sampled_reaction'] is not None]
loss=[]
for i,x in enumerate(after):
    if all(np.max(np.abs(y['sampled_reaction']['opening_reactions_n']))<.1 and max(abs(y['finger_half_openings_m']))<.001 for y in after[i:]):loss=[x['time_s'],after[-1]['time_s'],len(after)-i];break
changes=differences(old,c)
checks=dict(only_config_change_three_to_two_N=changes==[dict(path='.task.bear_finger_force_limit',old=3.,new=2.)],frozen_manifest_all_matches=all(sha(R/n)==h for n,h in manifest.items()),sources_and_assets_same_as_3N=all(sha(OLD/n)==h for n,h in manifest.items() if n.startswith('source/')),rest_and_initial_exactly_same_as_3N=all(all(np.array_equal(z[k],np.load(OLD/n)[k]) for k in z.files) for n in ['rest.npz','initial.npz'] for z in [np.load(R/n)]),failure_is_native_slip_guard=report['error']=='GateFailure: Teddy slipped out of the right finger pads',failure_nearest_vertex_distance_exceeds_45mm=distance>.045,failed_soft_J_recomputes_and_did_not_trigger=abs(lastJ-soft[-1,1])<1e-12 and soft[:,1].min()>=.05,cloth_table_and_finite_gates_remain_valid=soft[:,4].max()<=1.15 and soft[:,2:4].max()<=.002 and np.isfinite(joint).all() and np.isfinite(soft).all() and soft[:,5].min()==1,per_step_trace_reaches_failure=abs(joint[-1,0]-float(failed['time']))<1e-9 and np.array_equal(soft[-1],failed['metrics']) and len(soft)==len(joint)+1 and np.allclose(soft[1:,0],joint[:,0],atol=1e-9,rtol=0),fingers_closed_and_contact_lost_before_distance_gate=bool(loss and loss[0]<float(failed['time'])-.1),lift_not_completed_or_claimed_success=not (R/'lift_teddy.npz').exists() and not report['requested_task_passed'],original_2N_PD_clamp_preserved=bool(np.max(abs(states['force_right'][states['time']>=1.66-1e-9,7:]))<=2+1e-9))
checks['protected_inputs_unchanged']=all(sha(p)==h for p,h in before.items())
out=dict(kind='independent_head_center_2N_slip_review',physical_grasp_success=False,full_task_success=False,checks=checks,config_diff=changes,native_error=report['error'],failure_time_s=float(failed['time']),lift_start_s=report['events'][-1]['start'],planned_lift_duration_s=report['events'][-1]['duration'],actual_lift_elapsed_s=float(failed['time'])-report['events'][-1]['start'],global_min_J=float(soft[:,1].min()),global_min_J_time_s=float(soft[soft[:,1].argmin(),0]),failure_J=lastJ,failure_nearest_teddy_vertex_to_reconstructed_TCP_m=distance,failure_joint_finger_half_openings_m=joint[-1,17:19],last_complete_state_s=float(states['time'][-1]),sustained_sampled_closed_empty_gripper_interval=dict(start_s=loss[0],end_s=loss[1],samples=loss[2]) if loss else None,selected_timeline=[min(trajectory,key=lambda x:abs(x['time_s']-t)) for t in [2.66,3.36,3.86,4.16,4.66,5.36]],conclusion='The actual fingers closed to essentially zero, sampled opening reactions vanished, and the teddy remained on the table while the TCP continued upwards. This supports actual loss of the teddy from the grasp before the 45mm proximity gate stopped the run. Reducing 3N to 2N avoided the earlier 5% volume-ratio stop but did not produce a completed lift.',limitations=['The 45mm stop uses nearest teddy vertex to TCP, not exact contact distance. Independent finger positions and sparse reactions substantiate the loss-of-grasp interpretation.','Reactions are only saved at complete frames (last5.36s), not at every integration step or exact5.395s failure.','Failure TCP is reconstructed from actual failure-time joints using the frozen FK; the failed snapshot itself contains soft geometry only.','No more head-grasp candidate exploration is performed.'],input_sha256=before)
write(R/'independent-2N-slip-review.json',out)

F=ROOT/'runs/two_panel_full_24';P=ROOT/'runs/two_panel_full_23';launch=read(ROOT/'runtime_ipc/full24-launch.json');f=read(F/'config.json');p=read(P/'config.json');m=read(F/'input_sha256.json');diff=differences(p,f)
protected=[F/n for n in m]+[F/'rest.npz',F/'initial.npz',ROOT/'config_guarded_inward_15mm_trial.json'];before={str(x):sha(x) for x in protected};initial=dict(np.load(F/'initial.npz'));rst=dict(np.load(F/'rest.npz'));oldinitial=dict(np.load(P/'initial.npz'));oldrest=dict(np.load(P/'rest.npz'))
pid=int(launch['pid']);proc=Path('/proc')/str(pid);cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None
checks=dict(only_config_COM_offset_changed=diff==[dict(path='.task.contact_trial.center_offsets_xy_m',old=[[0.,.0345]],new=[[.015,.03075]])],live_config_hash_matches_launch=sha(ROOT/'config_guarded_inward_15mm_trial.json')==launch['config_sha256'],live_config_semantics_matches_frozen=read(ROOT/'config_guarded_inward_15mm_trial.json')==f,all_manifest_hashes_match=all(sha(F/n)==h for n,h in m.items()),all_frozen_source_assets_equal_full23=all(sha(P/n)==h for n,h in m.items() if n.startswith('source/')),all_frozen_sources_assets_equal_live=all(sha(ROOT/n.removeprefix('source/'))==h for n,h in m.items() if n.startswith('source/')),rest_arrays_exactly_equal_full23=set(rst)==set(oldrest) and all(np.array_equal(v,oldrest[k]) for k,v in rst.items()),initial_arrays_exactly_equal_full23=set(initial)==set(oldinitial) and all(np.array_equal(v,oldinitial[k]) for k,v in initial.items()),initial_native_clock_zero=float(initial['time'])==0,upper_arm_3N_minus30_yaw_no_recenter_preserved=f['task']['bear_finger_force_limit']==3 and f['task']['bear_grasp_local']==p['task']['bear_grasp_local'] and f['task']['contact_trial']['postlift_yaw']=={'world_z_degrees':-30.,'duration_s':6.} and f['task']['contact_trial']['recenter_enabled'] is False,native_process_matches_this_run=cmd is not None and 'two_panel_full_24' in cmd and 'config_guarded_inward_15mm_trial.json' in cmd,decision_hash_matches=sha(ROOT/launch['pretrial_decision'])==launch['pretrial_decision_sha256'])
checks['startup_inputs_unchanged_during_review']=all(sha(x)==h for x,h in before.items())
write(F/'independent-startup-input-review.json',dict(kind='independent_full24_startup_input_review',checks=checks,all_checks_passed=all(checks.values()),physical_task_success=False,native_validation_pending=True,config_diff=diff,launch_pid=pid,observed_process_cmdline=cmd,source_comparison='All frozen source/*.py and asset bytes equal full23 and current live input; only explicitly listed config offset changed.',topology=dict(bear_nodes=len(rst['bear']),bear_tetrahedra=len(rst['bear_tets']),bag_nodes=len(rst['bag']),bag_faces=len(rst['bag_faces'])),material_scope='Material config and material implementation are unchanged. This startup check does not invent an end-time or fresh native material-buffer readback before report.json exists.',input_sha256=before))
