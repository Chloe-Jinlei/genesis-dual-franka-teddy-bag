"""Independent optional-yaw audit contracts using a marked synthetic fixture.

The source lift is real full17. The endpoint, guard rows and execution are
synthetic for auditor testing, never offered as native execution evidence.
"""
from pathlib import Path
import copy,hashlib,json,sys,tempfile
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'runtime_ipc')]
import audit_guarded_run as audit
from postlift_yaw import plan_postlift_yaw
from motion import Planner
from contact_insertion import _state_context,_policy
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,default=audit._json_scalar)+'\n')
def copied(x):return copy.deepcopy(x)
source_run=ROOT/'runs/two_panel_full_17'
cfg=json.loads((ROOT/'config_guarded_postlift_yaw_trial.json').read_text())
state=dict(np.load(source_run/'lift_teddy.npz',allow_pickle=False));rest=dict(np.load(source_run/'rest.npz',allow_pickle=False))
old=json.loads((source_run/'independent-production-postlift-yaw-review.json').read_text())
command={s:np.asarray(q) for s,q in old['command_qs'].items()}
helper_before=sha(ROOT/'postlift_yaw.py');runtime_before=sha(ROOT/'run.py')
candidate,plan=plan_postlift_yaw(rest,state,cfg,command)
assert candidate['status']=='candidate'
planner=Planner(cfg);dt=cfg['simulation']['dt'];start=float(state['time']);steps=int(np.ceil(plan.duration/dt));endtime=start+steps*dt
endpoint={k:v.copy() for k,v in state.items()};endpoint['time']=np.array(endtime)
qend=plan.end_q();initialp,A=planner.fk('right',state['q_right'][:7]);endp,E=planner.fk('right',qend['right'])
endpoint['q_right'][:7]=qend['right'];endpoint['tcp_right']=endp
endpoint['q_left'][:7]=qend['left'];endpoint['tcp_left']=planner.fk('left',qend['left'])[0]
endpoint['bear']=(E*A.inv()).apply(state['bear']-initialp)+endp
physical=_state_context(rest,endpoint,cfg,_policy(cfg))['physical']
execution=dict(status='executed_endpoint_passed',start_time_s=start,end_time_s=endtime,endpoint_time_s=endtime,
    planned_duration_s=plan.duration,actual_step_count=steps,physical_success=False,exact_checked_plan_executed=True,contact_guard_baseline_reset=False,
    endpoint_tcp_m=endp.tolist(),endpoint_rotation_xyzw=E.as_quat().tolist(),endpoint_position_error_m=float(np.linalg.norm(endp-state['tcp_right'])),
    endpoint_rotation_error_rad=float((E*Rotation.from_quat(candidate['target_rotation_xyzw']).inv()).magnitude()),actual_physical_state=physical,endpoint_rejections=[])
tets=rest['bear_tets'];tet=rest['bear'][tets];weights=np.abs(np.linalg.det(np.stack([tet[:,i]-tet[:,0] for i in (1,2,3)],axis=-1)))
com=np.average(endpoint['bear'][tets].mean(axis=1),weights=weights,axis=0)
preparation=dict(status='candidate',physical_success=False,runtime_execution_performed=False,observed_time_s=endtime,
    route={'observed_time_s':endtime,'rotation_xyzw':E.as_quat().tolist()},
    report={'observed_time_s':endtime,'observed_right_tcp_world_m':endp.tolist(),'observed_center_of_mass_world_m':com.tolist(),'physical_state':copied(physical),'predicted_targets':{'rotation_xyzw':E.as_quat().tolist()}})
original=json.loads((source_run/'report.json').read_text());lift=next(e for e in original['events'] if e['phase']=='lift_teddy')
events=[copied(lift),dict(phase='orient_teddy_postlift',start=start,duration=plan.duration)]
baseline={'time_s':start}
report=dict(postlift_yaw=candidate,postlift_yaw_execution=execution,events=events,simulated_seconds=endtime,
    contact_guard_baseline=baseline,contact_guard={'baseline':copied(baseline)},guarded_contact_candidate=preparation)
states={k:np.stack([state[k],endpoint[k]]) for k in state}
trace=[]
for i in range(1,steps+1):
    q=plan.sample(min(i*dt,plan.duration))['q'];trace.append(np.r_[start+i*dt,q['left'],state['q_left'][7:],q['right'],state['q_right'][7:]])
trace=np.asarray(trace)
guards=[dict(time_s=start+i*dt,right_two_sided_contact_required=True) for i in range(steps+1)]
archive=dict(duration=plan.duration,observed_time=start,left_x=plan.curves['left'].x,left_c=plan.curves['left'].c,right_x=plan.curves['right'].x,right_c=plan.curves['right'].c,finger_left=0.,finger_right=cfg['task']['bear_finger_target'])
checks=[]
with tempfile.TemporaryDirectory(prefix='independent-yaw-audit-') as tmp:
    run=Path(tmp);(run/'source/assets').mkdir(parents=True)
    for name in ['postlift_yaw.py','motion.py','run.py']:(run/'source'/name).write_bytes((ROOT/name).read_bytes())
    for asset in ['panda_tactile.xml',Path(cfg['bag']['npz_file']).name]:(run/'source/assets'/asset).write_bytes((ROOT/'assets'/asset).read_bytes())
    np.savez_compressed(run/'rest.npz',**rest)
    def write_inputs(c,r,g,a):
        save(run/'config.json',c);save(run/'events.json',r['events']);save(run/'contact_guard_trace.json',g)
        for key,name in [('postlift_yaw','postlift_yaw_candidate.json'),('postlift_yaw_execution','postlift_yaw_execution.json')]:
            path=run/name
            if key in r:save(path,r[key])
            elif path.exists():path.unlink()
        np.savez_compressed(run/'postlift_yaw_plan.npz',**a)
        save(run/'input_sha256.json',{name:sha(run/name) for name in ['config.json','source/postlift_yaw.py','source/motion.py','source/run.py','source/assets/panda_tactile.xml','source/assets/'+Path(cfg['bag']['npz_file']).name]})
    def case(name,expected=False,mutate=None):
        c,r,s,t,g,a=map(copied,(cfg,report,states,trace,guards,archive))
        if mutate:mutate(c,r,s,t,g,a)
        write_inputs(c,r,g,a)
        out=audit.audit_postlift_yaw(run,r,s,t,c,{'passed':True})
        ok=out['passed']==expected
        checks.append(dict(name=name,passed=ok,expected_audit_passed=expected,actual_status=out['status'],
            failed_checks=[k for k,v in out['checks'].items() if not v],error=out.get('error')))
        print(name,ok,out.get('error'),flush=True)
        if name=='synthetic_complete_yaw_contract':save(ROOT/'runtime_ipc/postlift-yaw-audit-synthetic-example.json',dict(out,synthetic_fixture_is_not_native_execution=True))
    case('synthetic_complete_yaw_contract',True)
    mutations={
        'bad_frozen_angle':lambda c,r,s,t,g,a:c['task']['contact_trial']['postlift_yaw'].update(world_z_degrees=15.1),
        'nan_frozen_angle':lambda c,r,s,t,g,a:c['task']['contact_trial']['postlift_yaw'].update(world_z_degrees=float('nan')),
        'boolean_duration':lambda c,r,s,t,g,a:c['task']['contact_trial']['postlift_yaw'].update(duration_s=True),
        'hidden_nonfinite_candidate':lambda c,r,s,t,g,a:r['postlift_yaw'].update(hidden=float('inf')),
        'missing_required_check':lambda c,r,s,t,g,a:r['postlift_yaw']['checks'].pop('continuous_reference_rate_bounds'),
        'changed_check_value':lambda c,r,s,t,g,a:r['postlift_yaw']['checks'].update(returns_exact_checked_plan=False),
        'forged_planning_success':lambda c,r,s,t,g,a:r['postlift_yaw'].update(physical_success=True),
        'future_route_rotation_substituted':lambda c,r,s,t,g,a:r['postlift_yaw'].update(target_rotation_xyzw=A.as_quat().tolist()),
        'tcp_target_moved':lambda c,r,s,t,g,a:r['postlift_yaw']['target_tcp_m'].__setitem__(0,r['postlift_yaw']['target_tcp_m'][0]+.003),
        'source_actual_q_forged':lambda c,r,s,t,g,a:r['postlift_yaw']['actual_source_poses']['right']['q'].__setitem__(0,0.),
        'source_soft_measure_forged':lambda c,r,s,t,g,a:r['postlift_yaw']['actual_physical_state'].update(minimum_tetra_volume_ratio=1.),
        'source_missing_rigid_class':lambda c,r,s,t,g,a:r['postlift_yaw']['actual_physical_state']['actual_rigid_clearances'].pop('arms'),
        'source_nan_rigid_hidden':lambda c,r,s,t,g,a:r['postlift_yaw']['actual_physical_state']['actual_rigid_clearances']['left_mounts'].update(distance=float('nan')),
        'duration_too_short':lambda c,r,s,t,g,a:r['postlift_yaw'].update(duration_s=5.999),
        'integer_step_count_tampered':lambda c,r,s,t,g,a:r['postlift_yaw_execution'].update(actual_step_count=steps-1),
        'boolean_step_count':lambda c,r,s,t,g,a:r['postlift_yaw_execution'].update(actual_step_count=True),
        'baseline_reset_claim':lambda c,r,s,t,g,a:r['postlift_yaw_execution'].update(contact_guard_baseline_reset=True),
        'final_baseline_reset':lambda c,r,s,t,g,a:r['contact_guard']['baseline'].update(time_s=endtime),
        'baseline_starts_at_prep':lambda c,r,s,t,g,a:r['contact_guard_baseline'].update(time_s=endtime),
        'guard_missing_start':lambda c,r,s,t,g,a:g.pop(0),
        'guard_missing_interior':lambda c,r,s,t,g,a:g.pop(100),
        'guard_teddy_hold_disabled':lambda c,r,s,t,g,a:g[200].update(right_two_sided_contact_required=False),
        'duplicate_yaw_event':lambda c,r,s,t,g,a:r['events'].append(copied(r['events'][-1])),
        'event_before_lift':lambda c,r,s,t,g,a:r['events'].reverse(),
        'changed_event_duration':lambda c,r,s,t,g,a:r['events'][-1].update(duration=6.),
        'successful_endpoint_past_simulated_end':lambda c,r,s,t,g,a:(r.update(simulated_seconds=start),g.__delitem__(slice(1,None))),
        'curve_position_jump':lambda c,r,s,t,g,a:a['right_c'].__setitem__((3,10,0),a['right_c'][3,10,0]+1e-7),
        'curve_derivative_jump':lambda c,r,s,t,g,a:a['right_c'].__setitem__((2,10,0),a['right_c'][2,10,0]+1e-7),
        'curve_acceleration_jump':lambda c,r,s,t,g,a:a['right_c'].__setitem__((1,10,0),a['right_c'][1,10,0]+1e-7),
        'finger_right_opens':lambda c,r,s,t,g,a:a.update(finger_right=np.array(.04)),
        'finger_left_changes_grasp':lambda c,r,s,t,g,a:a.update(finger_left=np.array(.01)),
        'actual_trace_endpoint_q_disagrees':lambda c,r,s,t,g,a:t.__setitem__((-1,10),t[-1,10]+.01),
        'actual_trace_has_one_column':lambda c,r,s,t,g,a:t.resize((len(t),1),refcheck=False),
        'nested_prep_original_lift_rotation':lambda c,r,s,t,g,a:r['guarded_contact_candidate']['report']['predicted_targets'].update(rotation_xyzw=A.as_quat().tolist()),
        'route_prep_original_lift_rotation':lambda c,r,s,t,g,a:r['guarded_contact_candidate']['route'].update(rotation_xyzw=A.as_quat().tolist()),
        'nested_prep_clock_stale':lambda c,r,s,t,g,a:r['guarded_contact_candidate']['report'].update(observed_time_s=start),
        'route_prep_clock_stale':lambda c,r,s,t,g,a:r['guarded_contact_candidate']['route'].update(observed_time_s=start),
        'nested_prep_tcp_stale':lambda c,r,s,t,g,a:r['guarded_contact_candidate']['report'].update(observed_right_tcp_world_m=[0.,0.,0.]),
        'nested_prep_com_stale':lambda c,r,s,t,g,a:r['guarded_contact_candidate']['report'].update(observed_center_of_mass_world_m=[0.,0.,0.]),
        'nested_prep_physical_clock_stale':lambda c,r,s,t,g,a:r['guarded_contact_candidate']['report']['physical_state'].update(time_s=start),
        'nested_prep_physical_rigid_stale':lambda c,r,s,t,g,a:r['guarded_contact_candidate']['report']['physical_state']['actual_rigid_clearances']['arms'].update(distance=.1),
        'nan_curve':lambda c,r,s,t,g,a:a['right_c'].__setitem__((0,0,0),float('nan')),
        'truncated_curve_domain':lambda c,r,s,t,g,a:a['right_x'].__setitem__(0,.001),
        'left_curve_moves_between_endpoints':lambda c,r,s,t,g,a:a['left_c'].__setitem__((1,0,0),.01),
        'curve_endpoint_tamper':lambda c,r,s,t,g,a:a['right_c'].__setitem__((3,-1,0),a['right_c'][3,-1,0]+.001),
        'reported_rate_understates_curve':lambda c,r,s,t,g,a:r['postlift_yaw']['robot_preflight']['continuous_rate_bounds']['right'].update(velocity_upper_bound_rad_s=[0.]*7),
        'frozen_arc_missing_geom':lambda c,r,s,t,g,a:r['postlift_yaw']['frozen_bear_left_rotation_sweep']['per_pose'][0]['segments'][0]['per_geom'].pop(),
        'frozen_arc_wrong_axis':lambda c,r,s,t,g,a:r['postlift_yaw']['frozen_bear_left_rotation_sweep']['per_pose'][0]['segments'][0]['per_geom'][0]['certificate'].update(separating_axis_world=[0.,0.,1.]),
        'frozen_arc_sagitta_omitted':lambda c,r,s,t,g,a:r['postlift_yaw']['frozen_bear_left_rotation_sweep'].update(sagitta_outer_bound_m=0.),
        'frozen_arc_missing_pose':lambda c,r,s,t,g,a:r['postlift_yaw']['frozen_bear_left_rotation_sweep']['per_pose'].pop(),
        'frozen_arc_lower_forged':lambda c,r,s,t,g,a:r['postlift_yaw']['frozen_bear_left_rotation_sweep']['per_pose'][0]['segments'][0]['per_geom'][0].update(continuous_arc_lower_bound_m=2.),
        'endpoint_time_not_integer_step':lambda c,r,s,t,g,a:r['postlift_yaw_execution'].update(endpoint_time_s=endtime-.001),
        'endpoint_record_wrong_position':lambda c,r,s,t,g,a:r['postlift_yaw_execution']['endpoint_tcp_m'].__setitem__(0,0.),
        'endpoint_actual_q_forged':lambda c,r,s,t,g,a:s['q_right'].__setitem__((1,0),s['q_right'][1,0]+.05),
        'endpoint_soft_claim_forged':lambda c,r,s,t,g,a:r['postlift_yaw_execution']['actual_physical_state'].update(minimum_tetra_volume_ratio=1.),
        'endpoint_compressed_tet':lambda c,r,s,t,g,a:s['bear'].__setitem__((1,rest['bear_tets'][0,0]),s['bear'][1,rest['bear_tets'][0,1]]),
        'prep_before_yaw_end':lambda c,r,s,t,g,a:r['guarded_contact_candidate'].update(observed_time_s=start),
        'prep_stale_one_step':lambda c,r,s,t,g,a:r['guarded_contact_candidate'].update(observed_time_s=endtime-dt),
        'missing_actual_endpoint_frame':lambda c,r,s,t,g,a:s['time'].__setitem__(1,endtime-.005),
        'actual_trace_step_missing':lambda c,r,s,t,g,a:t.__setitem__((50,0),t[49,0]),
        'unconfigured_yaw_evidence':lambda c,r,s,t,g,a:c['task']['contact_trial'].pop('postlift_yaw'),
        'native_yaw_execution_rejected':lambda c,r,s,t,g,a:r['postlift_yaw_execution'].update(status='execution_rejected'),
    }
    for name,mutate in mutations.items():case(name,mutate=mutate)
    case('fresh_rejected_preparation_still_proves_yaw_observed',True,lambda c,r,s,t,g,a:r['guarded_contact_candidate'].update(status='rejected',route=None))
    # Explicit missing/tampered serialized records are independent of report fields.
    for name,mutate_file in [
        ('candidate_file_differs_from_report',lambda:save(run/'postlift_yaw_candidate.json',{})),
        ('execution_file_differs_from_report',lambda:save(run/'postlift_yaw_execution.json',{})),
        ('manifest_missing_helper',lambda:save(run/'input_sha256.json',{})),
        ('frozen_helper_hash_mismatch',lambda:(run/'source/postlift_yaw.py').write_text('#tampered\n'))]:
        write_inputs(cfg,report,guards,archive);mutate_file();out=audit.audit_postlift_yaw(run,report,states,trace,cfg,{'passed':True})
        checks.append(dict(name=name,passed=not out['passed'],actual_status=out['status'],failed_checks=[k for k,v in out['checks'].items() if not v],error=out.get('error')))
    # Disabled legacy runs require no new files and are not failed by absence.
    for number in [17,18,19]:
        legacy=ROOT/f'runs/two_panel_full_{number}';lc=json.loads((legacy/'config.json').read_text());lr=json.loads((legacy/'report.json').read_text())
        out=audit.audit_postlift_yaw(legacy,lr,{},None,lc,{})
        checks.append(dict(name=f'legacy_full{number}_exact_disabled_contract',passed=out['passed'] and out['status']=='not_configured' and not out['required']))
checks.append(dict(name='runtime_and_helper_not_changed',passed=helper_before==sha(ROOT/'postlift_yaw.py') and runtime_before==sha(ROOT/'run.py')))
result=dict(kind='optional_postlift_yaw_independent_auditor_regression',physical_success=False,
    synthetic_fixture_is_not_native_execution=True,all_checks_passed=all(v['passed'] for v in checks),check_count=len(checks),
    failed_checks=[v for v in checks if not v['passed']],checks=checks,auditor_sha256=sha(ROOT/'runtime_ipc/audit_guarded_run.py'))
save(ROOT/'runtime_ipc/postlift-yaw-audit-regression.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if not result['all_checks_passed']:raise SystemExit(1)
