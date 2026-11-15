"""Bounded independent mutations of marked synthetic optional-yaw audit evidence.
No simulation, helper planning, production writes, or physical-success claim.
"""
from pathlib import Path
import copy,hashlib,json,shutil,sys,tempfile,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'runtime_ipc')]
import audit_guarded_run as audit
FIX=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'runtime_ipc/postlift-yaw-audit-synthetic-fixture'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,default=audit._json_scalar,allow_nan=True)+'\n')
base={k:json.loads((FIX/f).read_text()) for k,f in [('config','config.json'),('report','report.json'),('guards','contact_guard_trace.json')]}
base['states']=dict(np.load(FIX/'states.npz',allow_pickle=False));base['trace']=np.load(FIX/'joint_trace.npy',allow_pickle=False)
base['archive']=dict(np.load(FIX/'postlift_yaw_plan.npz',allow_pickle=False))
tracked=[ROOT/'run.py',ROOT/'postlift_yaw.py',ROOT/'config_guarded_postlift_yaw_trial.json',ROOT/'runtime_ipc/audit_guarded_run.py']
source_before={str(p):sha(p) for p in tracked}
fixture_before={str(p.relative_to(FIX)):sha(p) for p in FIX.rglob('*') if p.is_file()}
checks=[]

def small_curve_jump(b): b['archive']['right_c'][3,10,0]+=1e-7

def past_run_end(b):
    b['report']['simulated_seconds']=b['report']['postlift_yaw']['observed_time_s'];b['guards']=b['guards'][:1]

def unentered(b):
    b['report'].pop('postlift_yaw');b['report'].pop('postlift_yaw_execution');b['report']['events']=b['report']['events'][:1]

def noexecution(b):
    b['report'].pop('postlift_yaw_execution');b['report']['events']=b['report']['events'][:1]

def failedexec(b):
    b['report']['postlift_yaw_execution']['status']='execution_rejected';b['report'].pop('guarded_contact_candidate')

def fresh_rejected_preparation(b):
    b['report']['guarded_contact_candidate']['status']='rejected'
    b['report']['guarded_contact_candidate']['route']=None
    b['report']['guarded_contact_candidate']['report']['rejections']=['Synthetic route rejection after fresh actual observation']

mutations=[('baseline_success',True,None),('interior_curve_position_jump',False,small_curve_jump),('successful_endpoint_past_reported_run_end_with_truncated_guard',False,past_run_end),('trace_wrong_column_count',False,lambda b:b.update(trace=b['trace'][:,:1])),('trace_endpoint_disagrees_saved_state',False,lambda b:b['trace'].__setitem__((-1,10),b['trace'][-1,10]+.01)),('archived_finger_target_changes_grasp',False,lambda b:b['archive'].update(finger_right=np.array(.04))),('configured_not_entered',False,unentered),('candidate_not_executed',False,noexecution),('execution_rejected',False,failedexec)]
if 'report' in base['report'].get('guarded_contact_candidate',{}):
    mutations += [
        ('fresh_preparation_geometrically_rejected_does_not_erase_yaw_evidence',True,fresh_rejected_preparation),
        ('stale_preparation_inner_clock',False,lambda b:b['report']['guarded_contact_candidate']['report'].update(observed_time_s=b['report']['postlift_yaw']['observed_time_s'])),
        ('stale_preparation_route_clock',False,lambda b:b['report']['guarded_contact_candidate']['route'].update(observed_time_s=b['report']['postlift_yaw']['observed_time_s'])),
        ('stale_preparation_physical_clock',False,lambda b:b['report']['guarded_contact_candidate']['report']['physical_state'].update(time_s=b['report']['postlift_yaw']['observed_time_s'])),
        ('wrong_preparation_tcp',False,lambda b:b['report']['guarded_contact_candidate']['report']['observed_right_tcp_world_m'].__setitem__(0,0.)),
        ('wrong_preparation_rest_mass_com',False,lambda b:b['report']['guarded_contact_candidate']['report']['observed_center_of_mass_world_m'].__setitem__(0,0.)),
        ('stale_preparation_complete_rotation',False,lambda b:b['report']['guarded_contact_candidate']['route'].update(rotation_xyzw=b['report']['postlift_yaw']['actual_source_poses']['right']['rotation_xyzw'])),
        ('guard_interior_step_deleted',False,lambda b:b['guards'].pop(100)),
        ('guard_baseline_reset',False,lambda b:b['report']['contact_guard']['baseline'].update(time_s=b['report']['postlift_yaw_execution']['end_time_s'])),
        ('archived_left_motor_target_changed',False,lambda b:b['archive'].update(finger_left=np.array(.001))),
        ('curve_first_derivative_jump',False,lambda b:b['archive']['right_c'].__setitem__((2,10,0),b['archive']['right_c'][2,10,0]+1e-7)),
        ('curve_second_derivative_jump',False,lambda b:b['archive']['right_c'].__setitem__((1,10,0),b['archive']['right_c'][1,10,0]+1e-7)),
    ]
with tempfile.TemporaryDirectory(prefix='independent-optional-yaw-review-') as tmp:
    run=Path(tmp)/'fixture';shutil.copytree(FIX,run)
    for name,expected,mutate in mutations:
        b=copy.deepcopy(base)
        if mutate:mutate(b)
        write(run/'config.json',b['config']);write(run/'events.json',b['report']['events']);write(run/'contact_guard_trace.json',b['guards'])
        for key,f in [('postlift_yaw','postlift_yaw_candidate.json'),('postlift_yaw_execution','postlift_yaw_execution.json')]:
            if key in b['report']:write(run/f,b['report'][key])
            elif (run/f).exists():(run/f).unlink()
        np.savez_compressed(run/'postlift_yaw_plan.npz',**b['archive'])
        manifest=json.loads((FIX/'input_sha256.json').read_text());manifest['config.json']=sha(run/'config.json');write(run/'input_sha256.json',manifest)
        t=time.monotonic();result=audit.audit_postlift_yaw(run,b['report'],b['states'],b['trace'],b['config'],{'passed':True})
        row=dict(name=name,expected_passed=expected,passed=result['passed']==expected,actual_status=result['status'],actual_passed=result['passed'],failed_checks=[k for k,v in result['checks'].items() if not v],error=result.get('error'),seconds=time.monotonic()-t)
        checks.append(row);print(json.dumps(row),flush=True)
checks.append(dict(name='production_source_config_and_auditor_unchanged_during_review',passed=source_before=={str(p):sha(p) for p in tracked}))
checks.append(dict(name='synthetic_fixture_unchanged',passed=fixture_before=={str(p.relative_to(FIX)):sha(p) for p in FIX.rglob('*') if p.is_file()}))
report=dict(kind='independent_optional_postlift_auditor_mutation_review',physical_success=False,fixture_is_synthetic_not_native=True,auditor_sha256=sha(ROOT/'runtime_ipc/audit_guarded_run.py'),source_sha256=source_before,fixture=str(FIX),check_count=len(checks),all_checks_passed=all(r['passed'] for r in checks),checks=checks)
report['review_scope']=['Only marked synthetic evidence was mutated; no native state, production source, configuration or GPU activity.', 'Archived cubics, source/endpoints, new actual preparation and guard clocks are cross-checked; legitimate subsequent geometric prep rejection does not invalidate completed yaw evidence.', 'Rigid/FK reference checks remain sampled. Frozen ideal-grasp arc certificates do not establish native soft deformation, continuous FK safety or bagging success.', 'The test supplies reactions.passed=true only to isolate optional-yaw contracts; complete native reaction auditing remains mandatory in production main.']
write(ROOT/'runtime_ipc/independent-postlift-auditor-review.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('checks','source_sha256')},indent=2))
