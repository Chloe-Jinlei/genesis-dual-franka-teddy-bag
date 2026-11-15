"""Synthetic saved-evidence contract tests; no physical success is inferred."""
import copy,json,hashlib,tempfile,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent;H=HERE.parent;sys.path.insert(0,str(HERE))
import audit_guarded_run as audit
P=H/'runs/two_panel_full_12';config=json.loads((H/'config_guarded_staged_trial.json').read_text());source=dict(np.load(P/'transport_teddy.npz'));choice=json.loads((P/'guarded_contact_candidate.json').read_text());plans=[json.loads((P/f'staged_{p}_CPU_candidate.json').read_text())for p in ('entry','center','insert')]
start=float(source['time']);dt=config['simulation']['dt'];events=[dict(phase='transport_teddy',start=start-1.,duration=1.)];frames=[dict(time=start,tcp_right=source['tcp_right'].copy(),q_right=source['q_right'].copy())];trace=[np.r_[start,source['q_left'],source['q_right']]];now=start;total=0.
for i,p in enumerate(plans):
 p['observed_time_s']=now;p['checkpoint']['observed_tcp_world_m']=frames[-1]['tcp_right'].tolist();stage=p['stage'];before=frames[-1]['tcp_right'].copy();target=np.array(stage['right_tcp_m']);duration=stage['duration_s'];events.append(dict(phase=stage['name'],start=now,duration=duration));now+=np.ceil(duration/dt)*dt;q=source['q_right'].copy();q[:7]=stage['right_q_end'];distance=float(np.linalg.norm(target[:2]-before[:2]));p.update(cumulative_xy_m_before=total,cumulative_xy_m_after=total+distance,runtime_observed_endpoint_displacement_m=distance,runtime_observed_cumulative_endpoint_displacement_m=total+distance,runtime_endpoint_time_s=now);total+=distance;frames.append(dict(time=now,tcp_right=target,q_right=q));trace.append(np.r_[now,source['q_left'],q])
events.append(dict(phase='release_teddy',start=now,duration=.5));states={k:np.asarray([f[k]for f in frames])for k in frames[0]};trace=np.asarray(trace);baseline={'time_s':choice['observed_time_s']};report=dict(events=events,staged_contact_plans=plans,guarded_contact_candidate=choice,contact_guard_baseline=baseline,contact_guard={'baseline':copy.deepcopy(baseline)},sequence_completed=True,simulated_seconds=now+.5)
checks=[]
def check(name,ok):checks.append(dict(name=name,passed=bool(ok)));print(name,bool(ok),flush=True)
original_hand=audit.audit_staged_hand;audit.audit_staged_hand=lambda *args,**kwargs:dict(passed=True,status='synthetic_separately_tested_hand_audit')
with tempfile.TemporaryDirectory(prefix='staged-sequence-contract-')as path:
 run=Path(path);(run/'source/assets').mkdir(parents=True);(run/'source/assets/panda_tactile.xml').write_bytes((P/'source/assets/panda_tactile.xml').read_bytes());(run/'source/motion.py').write_bytes((P/'source/motion.py').read_bytes());(run/'contact_guard_trace.json').write_text(json.dumps([{'time_s':choice['observed_time_s']}]))
 def evaluate(rep=None,ss=None,tt=None,reactions=None):return audit.audit_staged_sequence(run,report if rep is None else rep,states if ss is None else ss,trace if tt is None else tt,config,{'passed':True}if reactions is None else reactions)
 good=evaluate();check('synthetic_valid_staged_sequence_passes',good['passed'])
 mutations={
 'duplicate_physical_entry':lambda r:r['events'].insert(1,copy.deepcopy(r['events'][1])),
 'wrong_phase_plan_order':lambda r:r['staged_contact_plans'].reverse(),
 'stale_center_observation':lambda r:r['staged_contact_plans'][1].update(observed_time_s=start),
 'wrong_full_rotation_target':lambda r:r['staged_contact_plans'][1]['stage'].update(rotation_xyzw=(Rotation.from_rotvec([0,0,.03])*Rotation.from_quat(r['staged_contact_plans'][1]['stage']['rotation_xyzw'])).as_quat().tolist()),
 'baseline_reset':lambda r:r['contact_guard']['baseline'].update(time_s=start),
 'legacy_high_fresh_fabricated':lambda r:r.update(guarded_contact_reobservations=[{'status':'candidate'}]),
 'cumulative_budget_reset':lambda r:r['staged_contact_plans'][2].update(cumulative_xy_m_before=0.),
 'planned_xy_over_30mm':lambda r:r['staged_contact_plans'][1]['stage']['right_tcp_m'].__setitem__(0,frames[1]['tcp_right'][0]-.031),
 'release_before_insert_endpoint':lambda r:r['events'][-1].update(start=r['events'][-1]['start']-.01),
 'NaN_runtime_displacement':lambda r:r['staged_contact_plans'][1].update(runtime_observed_endpoint_displacement_m=float('nan'))}
 for name,mutate in mutations.items():
  bad=copy.deepcopy(report);mutate(bad);check(name+'_rejected',not evaluate(bad)['passed'])
 bad={k:v.copy()for k,v in states.items()};bad['tcp_right'][-1,0]+=.004;check('actual_endpoint_tracking_error_rejected',not evaluate(ss=bad)['passed'])
 bad={k:v[:-1].copy()for k,v in states.items()};check('missing_actual_endpoint_rejected',not evaluate(ss=bad)['passed'])
 bad=trace.copy();bad[2,16]+=.1;check('actual_full_rotation_deviation_rejected',not evaluate(tt=bad)['passed'])
 check('continuous_guard_failure_rejected',not evaluate(reactions={'passed':False})['passed'])
 audit.audit_staged_hand=lambda *args,**kwargs:dict(passed=False,status='synthetic_height_failure');check('native_hand_failure_rejected',not evaluate()['passed']);audit.audit_staged_hand=original_hand
 # Old mode must retain the exact original correction-loop result.
 legacy=json.loads((P/'report.json').read_text());legacy_states=dict(np.load(P/'states.npz'));old=json.loads((P/'independent-correction-sequence.json').read_text());check('legacy_correction_audit_unchanged',audit.audit_corrections(legacy,legacy_states,dt)==old)
result=dict(kind='synthetic_staged_sequence_auditor_regression',all_checks_passed=all(c['passed']for c in checks),checks=checks,physical_success=False,scope='Sequence evaluator only; native-height and every-step raw-reaction auditors are separately exercised, not mocked into a physical success claim.',auditor_sha256=hashlib.sha256((HERE/'audit_guarded_run.py').read_bytes()).hexdigest());(HERE/'staged-sequence-audit-regression.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
