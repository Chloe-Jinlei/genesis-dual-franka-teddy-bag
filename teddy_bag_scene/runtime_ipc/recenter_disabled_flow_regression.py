"""Isolated execution of the real runner method; no simulation or live config edits."""
from pathlib import Path
import ast,copy,hashlib,json,sys,tempfile,types
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];source=(ROOT/'run.py').read_text()
tree=ast.parse(source);method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_guarded_contact_trial')
class GateFailure(Exception):pass
namespace={'np':np,'GateFailure':GateFailure}
exec(compile(ast.Module(body=[method],type_ignores=[]),'<isolated real run_guarded_contact_trial>','exec'),namespace)
MISSING=object()

def scenario(enabled=MISSING,observations=('candidate',)):
    calls=[];queue=list(observations);fake=types.ModuleType('contact_insertion')
    def emit(output,result):
        Path(output).write_text(json.dumps(result,indent=2)+'\n');return result
    def stage(name,target):return dict(name=name,right_tcp_m=np.asarray(target).tolist(),rotation_xyzw=[1.,0.,0.,0.],duration_s=1.)
    def prepare(rest,state,cfg,output):
        calls.append('prepare');p=np.asarray(state['tcp_right']);target=p+np.array([.04,.02,.10])
        return emit(output,dict(status='candidate',stages_to_execute=[stage('height_at_current_xy',[p[0],p[1],target[2]]),stage('translate_outside_mouth',target)]))
    def reobserve(rest,state,cfg,selected,output):
        calls.append('fresh');status=queue.pop(0);p=np.asarray(state['tcp_right'])
        return emit(output,dict(status=status,observed_time_s=float(state['time']),report={'rejections':[] if status=='candidate' else ['synthetic actual fresh geometry rejection']},
            insert_stage=stage('insert_candidate',p-np.array([0.,0.,.10])),outside_tcp_m=p.tolist()))
    def recenter(rest,state,cfg,selected,attempt_index,cumulative_xy_m,output):
        calls.append('correction_plan');p=np.asarray(state['tcp_right']);delta=np.array([.01,0.]);target=p+np.r_[delta,0.]
        return emit(output,dict(status='candidate',observed_time_s=float(state['time']),attempt_index=attempt_index,
            correction_distance_m=.01,cumulative_xy_m_before=cumulative_xy_m,cumulative_xy_m_after=cumulative_xy_m+.01,
            correction_xy_m=delta.tolist(),stages_to_execute=[stage('height_at_current_xy',p),stage('translate_outside_mouth',target)]))
    fake.plan_guarded_contact_trial=prepare;fake.reobserve_guarded_contact_trial=reobserve;fake.recenter_guarded_contact_trial=recenter
    class Sim:
        run_guarded_contact_trial=namespace['run_guarded_contact_trial']
        def __init__(self,out):
            self.config={'task':{'contact_trial':{'transport_xy_strategy':'direct','insertion_strategy':'vertical'}}}
            if enabled is not MISSING:self.config['task']['contact_trial']['recenter_enabled']=enabled
            self.out=out;self.rest={};self.time=10.;self.frames=[{'time':self.time,'tcp_right':np.array([.2,0.,1.3])}]
            self.report={};self.must_hold_teddy=True
        def begin_guarded_contact_monitor(self):calls.append('guard')
        def move_guarded(self,label,stage):
            calls.append(label);self.time+=stage['duration_s'];self.frames.append({'time':self.time,'tcp_right':np.array(stage['right_tcp_m'])})
        def hold(self,label,duration,**kwargs):
            calls.append(label);self.time+=duration;self.frames.append({'time':self.time,'tcp_right':self.frames[-1]['tcp_right'].copy()})
    with tempfile.TemporaryDirectory(prefix='recenter-disabled-flow-') as tmp:
        sim=Sim(Path(tmp));error=None
        with patch.dict(sys.modules,{'contact_insertion':fake}):
            try:sim.run_guarded_contact_trial()
            except Exception as exc:error=type(exc).__name__+': '+str(exc)
        saved={p.name:json.loads(p.read_text()) for p in Path(tmp).glob('*.json')}
        return dict(calls=calls,error=error,report=sim.report,saved_json=saved,remaining_fresh=len(queue))

checks=[];cases={}
def check(name,condition):checks.append(dict(name=name,passed=bool(condition)))
for mode,value in [('false',False),('default',MISSING),('true',True)]:
    row=scenario(value,('candidate',));cases[mode+'_fresh_pass']=row
    check(mode+'_fresh_pass_reaches_insert_release_withdraw',row['error'] is None and all(x in row['calls'] for x in ['insert_teddy','release_teddy','withdraw_right_arm','final_settle']) and row['report'].get('sequence_completed') is True)
    check(mode+'_fresh_pass_never_recentered','correction_plan' not in row['calls'] and not row['report']['guarded_contact_corrections'])
row=scenario(False,('rejected',));cases['false_fresh_reject']=row
check('false_reject_raises_explicit_GateFailure',row['error']=='GateFailure: Fresh contact insertion rejected and recentering is disabled; see guarded_contact_reobserved.json')
check('false_reject_no_correction_planner_or_motion','correction_plan' not in row['calls'] and not any(x.startswith('recenter_') for x in row['calls']))
check('false_reject_no_insert_release_withdraw',all(x not in row['calls'] for x in ['insert_teddy','release_teddy','withdraw_right_arm','final_settle']))
check('false_reject_saved_fresh_before_stop',row['saved_json']['guarded_contact_reobserved.json']['status']=='rejected' and len(row['report']['guarded_contact_reobservations'])==1)
check('false_reject_correction_list_empty',row['report']['guarded_contact_corrections']==[] and not any(x.startswith('guarded_contact_correction') for x in row['saved_json']))
for mode,value in [('default',MISSING),('true',True)]:
    for n in (1,2):
        row=scenario(value,('rejected',)*n+('candidate',));cases[f'{mode}_{n}_corrections_then_pass']=row
        check(f'{mode}_{n}_legacy_corrections_execute',row['error'] is None and row['calls'].count('correction_plan')==n and all(f'recenter_teddy_xy_{j}' in row['calls'] for j in range(1,n+1)) and 'insert_teddy' in row['calls'])
    row=scenario(value,('rejected',)*3);cases[mode+'_three_fresh_reject']=row
    check(mode+'_legacy_two_correction_limit',row['calls'].count('correction_plan')==2 and row['calls'].count('fresh')==3 and 'insert_teddy' not in row['calls'] and 'after two' in row['error'])
for bad in [None,0,1,0.,1.,'false','true',[],{},np.bool_(False)]:
    row=scenario(bad);cases['invalid_'+repr(bad)+'_'+type(bad).__name__]=row
    check('invalid_type_'+repr(bad)+'_'+type(bad).__name__,bool(row['error'] and 'must be a boolean' in row['error']) and row['calls']==[] and row['report']=={} and row['saved_json']=={})
check('true_and_default_identical_calls',cases['true_1_corrections_then_pass']['calls']==cases['default_1_corrections_then_pass']['calls'] and cases['true_three_fresh_reject']['calls']==cases['default_three_fresh_reject']['calls'])
check('runner_source_not_changed_during_regression',source==(ROOT/'run.py').read_text())
result=dict(kind='explicit_recenter_disable_runner_flow_regression',physical_success=False,mocked_native_execution=True,
    all_checks_passed=all(c['passed'] for c in checks),check_count=len(checks),failed_checks=[c for c in checks if not c['passed']],checks=checks,cases=cases,
    run_source_sha256=hashlib.sha256(source.encode()).hexdigest(),scope='Real runner method executed with deterministic plan/motion/observation mocks; config parsing and control flow only, not physics evidence.')
out=ROOT/'runtime_ipc/recenter-disabled-flow-regression.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ['all_checks_passed','check_count','failed_checks','run_source_sha256']},indent=2))
if not result['all_checks_passed']:raise SystemExit(1)
