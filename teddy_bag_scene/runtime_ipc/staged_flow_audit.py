"""Isolated control-flow tests; no simulator or state restoration."""
import ast, copy, hashlib, json, sys, types
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parents[1]
src=(HERE/'run.py').read_text()
cls=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='Experiment')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('run_staged_contact_insertion','check_staged_hand_clearance')]
class GateFailure(Exception): pass
ns=dict(np=np,Rotation=Rotation,GateFailure=GateFailure,positions=lambda x: np.zeros((4,3)))
exec(compile(ast.Module(body=methods,type_ignores=[]),'isolated_staged_methods','exec'),ns)

def scenario(fault=None,settings=None):
    log=[]
    def plan(rest,state,config,*,phase,previous,output):
        log.append(('observe',phase,float(state['time'])))
        target=state['tcp_right'].copy()
        options=config['task']['contact_trial']['staged']
        adaptive='center_offsets_xy_m' in options
        selected_index=min(1,len(options.get('center_offsets_xy_m',[]))-1) if adaptive else None
        selected=options['center_offsets_xy_m'][selected_index] if adaptive else options['center_offset_xy_m']
        if phase=='center': target[:2]+=selected
        else: target[2]-=.17 if phase=='entry' else .24
        name=dict(entry='lower_teddy_to_rim',center='center_teddy_at_rim',insert='insert_teddy')[phase]
        r=dict(status='candidate',phase=phase,observed_time_s=float(state['time']),original_rotation_xyzw=[0,0,0,1],
               stage=dict(name=name,right_tcp_m=target.tolist(),rotation_xyzw=[0,0,0,1],duration_s=6.),outside_tcp_m=[target[0],target[1],1.2])
        before=0. if phase=='entry' else previous['runtime_observed_cumulative_endpoint_displacement_m']
        r['cumulative_xy_m_before']=before
        r['cumulative_xy_m_after']=before+float(np.linalg.norm(target[:2]-state['tcp_right'][:2]))
        if fault==phase+'_reject': r['status']='rejected'
        if phase=='center':
            if adaptive:
                r['selected_center_offset_xy_m']=copy.deepcopy(selected)
                r['center_selection']=dict(selected_index=selected_index,
                    selected_offset_xy_m=copy.deepcopy(selected),actual_observed_time_s=float(state['time']),
                    later_insert_requires_fresh_actual_observation=True)
                if fault=='missing_selection':r.pop('center_selection')
                if fault=='noninteger_index':r['center_selection']['selected_index']=True
                if fault=='wrong_index':r['center_selection']['selected_index']=0
                if fault=='out_of_range_index':r['center_selection']['selected_index']=len(options['center_offsets_xy_m'])
                if fault=='wrong_selected_delta':r['selected_center_offset_xy_m'][0]+=.001
                if fault=='wrong_selection_delta':r['center_selection']['selected_offset_xy_m'][0]+=.001
                if fault=='stale_selection_time':r['center_selection']['actual_observed_time_s']-=.005
                if fault=='future_insert_authorized':r['center_selection']['later_insert_requires_fresh_actual_observation']=False
            if fault=='wrong_phase': r['phase']='insert'
            if fault=='wrong_name': r['stage']['name']='insert_teddy'
            if fault=='stale_plan': r['observed_time_s']-=.005
            if fault=='nan_plan_time': r['observed_time_s']=float('nan')
            if fault=='rotated': r['stage']['rotation_xyzw']=Rotation.from_euler('z',.01).as_quat().tolist()
            if fault=='changed_reference': r['original_rotation_xyzw']=Rotation.from_euler('z',.01).as_quat().tolist()
            if fault=='nan_target': r['stage']['right_tcp_m'][0]=float('nan')
            if fault=='short_duration': r['stage']['duration_s']=.1
            if fault=='z_centering': r['stage']['right_tcp_m'][2]+=.001
            if fault=='wrong_xy': r['stage']['right_tcp_m'][0]-=.01
        if phase=='insert' and fault=='diagonal_insert': r['stage']['right_tcp_m'][0]+=.001
        if phase=='entry' and fault=='ascending_entry': r['stage']['right_tcp_m'][2]+=1.
        return r
    mod=types.ModuleType('staged_contact_insertion');mod.plan_staged_contact=plan;sys.modules[mod.__name__]=mod
    mon=types.ModuleType('staged_hand_monitor')
    class Monitor:
        def __init__(self,scene,minimum_gap_m): assert minimum_gap_m==.005
        def reference_geometry(self): return {'mock':np.zeros((4,3))}
        def report(self): return dict(source='mock native all vertices')
        def measure(self,t,bear):
            return dict(time_s=t,clearance_m=.004 if fault=='native_gap' else .01), 'low native gap' if fault=='native_gap' else None
    mon.NativeHandHeightMonitor=Monitor;sys.modules[mon.__name__]=mon
    class Sim:
        check_staged_hand_clearance=ns['check_staged_hand_clearance']
        def __init__(self):
            self.scene=None;self.rest={};self.time=42.;self.frames=[dict(time=42.,tcp_right=np.array([-.1,.057,1.37]))]
            self.out=Path('/tmp');self.report={};self.config={'task':{'contact_trial':{'staged':{'center_offset_xy_m':[-.027,-.010]}}}}
            if settings: self.config['task']['contact_trial']['staged'].update(settings)
            self.require_bear_below_left_hand=False;self.must_hold_teddy=True;self.staged_hand_trace=[];self.bear=None
            self.guard_token=object()
            if fault=='stale_actual': self.frames[0]['time']=41.9
        def move_guarded(self,label,stage):
            log.append(('move',label,self.must_hold_teddy,self.require_bear_below_left_hand,id(self.guard_token)))
            self.time+=stage['duration_s']
            target=np.array(stage['right_tcp_m'])
            if label=='center_teddy_at_rim' and fault=='actual_xy_overshoot': target[0]-=.004
            if label=='center_teddy_at_rim' and fault=='expanded_actual_xy_overshoot': target[1]-=.03
            self.frames.append(dict(time=self.time,tcp_right=target))
        def hold(self,label,duration,**fingers):
            log.append(('hold',label,self.must_hold_teddy,self.require_bear_below_left_hand,id(self.guard_token)))
            self.time+=duration
    sim=Sim();error=None
    choice=dict(stages_to_execute=[dict(rotation_xyzw=[0,0,0,1])])
    try: ns['run_staged_contact_insertion'](sim,choice)
    except GateFailure as exc: error=str(exc)
    return dict(error=error,complete=sim.report.get('sequence_completed',False),log=log)

results={}
for fault in [None,'entry_reject','center_reject','insert_reject','wrong_phase','wrong_name','stale_plan','nan_plan_time','rotated','changed_reference','nan_target','short_duration','z_centering','wrong_xy','diagonal_insert','ascending_entry','native_gap','stale_actual','actual_xy_overshoot']:
    result=scenario(fault);results[fault or 'successful_three_fresh_stages']=result
    if fault is None:
        assert result['complete'] and result['error'] is None
        observations=[x for x in result['log'] if x[0]=='observe']
        assert [x[1] for x in observations]==['entry','center','insert']
        assert all(b[2]>a[2] for a,b in zip(observations,observations[1:]))
        operations=[x for x in result['log'] if x[0] in ('move','hold')]
        assert [x[1] for x in operations]==['lower_teddy_to_rim','center_teddy_at_rim','insert_teddy','release_teddy','withdraw_right_arm','final_settle']
        assert all(x[2] for x in operations[:3]) and all(not x[2] for x in operations[3:])
        assert [x[3] for x in operations[:3]]==[False,True,True]
        assert len({x[4] for x in operations})==1
    else:
        assert not result['complete'] and result['error'], (fault,result)
        assert not any(x[1]=='release_teddy' for x in result['log']), (fault,result)
expanded={'center_offset_xy_m':[-.02,-.05],'maximum_individual_xy_m':.07,'maximum_cumulative_xy_m':.09}
for label,fault,settings,passed in [
    ('explicit_expanded_budget',None,expanded,True),
    ('expanded_actual_xy_overshoot','expanded_actual_xy_overshoot',expanded,False),
    ('expanded_delta_without_explicit_budget',None,{'center_offset_xy_m':[-.02,-.05]},False),
    ('nonfinite_budget',None,dict(expanded,maximum_individual_xy_m=float('nan')),False),
    ('excessive_budget',None,dict(expanded,maximum_individual_xy_m=.101),False),
    ('inconsistent_budget',None,dict(expanded,maximum_individual_xy_m=.08,maximum_cumulative_xy_m=.07),False)]:
    result=scenario(fault,settings);results[label]=result
    assert result['complete']==passed and (result['error'] is None)==passed,(label,result)
    if not passed: assert not any(x[1]=='release_teddy' for x in result['log']),(label,result)
adaptive=dict(expanded,center_offsets_xy_m=[[-.015,-.055],[-.02,-.05]])
for fault in [None,'missing_selection','noninteger_index','wrong_index','out_of_range_index',
              'wrong_selected_delta','wrong_selection_delta','stale_selection_time','future_insert_authorized']:
    result=scenario(fault,adaptive);results['adaptive_'+str(fault or 'select_second_and_reobserve')]=result
    assert result['complete']==(fault is None) and (result['error'] is None)==(fault is None),(fault,result)
    if fault is not None: assert not any(x[1]=='release_teddy' for x in result['log']),(fault,result)
for label,offsets in [('duplicate',[[-.02,-.05],[-.02,-.05]]),('zero',[[0.,0.]]),
                      ('overbudget',[[-.07,-.01]]),('too_many',[[-.01,-.03-i*.001] for i in range(9)])]:
    result=scenario(None,dict(adaptive,center_offsets_xy_m=offsets));results['adaptive_invalid_'+label]=result
    assert result['error'] and not result['complete'] and not any(x[1]=='release_teddy' for x in result['log']),(label,result)
report=dict(passed=True,case_count=len(results),source_sha256=hashlib.sha256(src.encode()).hexdigest(),results=results,
            scope='Runner sequencing, fresh observations, complete orientation, bounded low XY shift and failure before release. Mock physics only.')
(HERE/'runtime_ipc/staged-flow-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='results'}))
