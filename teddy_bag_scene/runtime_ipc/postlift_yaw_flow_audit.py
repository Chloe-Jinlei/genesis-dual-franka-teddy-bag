"""Isolate optional yaw runner sequencing, failure handling and plan identity.

Mock physics only; independent helper and native-run audits cover actual data.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from motion import MotionPlan
from postlift_yaw import _finite_tree, _clearance_rows, _serial, RIGID_CATEGORIES

source=(HERE/'run.py').read_text()
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Experiment')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in
         ('run_postlift_yaw','run_guarded_contact_trial')]
class GateFailure(Exception): pass
namespace=dict(np=np,json=json,Rotation=Rotation,GateFailure=GateFailure)
exec(compile(ast.Module(body=methods,type_ignores=[]),'isolated_postlift_yaw_runner','exec'),namespace)


def scenario(fault=None,legacy=False):
    events=[]
    target=Rotation.from_euler('z',15,degrees=True)
    right0=np.zeros(7); right1=np.r_[np.deg2rad(15),np.zeros(6)]
    plan=MotionPlan({'left':CubicSpline([0.,1.],np.zeros((2,7)),axis=0),
                     'right':CubicSpline([0.,1.],np.array([right0,right1]),axis=0)},
                    6.,{'left':.012,'right':.03},{'left':.012,'right':.03})
    def planner(rest,state,config,qs):
        events.append(('yaw_plan',float(state['time'])))
        report=dict(kind='guarded_postlift_yaw',status='candidate',physical_success=False,
            runtime_execution_performed=False,modifies_soft_points=False,
            requires_fresh_actual_preparation_after_execution=True,
            observed_time_s=float(state['time']),parameters=copy.deepcopy(config['task']['contact_trial']['postlift_yaw']),
            checks=dict.fromkeys(['actual_original_physical_gates','lift_above_table_40mm',
                'command_actual_pose_consistent','analytic_joint_limits','continuous_reference_rate_bounds',
                'sampled_all_raw_rigid_checks_finite_and_above_8mm','sampled_actual_reference_fk_tolerances',
                'actual_and_commanded_left_frozen_rotation_above_5mm','returns_exact_checked_plan'],True),
            target_tcp_m=state['tcp_right'].tolist(),
            target_rotation_xyzw=target.as_quat().tolist(),duration_s=plan.duration,
            left_q_end=plan.end_q()['left'].tolist(),right_q_end=plan.end_q()['right'].tolist())
        if fault=='rejected': report['status']='rejected'
        if fault=='wrong_kind': report['kind']='other'
        if fault=='wrong_parameters': report['parameters']['world_z_degrees']=10.
        if fault=='stale_plan': report['observed_time_s']-=.005
        if fault=='claims_success': report['physical_success']=True
        if fault=='claims_execution': report['runtime_execution_performed']=True
        if fault=='mutates_soft': report['modifies_soft_points']=True
        if fault=='skips_fresh': report['requires_fresh_actual_preparation_after_execution']=False
        if fault=='failed_check': report['checks']['analytic_joint_limits']=False
        if fault=='missing_check': report['checks'].pop('analytic_joint_limits')
        if fault=='empty_checks': report['checks']={}
        if fault=='nan_report': report['duration_s']=float('nan')
        if fault=='wrong_target': report['target_tcp_m'][0]+=.001
        if fault=='wrong_rotation': report['target_rotation_xyzw']=[0.,0.,0.,1.]
        if fault=='wrong_duration': report['duration_s']+=.01
        if fault=='short_duration': plan.duration=5.;report['duration_s']=5.
        if fault=='different_start': plan.curves['right'].c[-1,0,1]+=.001
        if fault=='different_end': report['right_q_end'][0]+=.001
        if fault=='moving_left':
            plan.curves['left']=CubicSpline([0.,1.],np.array([np.zeros(7),np.ones(7)*.01]),axis=0)
            report['left_q_end']=plan.end_q()['left'].tolist()
        return report,None if fault=='missing_plan' else plan
    def context(rest,state,config,policy):
        if fault=='context_exception': raise ValueError('malformed actual endpoint')
        return dict(physical={'actual_rigid_clearances':{key:{'distance':float('nan') if fault=='nan_endpoint_clearance' else .007 if fault=='endpoint_collision' else .02} for key in RIGID_CATEGORIES}},
                    reasons=['physical rejection'] if fault=='endpoint_physics' else [])
    def prepare(rest,state,config,output):
        events.append(('prepare',float(state['time'])))
        names=['height_at_current_xy','transport_y_at_current_x','translate_outside_mouth']
        xyz=[state['tcp_right']+[0,0,.05],state['tcp_right']+[0,.1,.05],state['tcp_right']+[.1,.1,.05]]
        return dict(status='candidate',stages_to_execute=[dict(name=n,right_tcp_m=p.tolist()) for n,p in zip(names,xyz)])
    module=types.ModuleType('postlift_yaw')
    module.plan_postlift_yaw=planner;module._finite_tree=_finite_tree;module._clearance_rows=_clearance_rows;module._serial=_serial
    sys.modules['postlift_yaw']=module
    module=types.ModuleType('contact_insertion')
    module._state_context=context;module._policy=lambda config: {}
    module.plan_guarded_contact_trial=prepare
    module.reobserve_guarded_contact_trial=module.recenter_guarded_contact_trial=lambda *args,**kwargs: None
    sys.modules['contact_insertion']=module
    class Sim:
        run_postlift_yaw=namespace['run_postlift_yaw']
        def __init__(self,out):
            self.time=29.;self.out=out;self.rest={};self.report={};self.dt=.005
            self.qs={'left':np.zeros(7),'right':np.zeros(7)};self.finger={'left':0.,'right':0.}
            self.rotations={};self.contact_guard=None;self.baseline_count=0
            self.must_hold_bag=fault!='no_bag_hold';self.must_hold_teddy=fault!='no_teddy_hold'
            self.frames=[dict(time=28.995 if fault=='stale_state' else 29.,
                tcp_right=np.array([.3,-.25,1.37]),q_right=np.zeros(9),bear=np.ones((4,3))*1.2)]
            self.config=dict(table={'height':.74},task={'contact_trial':{
                'transport_xy_strategy':'y_then_x','insertion_strategy':'lower_then_center'}})
            if not legacy: self.config['task']['contact_trial']['postlift_yaw']={'world_z_degrees':15.,'duration_s':6.}
            self.planner=types.SimpleNamespace(fk=lambda side,q:(np.array([.3,-.25,1.37]),Rotation.from_rotvec([0.,0.,q[0]])))
        def begin_guarded_contact_monitor(self):
            if self.contact_guard is None:
                self.baseline_count+=1;self.contact_guard=object();events.append(('baseline',self.time))
            if fault=='missing_guard': self.contact_guard=None
        def execute(self,label,executed):
            assert executed is plan
            assert executed.finger_start==self.finger==executed.finger_end
            events.append(('execute',label,self.time,id(self.contact_guard)))
            if fault=='native_stop': raise GateFailure('actual native gate stopped')
            self.time+=int(np.ceil(executed.duration/self.dt))*self.dt
            if fault=='wrong_clock': self.time+=.005
            state=copy.deepcopy(self.frames[-1]);state['time']=self.time
            if fault=='stale_endpoint': state['time']-=.005
            state['q_right'][:7]=executed.end_q()['right']
            if fault=='endpoint_rotation': state['q_right'][0]+=.016
            if fault=='endpoint_position': state['tcp_right'][0]+=.003
            if fault=='endpoint_low_bear': state['bear'][:,2]=.78
            self.frames.append(state);self.qs=executed.end_q()
        def move_guarded(self,label,stage): events.append(('transport',label,self.time,id(self.contact_guard)))
        def run_staged_contact_insertion(self,choice): events.append(('staged',self.time,id(self.contact_guard)))
    with tempfile.TemporaryDirectory() as directory:
        sim=Sim(Path(directory));error=None
        try: namespace['run_guarded_contact_trial'](sim)
        except Exception as exc: error=type(exc).__name__+': '+str(exc)
        return dict(error=error,events=events,baseline_count=sim.baseline_count,
                    execution=sim.report.get('postlift_yaw_execution'),legacy=legacy)


results={}
faults=[None,'rejected','missing_plan','wrong_kind','wrong_parameters','stale_plan','claims_success',
        'claims_execution','mutates_soft','skips_fresh','failed_check','missing_check','empty_checks','nan_report',
        'wrong_target','wrong_rotation','wrong_duration','short_duration','different_start',
        'different_end','moving_left','no_bag_hold','no_teddy_hold','stale_state','missing_guard',
        'native_stop','wrong_clock','stale_endpoint','endpoint_position','endpoint_rotation',
        'endpoint_physics','endpoint_collision','nan_endpoint_clearance','endpoint_low_bear','context_exception']
for fault in faults:
    result=scenario(fault);results[fault or 'same_plan_fresh_preparation']=result
    if fault is None:
        assert result['error'] is None,result
        assert result['baseline_count']==1
        assert [e[0] for e in result['events']]==['baseline','yaw_plan','execute','prepare','transport','transport','transport','staged']
        assert result['events'][3][1]==35.
        assert result['execution']['status']=='executed_endpoint_passed'
        assert not result['execution']['physical_success']
    else:
        assert result['error'],(fault,result)
        assert not any(e[0]=='prepare' for e in result['events']),(fault,result)
legacy=scenario(legacy=True);results['legacy_no_yaw']=legacy
assert legacy['error'] is None and legacy['baseline_count']==1
assert [e[0] for e in legacy['events']]==['prepare','baseline','transport','transport','transport','staged']
report=dict(passed=True,case_count=len(results),source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    scope='Mock runner control-flow only; no physical task success claim.',results=results)
(HERE/'runtime_ipc/postlift-yaw-flow-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='results'}))
