"""Bounded CPU-only contract probes; never starts a simulator or edits runtime."""
from pathlib import Path
import ast, copy, hashlib, json, sys, types
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
source=(ROOT/'staged_contact_insertion.py').read_text()
runner_source=(ROOT/'run.py').read_text()
m=types.ModuleType('independent_staged_snapshot')
exec(compile(source,str(ROOT/'staged_contact_insertion.py'),'exec'),m.__dict__)
cfg=json.loads((ROOT/'runs/two_panel_full_17/config.json').read_text())
cfg['task']['contact_trial']['staged']['center_offsets_xy_m']=[[-.015,-.055],[-.02,-.05]]
results=dict(kind='independent_adaptive_center_contract_audit',runtime_execution_performed=False,
 source_sha256=hashlib.sha256(source.encode()).hexdigest(),
 runner_sha256=hashlib.sha256(runner_source.encode()).hexdigest(),checks=[])
def record(name,**kw):results['checks'].append(dict(name=name,**kw))

for label,value in [('empty',[]),('none',None),('flat',[.01,.02]),('ragged',[[.01,.02],[.02]]),
 ('duplicate',[[.01,.02],[.01,.02]]),('zero',[[0,0]]),('nonfinite',[[float('nan'),.01]]),
 ('nine',[[.001*i,.01] for i in range(1,10)]),('over_budget',[[1.,0.]])]:
 c=copy.deepcopy(cfg);c['task']['contact_trial']['staged']['center_offsets_xy_m']=value
 try:r=m.plan_staged_contact({}, {}, c,phase='center');record('invalid_offsets_'+label,status=r['status'],reasons=r['report']['rejections'])
 except Exception as exc:record('invalid_offsets_'+label,exception=type(exc).__name__,message=str(exc))

# Reproduce the 201-versus-101 point acceptance gap using a deterministic stub.
class FakePlan:
 duration=1.
 def sample(self,t):
  u=float(m.minimum_jerk(t)[0]);return {'q':{'right':np.array([u]),'left':np.zeros(7)},'finger':{'left':.01,'right':.02}}
 def end_q(self):return {'right':np.zeros(7),'left':np.zeros(7)}
 def joint_extrema(self):return {'synthetic':True}
class FakePlanner:
 suspect=.0075
 def trajectory(self,*args,**kw):
  self.preflight_min=min(self.clearance(FakePlan().sample(t)['q'])['arms']['distance'] for t in np.linspace(0,1,kw['validate_samples']))
  assert self.preflight_min>=.008
  return FakePlan()
 def fk(self,side,q):return np.array([q[0],0.,0.]),Rotation.identity()
 def clearance(self,qs,*args):
  # t=.505 is an odd 201-point time and absent from 101-point times.
  near=abs(qs['right'][0]-float(m.minimum_jerk(.505)[0]))<1e-10
  return {'arms':{'distance':self.suspect if near else .02}}
for suspect in [.0075,float('nan'),float('inf')]:
 p=FakePlanner();p.suspect=suspect
 try:
  _,ev,_=m._robot_segment({'planner':p,'qs':{'right':np.zeros(7),'left':np.zeros(7)},'fingers':{'left':.01,'right':.02}},
   {'tcp_right':np.zeros(3)},cfg,np.array([1.,0.,0.]),'center',Rotation.identity())
  record('robot_201_point_'+str(suspect),returned=True,preflight_101_minimum_m=p.preflight_min,reported_201_minimum_m=ev['minimum_sampled_rigid_clearance_m'])
 except Exception as exc:record('robot_201_point_'+str(suspect),returned=False,exception=type(exc).__name__,message=str(exc))

actual=dict(time=np.array(10.),tcp_right=np.array([0.,0.,1.]),tcp_left=np.array([-.1,0.,1.]),
 q_right=np.array([.1,-.2,.3,-2.,.4,1.8,.7,.027,.028]),q_left=np.arange(9,dtype=float),
 bag=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),bear=np.ones((4,3)))
before={k:v.copy() for k,v in actual.items()}
original_core=m._plan_staged_contact_single
def controlled_core(r,s,c,*,phase='entry',previous=None,output=None,center_delta_override=None):
 if phase=='center':
  delta=np.r_[center_delta_override,0.];target=s['tcp_right']+delta
  return dict(status='candidate',phase='center',stage=dict(right_tcp_m=target.tolist(),right_q_end=[.2,-.3,.4,-2.1,.5,1.9,.8],duration_s=4.),
   selected_center_offset_xy_m=list(center_delta_override),observed_time_s=float(s['time']),
   report=dict(rejections=[],robot_trajectory={'minimum_sampled_rigid_clearance_m':controlled_core.clearance},
    planned_com_contact_envelope_boundary_distance_m=.01))
 assert phase=='insert'
 controlled_core.calls.append(dict(time=float(s['time']),bear_shift=(s['bear']-actual['bear']).tolist(),
  left_unchanged=bool(np.array_equal(s['q_left'],actual['q_left'])),
  bag_unchanged=bool(np.array_equal(s['bag'],actual['bag'])),
  right_fingers_unchanged=bool(np.array_equal(s['q_right'][7:],actual['q_right'][7:])),
  right_q=s['q_right'].tolist(),same_array_alias=bool(any(np.shares_memory(s[k],actual[k]) for k in actual))))
 return dict(status='candidate',report=dict(rejections=[],robot_trajectory={'minimum_sampled_rigid_clearance_m':controlled_core.insert_clearance},
  release_withdraw_preflight={'minimum_release_opening_clearance_m':controlled_core.release_clearance}))
controlled_core.calls=[]
controlled_core.insert_clearance=.02
controlled_core.release_clearance=.02
m._plan_staged_contact_single=controlled_core
for val in [.0075,float('nan'),float('inf'),.02]:
 controlled_core.clearance=val
 try:
  out=m.plan_staged_contact({},actual,cfg,phase='center',previous={})
  record('wrapper_clearance_'+str(val),status=out['status'],selected_index=out['center_selection']['selected_index'],
   scores=[a.get('score_m') for a in out['center_selection']['attempts']],
   only_center_stage=out.get('phase')=='center',contains_executable_remainder_stage=any('stage' in a for a in out['center_selection']['attempts']))
 except Exception as exc:record('wrapper_clearance_'+str(val),exception=type(exc).__name__,message=str(exc))
for component in ['insert','release']:
 for val in [float('nan'),float('inf'),.0075]:
  controlled_core.clearance=.02;controlled_core.insert_clearance=.02;controlled_core.release_clearance=.02
  setattr(controlled_core,component+'_clearance',val)
  out=m.plan_staged_contact({},actual,cfg,phase='center',previous={})
  record('wrapper_'+component+'_raw_'+str(val),status=out['status'])
controlled_core.insert_clearance=.02;controlled_core.release_clearance=.02
record('state_isolation',actual_unchanged=bool(all(np.array_equal(v,before[k]) for k,v in actual.items())),hypothetical_calls=controlled_core.calls)
for label,mode in [('missing_simulation','missing'),('missing_dt','dtmissing'),('string_dt','string'),('nan_dt','nan'),('zero_dt','zero')]:
 c=copy.deepcopy(cfg)
 if mode=='missing':c.pop('simulation')
 elif mode=='dtmissing':c['simulation'].pop('dt')
 else:c['simulation']['dt']={'string':'bad','nan':float('nan'),'zero':0.}[mode]
 controlled_core.clearance=.02
 try:r=m.plan_staged_contact({},actual,c,phase='center',previous={});record(label,status=r['status'],reasons=r['report']['rejections'])
 except Exception as exc:record(label,exception=type(exc).__name__,message=str(exc))
m._plan_staged_contact_single=original_core

# The core accepts a list-only policy with an implicit legacy default, whereas
# the current runner's dict.get default eagerly indexes the missing legacy key.
c=copy.deepcopy(cfg);s=c['task']['contact_trial']['staged'];s.pop('center_offset_xy_m',None)
try:m.staged_policy(c);record('list_only_staged_policy',accepted=True)
except Exception as exc:record('list_only_staged_policy',accepted=False,exception=type(exc).__name__,message=str(exc))
try:
 runner_ast=ast.parse(runner_source)
 assigned=[n for n in ast.walk(runner_ast) if isinstance(n,ast.Assign) and
  any(isinstance(t,ast.Name) and t.id=='configured' for t in n.targets)]
 assert len(assigned)==1
 actual_expression=ast.Expression(assigned[0].value)
 configured=eval(compile(actual_expression,str(ROOT/'run.py'),'eval'),{'np':np,'settings':s})
 record('runner_list_only_current_expression',accepted=bool(np.array_equal(configured,s['center_offsets_xy_m'])),
  actual_expression=ast.get_source_segment(runner_source,assigned[0].value))
except Exception as exc:record('runner_list_only_current_expression',accepted=False,exception=type(exc).__name__,message=str(exc))

# Exercise real phase chronology and policy checks while stubbing only expensive
# physical/geometry calculations. No stub pretends that dynamics succeeded.
class ContractPlanner:
 raw=.02
 def clearance(self,*a,**k):return {} if self.raw=='empty' else {'synthetic':{'distance':self.raw}}
 def trajectory(self,*a,**k):return FakePlan()
contract_planner=ContractPlanner()
def context(r,s,c,b):
 return dict(time=float(s['time']),rotation=Rotation.identity(),physical={},reasons=[],
  com=np.mean(s['bear'],axis=0),planner=contract_planner,
  qs={side:np.asarray(s['q_'+side])[:7] for side in ['left','right']},fingers={'left':.01,'right':.02})
def contract_segment(ct,s,c,target,phase,rotation):
 stage=dict(name=m.PHASE_NAMES[phase],right_tcp_m=np.asarray(target).tolist(),rotation_xyzw=rotation.as_quat().tolist(),
  duration_s=4. if phase=='center' else 6.,right_q_end=np.asarray(s['q_right'])[:7].tolist())
 return stage,{'minimum_sampled_rigid_clearance_m':.02},FakePlan()
m._state_context=context
m._staged_mouth=lambda *a:(np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]),[],
 dict(rim_z_min_m=1.05,rim_z_max_m=1.1,available_nonstretch_rim_perimeter_m=2.,representation='spatial_loop_contact_envelope'))
m.left_hand_minimum_z=lambda *a,**k:2.
m._rim_envelope_review=lambda *a,**k:dict(section_guard_enforced=True,section_guard_rejections=[])
m.check_translation_sweep=lambda *a,**k:dict(status='clear')
m._robot_segment=contract_segment
st=copy.deepcopy(actual);st['tcp_right']=np.array([0.,0.,1.2]);st['bear']=np.array([[0.,0.,.9],[.01,0.,.9],[0.,.01,.9],[0.,0.,1.]])
reference=dict(bear_tets=np.array([[0,1,2,3]]))
policy=m.staged_policy(cfg);base=m.base_policy(cfg)
prior=dict(kind='staged_guarded_contact_trial',status='candidate',phase='entry',observed_time_s=9.,
 stage=dict(right_tcp_m=st['tcp_right'].tolist()),cumulative_xy_m_before=0.,cumulative_xy_m_after=0.,
 checkpoint=dict(staged_policy=policy,base_policy=base,observed_tcp_world_m=[0.,0.,1.4],
 stage_target_tcp_world_m=st['tcp_right'].tolist(),original_transport_tcp_xy_m=[0.,0.],
 cumulative_xy_m_before=0.,planned_xy_distance_m=0.,planned_cumulative_xy_m_after=0.,
 original_rotation_xyzw=[0.,0.,0.,1.],actual_stage_times_s=[9.]))
center=m._plan_staged_contact_single(reference,st,cfg,phase='center',previous=prior,center_delta_override=[-.015,-.055])
record('real_core_valid_center_contract',status=center['status'],reasons=center['report']['rejections'])
if center['status']=='candidate':
 next_state=copy.deepcopy(st);next_state['time']=np.array(11.);target=np.asarray(center['stage']['right_tcp_m'])
 next_state['bear']+=target-next_state['tcp_right'];next_state['tcp_right']=target.copy()
 for mode in ['fresh','stale_time','changed_list','changed_order','missing_list','missing_selected','wrong_allowed_selected','unreached_endpoint']:
  cc=copy.deepcopy(cfg);ss=copy.deepcopy(next_state);pp=copy.deepcopy(center)
  if mode=='stale_time':ss['time']=np.array(10.)
  elif mode=='changed_list':cc['task']['contact_trial']['staged']['center_offsets_xy_m'][0]=[-.014,-.055]
  elif mode=='changed_order':cc['task']['contact_trial']['staged']['center_offsets_xy_m'].reverse()
  elif mode=='missing_list':cc['task']['contact_trial']['staged'].pop('center_offsets_xy_m')
  elif mode=='missing_selected':pp.pop('selected_center_offset_xy_m')
  elif mode=='wrong_allowed_selected':pp['selected_center_offset_xy_m']=[-.02,-.05]
  elif mode=='unreached_endpoint':ss['tcp_right'][0]+=.003
  result=m._plan_staged_contact_single(reference,ss,cc,phase='insert',previous=pp)
  record('real_core_insert_contract_'+mode,status=result['status'],reasons=result['report']['rejections'])
 for value in [float('nan'),float('inf'),'empty',.0075]:
  contract_planner.raw=value
  result=m._plan_staged_contact_single(reference,next_state,cfg,phase='insert',previous=center)
  record('real_core_release_raw_'+str(value),status=result['status'],reasons=result['report']['rejections'])

failures=[]
for row in results['checks']:
 name=row['name']
 if name.startswith('robot_201_point_'):passed=row.get('returned') is False
 elif name=='state_isolation':
  passed=row['actual_unchanged'] and all(x['left_unchanged'] and x['bag_unchanged'] and x['right_fingers_unchanged'] and not x['same_array_alias'] and x['time']>10. for x in row['hypothetical_calls'])
 elif name in ['list_only_staged_policy','runner_list_only_current_expression']:passed=row.get('accepted') is True
 elif name in ['wrapper_clearance_0.02','real_core_valid_center_contract','real_core_insert_contract_fresh']:passed=row.get('status')=='candidate'
 else:passed=row.get('status')=='rejected'
 row['expected_contract_pass']=bool(passed)
 if not passed:failures.append(name)
results.update(check_count=len(results['checks']),all_expected_contracts_pass=not failures,remaining_failures=failures,
 limitations='Bounded CPU contract regressions with explicit geometry stubs; no dynamic or full task success is asserted.')

out=ROOT/'runtime_ipc/independent-adaptive-center-audit.json'
out.write_text(json.dumps(results,indent=2,allow_nan=True)+'\n')
for row in results['checks']:
 print({k:v for k,v in row.items() if k!='hypothetical_calls'})
print(out)
