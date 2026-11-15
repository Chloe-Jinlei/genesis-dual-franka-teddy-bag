"""Independent AST-only candidate isolation checks; no simulation or physical state mutations."""
from pathlib import Path
import ast,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parents[1]
source=(HERE/'contact_insertion.py').read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='recenter_guarded_contact_trial')
tests=[]
def scenario(mode):
 calls=[];pre=[]
 cfg={'task':{'contact_trial':{'recenter_offsets_xy_m':[[-.01,0],[.01,0]]}}}
 selected={'status':'candidate','route':{'rotation_xyzw':[0,0,0,1]},'observed_time_s':0.,'report':{'policy':{'immutable':1}}}
 state={'time':1.,'tcp_right':np.array([0.,0.,1.])};before=state['tcp_right'].copy()
 def policy(config):
  pre.append('policy')
  if mode=='invalid_global_policy':raise ValueError('invalid global policy')
  return {'immutable':1}
 def context(*args):
  pre.append('context')
  if mode=='invalid_global_state':raise ValueError('invalid global state')
  return {'reasons':['physical gate failed'] if mode=='physical_reasons' else [],'rotation':Rotation.identity()}
 def mouth(*args):
  pre.append('mouth')
  if mode=='global_mouth_failure':raise ValueError('bad topology')
 def plan(rest,state,config,*,correction_offsets):
  calls.append(correction_offsets[0].tolist())
  if len(calls)==1:
   if mode in ('candidate_value_error_then_pass','all_candidate_value_error'):raise ValueError('target impossible')
   if mode=='candidate_key_error':raise KeyError('corrupt source')
   if mode=='candidate_type_error':raise TypeError('invalid data')
  if mode=='all_candidate_value_error':raise ValueError('target also impossible')
  if len(calls)==1 and mode=='ordinary_rejection_then_pass':return {'status':'rejected','report':{'rejections':['outside COM'],'observed_center_of_mass_world_m':[0,0,1]},'route':None,'stages_to_execute':[]}
  return {'status':'candidate','report':{'rejections':[]},'route':{},'stages_to_execute':[{'name':'height_at_current_xy'},{'name':'translate_outside_mouth'}]}
 if mode=='changed_policy':selected['report']['policy']={'immutable':2}
 if mode=='invalid_offset':cfg['task']['contact_trial']['recenter_offsets_xy_m']=[[.031,0]]
 if mode=='stale_observation':state['time']=0.
 if mode=='bad_counter':index=3
 else:index=1
 namespace={'np':np,'Rotation':Rotation,'_policy':policy,'_state_context':context,'_mouth':mouth,'_plan':plan,'_emit':lambda result,output:result,'_validate_recenter_record':lambda selected:None}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'isolated_candidate_recenter','exec'),namespace)
 result=namespace['recenter_guarded_contact_trial']({},state,cfg,selected,attempt_index=index,cumulative_xy_m=0.)
 assert np.array_equal(before,state['tcp_right'])
 return result,calls,pre
for mode in ('candidate_value_error_then_pass','ordinary_rejection_then_pass'):
 r,c,p=scenario(mode);ok=r['status']=='candidate' and len(c)==2 and r['correction_xy_m']==[.01,0] and r['correction_distance_m']==.01 and len(r['report']['recenter']['candidates'])==2
 if mode=='ordinary_rejection_then_pass':ok=ok and 'planning_report' in r['report']['recenter']['candidates'][0]
 if mode=='candidate_value_error_then_pass':ok=ok and r['report']['recenter']['candidates'][0]['candidate_planning_exception']=='ValueError'
 tests.append(dict(name=mode,passed=ok,plan_calls=len(c)));assert ok,mode
r,c,p=scenario('all_candidate_value_error');ok=r['status']=='rejected' and r['route'] is None and not r['stages_to_execute'] and len(c)==2 and len(r['report']['candidates'])==2;tests.append(dict(name='all_candidate_errors_rejected_after_full_search',passed=ok));assert ok
for mode in ('invalid_global_policy','invalid_global_state','physical_reasons','global_mouth_failure','changed_policy','invalid_offset','stale_observation','bad_counter'):
 r,c,p=scenario(mode);ok=r['status']=='rejected' and r['route'] is None and not r['stages_to_execute'] and not c;tests.append(dict(name=mode,passed=ok,plan_calls=len(c)));assert ok,mode
for mode in ('candidate_key_error','candidate_type_error'):
 r,c,p=scenario(mode);ok=r['status']=='rejected' and len(c)==1 and not r['stages_to_execute'];tests.append(dict(name=mode+'_not_swallowed_as_candidate_infeasibility',passed=ok));assert ok,mode
out=dict(kind='independent_recenter_candidate_error_isolation_review',all_checks_passed=all(t['passed'] for t in tests),tests=tests,source_sha256=hashlib.sha256(source.encode()).hexdigest(),static_review={'state_policy_and_global_mouth_validated_before_candidates':True,'candidate_value_error_only_continues_local_search':True,'all_candidates_rejected_returns_no_executable_stages':True,'ordinary_no_com_infeasibility_keeps_mouth_physical_and_com_checks':True,'budget_counter_original_rotation_and_fresh_observation_contract_unchanged':True},scope='AST-executed control flow with deterministic stubs; no GPU, no Genesis initialization, no runtime file modifications.',physical_success=False)
(HERE/'runtime_ipc/independent-recenter-candidate-isolation-review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
