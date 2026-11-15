"""Independent CPU input, exact-plan, continuous-rate and fail-closed audit."""
import copy,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'runtime_ipc')]
import postlift_yaw as h
from motion import Planner,MotionPlan,CubicSpline
from postlift_yaw_regression import replay_lift_command
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((ROOT/'config_guarded_adaptive_alignment_trial.json').read_text());cfg['task']['contact_trial']['postlift_yaw']={'world_z_degrees':15.,'duration_s':6.}
source_before=sha(ROOT/'postlift_yaw.py');checks=[];cases=[];failures=[]
def check(name,ok,detail=None):
 row=dict(name=name,passed=bool(ok));
 if detail is not None:row['detail']=detail
 checks.append(row)
 if not ok:failures.append(row)
def digest(rest,state,config,commands):
 result={}
 for prefix,group in [('rest',rest),('state',state),('command',commands)]:
  for k,v in group.items():result[prefix+'.'+k]=hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
 result['config']=json.dumps(config,sort_keys=True,allow_nan=False);return result
for number in [17,18]:
 run=ROOT/f'runs/two_panel_full_{number}';rest=dict(np.load(run/'rest.npz',allow_pickle=False));state=dict(np.load(run/'lift_teddy.npz',allow_pickle=False));commands,replay=replay_lift_command(run,json.loads((run/'config.json').read_text()));before=digest(rest,state,cfg,commands)
 created=[]
 def tracked(*a,**kw):
  plan=MotionPlan(*a,**kw);created.append(plan);return plan
 with patch.object(h,'MotionPlan',tracked):report,plan=h.plan_postlift_yaw(rest,state,cfg,commands)
 check(f'{number} candidate',report['status']=='candidate' and plan is not None,report['rejections'])
 check(f'{number} exact returned object',len(created)==1 and plan is created[0])
 check(f'{number} inputs preserved',before==digest(rest,state,cfg,commands))
 check(f'{number} finite report',bool(json.dumps(report,allow_nan=False)))
 if plan is not None:
  check(f'{number} exact command continuity',all(np.array_equal(plan.sample(0)['q'][s],commands[s]) for s in commands))
  check(f'{number} exact endpoint contract',all(np.array_equal(plan.end_q()[s],report[s+'_q_end']) for s in commands))
  planner=Planner(cfg);_,A=planner.fk('right',state['q_right'][:7]);T=Rotation.from_euler('z',15,degrees=True)*A
  check(f'{number} full actual world-left rotation',float((Rotation.from_quat(report['target_rotation_xyzw'])*T.inv()).magnitude())<1e-12)
  check(f'{number} measured TCP fixed',np.array_equal(report['target_tcp_m'],state['tcp_right']))
  check(f'{number} left command always constant',all(np.array_equal(plan.sample(float(t))['q']['left'],commands['left']) for t in np.linspace(0,plan.duration,1001)))
  rates=report['robot_preflight']['continuous_rate_bounds'];check(f'{number} all continuous bounds meet limits',all(max(v['velocity_upper_bound_rad_s'])<=.35 and max(v['acceleration_upper_bound_rad_s2'])<=.5 for v in rates.values()))
  # Independent exact derivative extrema are checked densely as a consistency test.
  maximum_v=maximum_a=0
  for t in np.linspace(0,plan.duration,2001):
   point=plan.sample(float(t));maximum_v=max(maximum_v,*[float(np.abs(point['qd'][s]).max()) for s in commands]);maximum_a=max(maximum_a,*[float(np.abs(point['qdd'][s]).max()) for s in commands])
  check(f'{number} sampled derivatives below continuous certificates',all(maximum_v<=max(v['velocity_upper_bound_rad_s'])+1e-12 for s,v in rates.items() if s=='right') and maximum_a<=max(rates['right']['acceleration_upper_bound_rad_s2']))
  check(f'{number} soft both left poses complete',set(p['pose'] for p in report['frozen_bear_left_rotation_sweep']['per_pose'])=={'actual','commanded'} and all(len(seg['per_geom'])==22 for p in report['frozen_bear_left_rotation_sweep']['per_pose'] for seg in p['segments']))
  cases.append(dict(run=number,physical_success=False,helper_source_sha256=source_before,command_prefix_replay=replay,command_qs={s:v.tolist() for s,v in commands.items()},duration_s=plan.duration,
    actual_command_pose_errors=report['command_source_poses'],minimum_rigid_m=report['robot_preflight']['minimum_sampled_rigid_clearance_m'],minimum_ideal_soft_sweep_m=report['frozen_bear_left_rotation_sweep']['minimum_continuous_gap_lower_bound_m'],continuous_rate_bounds=rates))
# Last fixture: each test is isolated; no simulator or original arrays changed.
def rejected(name,s=None,commands_override=None,context=None):
 args=(rest,state if s is None else s,cfg,commands if commands_override is None else commands_override)
 if context is None:r,p=h.plan_postlift_yaw(*args)
 else:
  with context:r,p=h.plan_postlift_yaw(*args)
 check(name,r['status']=='rejected' and p is None and bool(r['rejections']) and bool(json.dumps(r,allow_nan=False)),r['rejections'])
for key in ['bag','bear']:
 bad={k:v.copy() for k,v in state.items()};bad[key][0,0]=float('nan');rejected(key+' nonfinite rejected',s=bad)
bad={k:v.copy() for k,v in state.items()};bad['bag']=state['tcp_left']+(state['bag']-state['tcp_left'])*1.3;rejected('original cloth stretch gate',s=bad)
bad={k:v.copy() for k,v in state.items()};bad['bag'][:,0]+=.1;rejected('original designated left material gate',s=bad)
bad={k:v.copy() for k,v in state.items()};bad['bear'][:,0]+=.1;rejected('original right teddy proximity gate',s=bad)
bad={k:v.copy() for k,v in state.items()};tet=rest['bear_tets'][0];bad['bear'][tet[0]]=bad['bear'][tet[1]];rejected('original tetra compression gate',s=bad)
bad={k:v.copy() for k,v in state.items()};bad['tcp_right'][0]+=.003;rejected('original measured TCP FK gate',s=bad)
original=Planner.clearance
for kind in ['missing_arms','nan_hidden_category','infinite_hidden_category','below8_hidden_category']:
 def inject(self,*a,**kw):
  out=original(self,*a,**kw)
  if kind=='missing_arms':out.pop('arms')
  else:out['left_mounts']['distance']={'nan_hidden_category':float('nan'),'infinite_hidden_category':float('inf'),'below8_hidden_category':.007999}[kind]
  return out
 rejected('raw clearance '+kind,context=patch.object(Planner,'clearance',inject))
original_path=Planner.plan_cartesian
for endpoint,value in [('start',1e-8),('finish',1-1e-8)]:
 def incomplete(self,*a,**kw):
  out=original_path(self,*a,**kw);out[0 if endpoint=='start' else -1]['fraction']=value;return out
 rejected('incomplete spline '+endpoint,context=patch.object(Planner,'plan_cartesian',incomplete))
check('helper source frozen during audit',source_before==sha(ROOT/'postlift_yaw.py'))
result=dict(kind='independent_postlift_helper_contract_review',physical_success=False,gpu_used=False,runtime_modified=False,helper_source_sha256=source_before,all_checks_passed=not failures,check_count=len(checks),failed_checks=failures,checks=checks,cases=cases,
 limitations=['MotionPlan object identity and command continuity are verified; actual native finite-force execution is runner-owned and not performed here.',
 'The continuous derivative proof was read independently; consistency sampling is supplementary, not its justification.',
 'Rigid clearance is sampled, ideal frozen tetrahedral yaw sweep is continuous only against the two checked static left poses.'])
p=ROOT/'runtime_ipc/independent-postlift-helper-audit.json';p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'source':source_before,'checks':len(checks),'failures':failures,'cases':[(c['run'],c['duration_s'],c['minimum_rigid_m']) for c in cases]},indent=2))
