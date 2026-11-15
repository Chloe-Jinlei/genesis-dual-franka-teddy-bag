"""Independent bounded reconstruction of two high-yaw hypotheses and selected routes."""
from pathlib import Path
import json,hashlib,sys,math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq
from numpy.polynomial import Polynomial as Poly
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion import Planner,MotionPlan,CubicSpline,minimum_jerk
from contact_insertion import _policy
from contact_robot_geometry import _convex,convex_clearance_certificate,left_collision_solids,check_translation_sweep
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfgpath=ROOT/'config_guarded_adaptive_alignment_trial.json';cfg=json.loads(cfgpath.read_text())
protected=[cfgpath,ROOT/'runtime_ipc/postlift_yaw15_preparation.py',*ROOT.glob('*.py')]
before={str(p):sha(p) for p in protected};rows=[];checks=[]
def check(name,condition):
 checks.append(dict(name=name,passed=bool(condition)))
 if not condition:raise AssertionError(name)
def max_bernstein(p):
 n=p.degree();b=[sum(p.coef[i]*math.comb(k,i)/math.comb(n,i) for i in range(k+1)) for k in range(n+1)]
 return max(abs(v) for v in b)
def continuous_rates(plan):
 # On each clock interval q(minimum_jerk(u)) is a degree <=15 polynomial.
 # Derivatives' Bernstein coefficients enclose all values, not just samples.
 velocity=acceleration=0.
 inv=lambda s:0. if s<=0 else 1. if s>=1 else brentq(lambda u:float(minimum_jerk(u)[0])-s,0.,1.,xtol=5e-15)
 for side,curve in plan.curves.items():
  for j in range(len(curve.x)-1):
   ua,ub=inv(curve.x[j]),inv(curve.x[j+1])
   for a,b in zip(np.linspace(ua,ub,9)[:-1],np.linspace(ua,ub,9)[1:]):
    u=Poly([a,b-a]);v=10*u**3-15*u**4+6*u**5-curve.x[j]
    for k in range(curve.c.shape[2]):
     c=curve.c[:,j,k];q=c[0]*v**3+c[1]*v**2+c[2]*v+c[3]
     velocity=max(velocity,max_bernstein(q.deriv())/((b-a)*plan.duration))
     acceleration=max(acceleration,max_bernstein(q.deriv(2))/((b-a)*plan.duration)**2)
 return dict(continuous_velocity_outer_bound_rad_s=velocity+1e-9,continuous_acceleration_outer_bound_rad_s2=acceleration+1e-9,
  method='Degree <=15 polynomial per spline segment composed with minimum jerk, eight clock subintervals; derivative Bernstein convex hull; numerical 1e-9 padding, not directed-rounding interval arithmetic')
def clearance(plan,planner,n=401):
 minimum=float('inf');count=0
 for t in np.linspace(0,plan.duration,n):
  s=plan.sample(float(t));v=np.asarray([x['distance'] for x in planner.clearance(s['q'],s['finger']).values()]);check('raw finite',v.size and np.isfinite(v).all());minimum=min(minimum,float(v.min()));count+=len(v)
 return dict(samples=n,raw_distance_count=count,all_raw_finite=True,minimum_m=minimum)
for number in [17,18]:
 run=ROOT/f'runs/two_panel_full_{number}';out=run/'independent-postlift-yaw15';r=json.loads((out/'review.json').read_text());prep=json.loads((out/'preparation.json').read_text())
 for p in [run/'lift_teddy.npz',run/'rest.npz',out/'review.json',out/'preparation.json',out/'hypothetical_after_yaw.npz',out/'yaw_path_samples.npz']:
  before[str(p)]=sha(p)
 state=dict(np.load(run/'lift_teddy.npz',allow_pickle=False));hyp=dict(np.load(out/'hypothetical_after_yaw.npz',allow_pickle=False));rest=dict(np.load(run/'rest.npz',allow_pickle=False));planner=Planner(cfg)
 check(f'{number} all reported protected hashes',all(sha(Path(p))==h for p,h in r['input_sha256'].items()))
 check(f'{number} no force inheritance',set(hyp)=={'bag','bear','q_left','q_right','tcp_left','tcp_right','time'})
 check(f'{number} bag unchanged',np.array_equal(hyp['bag'],state['bag']))
 check(f'{number} left q and fingers unchanged',np.array_equal(hyp['q_left'],state['q_left']) and np.array_equal(hyp['q_right'][7:],state['q_right'][7:]))
 p0,A=planner.fk('right',state['q_right'][:7]);p1,B=planner.fk('right',hyp['q_right'][:7]);C=Rotation.from_quat(r['original_command_rotation_xyzw']);target=Rotation.from_euler('z',15,degrees=True)*C
 check(f'{number} full18 fallback is actual initial FK',float((C*A.inv()).magnitude())<1e-12)
 exact=(B*A.inv()).apply(state['bear']-p0)+p1
 transform_error=float(np.linalg.norm(exact-hyp['bear'],axis=1).max());check(f'{number} actual FK transform',transform_error<2e-15)
 check(f'{number} correct world left multiplication',float((B*target.inv()).magnitude())<.002)
 construction=r['hypothetical_construction'];check(f'{number} integer clock',construction['clock_increment_steps']==math.ceil(r['yaw_preflight']['duration_s']/cfg['simulation']['dt']) and abs(float(hyp['time'])-(float(state['time'])+construction['clock_increment_steps']*cfg['simulation']['dt']))<1e-12)
 check(f'{number} unchanged policy',prep['report']['policy']==_policy(cfg))
 offsets=cfg['task']['contact_trial']['center_offsets_xy_m'];check(f'{number} unchanged 12 declared candidates',len(offsets)==12 and prep['report']['declared_com_offsets_from_mouth_area_centroid_m']==offsets and r['unchanged_initial_offsets_xy_m']==offsets and len(prep['report']['centering_com_checks'])==12)
 check(f'{number} prepared only 3 stages',prep['status']=='candidate' and len(prep['stages_to_execute'])==3 and all(s['name']!='insert_candidate' for s in prep['stages_to_execute']) and prep['next_observation']=='before_insertion')
 check(f'{number} mandatory physical gates retained',all(prep[k] for k in ['requires_physical_validation','requires_per_step_native_contact_force_torque_grasp_and_deformation_gates','requires_final_actual_3d_containment_audit']))
 q0={s:state['q_'+s][:7] for s in ['left','right']};fingers={s:float(state['q_'+s][7:].mean()) for s in q0}
 paths=planner.plan_cartesian(q0,{'right':state['tcp_right']},rotations={'right':target},samples=45);fractions=np.array([k['fraction'] for k in paths]);curves={s:CubicSpline(fractions,np.array([k['q'].get(s,q0[s]) for k in paths]),axis=0) for s in q0}
 yaw=MotionPlan(curves,6.,fingers,fingers);yaw.validate_joint_limits(planner.limits);yaw.limit_rate(.35,.5)
 saved=np.load(out/'yaw_path_samples.npz',allow_pickle=False)['samples'];actual=[]
 for t in np.linspace(0,yaw.duration,401):
  z=yaw.sample(float(t));p,R=planner.fk('right',z['q']['right']);actual.append(np.r_[t,z['q']['left'],z['q']['right'],p,R.as_quat()])
 sample_error=float(np.abs(np.asarray(actual)-saved).max());check(f'{number} entire saved sample replay',sample_error<1e-14)
 newclear=clearance(yaw,planner);check(f'{number} yaw 8mm',newclear['minimum_m']>=.008)
 rates=continuous_rates(yaw);check(f'{number} continuous yaw velocity acceleration',rates['continuous_velocity_outer_bound_rad_s']<=.35 and rates['continuous_acceleration_outer_bound_rad_s2']<=.5)
 rotvec=(B*A.inv()).as_rotvec();angle=float(np.linalg.norm(rotvec));axis=rotvec/angle;rel=state['bear']-state['tcp_right'];radius=float(np.linalg.norm(rel-np.outer(rel@axis,axis),axis=1).max());segments=max(1,int(np.ceil(angle/np.deg2rad(5))));sagitta=radius*(1-np.cos(angle/segments/2));translation=float(np.linalg.norm(p1-state['tcp_right']));minimum=float('inf');solid_count=0
 for j in range(segments):
  hull=_convex(np.vstack([Rotation.from_rotvec(rotvec*f).apply(rel)+state['tcp_right'] for f in [j/segments,(j+1)/segments]]))
  for solid in left_collision_solids(planner,state['q_left']):
   cert=convex_clearance_certificate(hull,solid,required_gap_m=.005+sagitta+translation);check(f'{number} 5mm arc certificate',cert['status']=='clear');minimum=min(minimum,cert['conservative_gap_lower_bound_m']-sagitta-translation);solid_count+=1
 check(f'{number} arc exact replay',abs(minimum-r['yaw_preflight']['rotation_left_soft_sweep']['minimum_continuous_frozen_arc_lower_bound_m'])<1e-12)
 qs={s:hyp['q_'+s][:7].copy() for s in q0};current=hyp['tcp_right'].copy();start=current.copy();rotation=B;stage_rows=[]
 base_durations={'height_at_current_xy':2.5,'transport_y_at_current_x':3.,'translate_outside_mouth':3.,'insert_candidate':6.}
 for stage in prep['route']['stages']:
  name=stage['name'];dest=np.asarray(stage['right_tcp_m']);plan=planner.trajectory(qs,{'right':dest},rotations={'right':rotation},samples=45,validate_samples=101,duration=base_durations[name],finger_start=fingers,finger_end=fingers,clearance_margin=.008)
  check(f'{number} selected stage exact q replay',np.max(np.abs(plan.end_q()['right']-stage['right_q_end']))<1e-12 and abs(plan.duration-stage['duration_s'])<1e-12)
  raw=clearance(plan,planner);check(f'{number} selected stage 8mm',raw['minimum_m']>=.008)
  sweep=check_translation_sweep(hyp['bear'],rest['bear_tets'],hyp['q_left'],cfg,translation_m=current-start,displacement_m=dest-current,required_gap_m=.005,planner=planner);check(f'{number} selected stage 5mm',sweep['status']=='clear')
  bound=continuous_rates(plan);stage_rows.append(dict(name=name,rigid=raw,rates=bound,continuous_rate_bounds_within_original_limits=bound['continuous_velocity_outer_bound_rad_s']<=.55 and bound['continuous_acceleration_outer_bound_rad_s2']<=1.,soft_sweep_status=sweep['status']))
  qs=plan.end_q();current=dest
 release_min=float('inf');opened={**fingers,'right':.04}
 for f in np.linspace(0,1,11):
  v=np.asarray([d['distance'] for d in planner.clearance(qs,{**fingers,'right':(1-f)*fingers['right']+f*.04}).values()]);check(f'{number} release raw finite',np.isfinite(v).all());release_min=min(release_min,float(v.min()))
 check(f'{number} release 8mm',release_min>=.008)
 stage=prep['route']['withdraw_stage'];plan=planner.trajectory(qs,{'right':np.asarray(stage['right_tcp_m'])},rotations={'right':rotation},samples=45,validate_samples=101,duration=3.,finger_start=opened,finger_end=opened,clearance_margin=.008)
 check(f'{number} withdraw q replay',np.max(np.abs(plan.end_q()['right']-stage['right_q_end']))<1e-12)
 withdraw_clear=clearance(plan,planner);check(f'{number} withdraw 8mm',withdraw_clear['minimum_m']>=.008)
 rows.append(dict(run=number,physical_success=False,source_time_s=float(state['time']),hypothetical_time_s=float(hyp['time']),actual_fk_transform_max_error_m=transform_error,saved_yaw_sample_max_error=sample_error,
  yaw_rigid=newclear,yaw_continuous_rate_bounds=rates,arc_tetrahedral_allpoint_lower_bound_m=minimum,arc_solid_pairs=solid_count,selected_preparation_offset_m=r['preparation']['selected_initial_com_offset_xy_m'],selected_route=stage_rows,release_minimum_m=release_min,withdraw_rigid=withdraw_clear,
  all_selected_route_continuous_rate_bounds_within_original_limits=all(x['continuous_rate_bounds_within_original_limits'] for x in stage_rows)))
check('all protected inputs unchanged',before=={p:sha(Path(p)) for p in before})
result=dict(kind='independent_two_case_postlift_yaw15_review',physical_success=False,gpu_used=False,runtime_config_modified=False,passed=True,check_count=len(checks),failed_checks=[c for c in checks if not c['passed']],rows=rows,input_sha256=before,
 findings=[dict(severity='offline_harness_robustness',description='The reviewed offline yaw harness uses min/max without checking all raw distances/FK outputs finite. Actual independent replay of both cases found every raw distance finite and reproduced all samples; this does not invalidate these two datasets. A reusable yaw helper must reject nonfinite raw values before aggregation.')],
 scope=['Original current policy and exactly 12 initial offsets retained. No rejected preparation route reused; full18 fallback supplies only initial orientation, independently equal to actual FK.',
 'No force or torque readings are carried into the hypothetical state. Bag and original finger observations are unchanged; no physical state transition is certified.',
 'Joint positions are analytically checked on continuous splines. Yaw rate bounds use continuous Bernstein polynomial enclosures; rigid clearance remains sampled, not continuous.',
 'Frozen arbitrary-axis solid arc covers all tetrahedral points against original left robot volumes; endpoint perturbation subtraction is only an ideal bounded translation, not an actual FK path bound.',
 'The three preparation stages require fresh native observation before any insertion; all native and final physical gates remain mandatory.'])
path=ROOT/'runtime_ipc/independent-postlift-yaw15-audit.json';path.write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda x:x.item())+'\n');print(json.dumps({'passed':result['passed'],'checks':len(checks),'rows':rows},indent=2,default=lambda x:x.item()));print(path,sha(path))
