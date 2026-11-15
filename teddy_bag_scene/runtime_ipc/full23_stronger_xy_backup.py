"""Single declared +15/-3.75 mm backup, read-only actual-state CPU preflight."""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_23';OUT=RUN/'independent-stronger-xy-backup';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
import contact_insertion as ci
from motion import Planner, minimum_jerk
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(ci._serial(v),indent=2,allow_nan=False)+'\n')
paths=[RUN/x for x in ['config.json','rest.npz','orient_teddy_postlift.npz','transport_teddy.npz','guarded_contact_candidate.json','source/contact_insertion.py','source/motion.py','source/contact_robot_geometry.py']]
before={str(p):sha(p) for p in paths};cfg=json.loads((RUN/'config.json').read_text());original=copy.deepcopy(cfg)
cfg['task']['contact_trial']['center_offsets_xy_m']=[[.015,.03075]]
rest=dict(np.load(RUN/'rest.npz'));save(OUT/'diagnostic-initial-config.json',cfg)
report={'scope':'Only one new initial COM offset [.015,.03075] and one equivalent high correction [+15,-3.75] mm on real full23 transport shape. CPU counterfactual only; no state/config/production modification or native success.','physical_success':False,'native_execution':False,'input_sha256':before,'initial_offset_radius_m':float(np.linalg.norm([.015,.03075])),'cases':[]}
original_trajectory=Planner.trajectory
captures=[]
def capture(self,*args,**kwargs):
 p=original_trajectory(self,*args,**kwargs)
 captures.append((self,p,copy.deepcopy(args[0]),copy.deepcopy(args[1]),copy.deepcopy(kwargs)))
 return p
Planner.trajectory=capture
required={'left_self','left_table','left_mounts','right_self','right_table','right_mounts','arms'}
def audit_case(name,state,choice,caseout):
 result={'name':name,'source_time_s':float(state['time']),'status':choice['status'],'rejections':choice['report']['rejections'],'physical_success':False,'captured_successful_robot_plans':len(captures),'stages':[]}
 if choice['status']!='candidate':return result
 stages=[*choice['route']['stages'],choice['route']['withdraw_stage']]
 assert len(stages)==len(captures)
 start=state['tcp_right'].copy()
 for i,(stage,cap) in enumerate(zip(stages,captures)):
  planner,p,qs,targets,kwargs=cap;target=np.asarray(stage['right_tcp_m']);R=kwargs['rotations']['right'];opened=stage['name']=='withdraw_after_release'
  assert np.max(abs(p.end_q()['right']-np.asarray(stage['right_q_end'])))<1e-12
  q=copy.deepcopy(p);q.limit_rate(.35,.5);rates=q.continuous_rate_extrema()
  vmax=max(np.max(v['maximum_velocity_rad_s']) for v in rates.values());amax=max(np.max(v['maximum_acceleration_rad_s2']) for v in rates.values())
  assert np.isfinite([vmax,amax]).all() and vmax<=.35 and amax<=.5
  minimum=np.inf;worst=None;pmax=rmax=0.;rows=[]
  for t in np.linspace(0,q.duration,401):
   sample=q.sample(float(t));u=float(minimum_jerk(t/q.duration)[0]);raws={}
   for kind in ['paired_mean','actual_individual']:
    if kind=='actual_individual':
     for side in qs:
      for j,v in enumerate(state['q_'+side][7:],1):planner.d.qpos[planner.m.joint(f'{side}_finger_joint{j}').qposadr[0]]=.04 if opened and side=='right' else v
    raw=planner.clearance(sample['q'],sample['finger'] if kind=='paired_mean' else None)
    assert set(raw)==required and np.isfinite([v['distance'] for v in raw.values()]).all()
    for key,v in raw.items():
     if v['distance']<minimum:minimum=v['distance'];worst={'time_s':float(t),'finger_kind':kind,'category':key,**v}
    raws[kind]=raw
   pos,rot=planner.fk('right',sample['q']['right']);pe=float(np.linalg.norm(pos-(start+u*(target-start))));re=float((rot*R.inv()).magnitude());assert np.isfinite([pe,re]).all();pmax=max(pmax,pe);rmax=max(rmax,re)
   rows.append({'time_s':float(t),'fraction':u,'raw_clearances':raws,'fk_position_error_m':pe,'fk_rotation_error_rad':re})
  prefix=f'{i}_{stage["name"]}';archive=caseout/(prefix+'.npz')
  np.savez_compressed(archive,original_duration_s=p.duration,duration_s=q.duration,left_x=q.curves['left'].x,left_c=q.curves['left'].c,right_x=q.curves['right'].x,right_c=q.curves['right'].c,start_tcp=start,target_tcp=target,rotation_xyzw=R.as_quat(),left_fingers=state['q_left'][7:],right_fingers=np.full(2,.04) if opened else state['q_right'][7:])
  save(caseout/(prefix+'-raw401.json'),rows)
  row={'stage':stage['name'],'original_duration_s':p.duration,'retimed035_05_duration_s':q.duration,'same_curve_coefficients':all(np.array_equal(p.curves[s].c,q.curves[s].c) for s in qs),'minimum_rigid_gap_m':minimum,'worst':worst,'max_sampled_fk_position_error_m':pmax,'max_sampled_full_R_error_rad':rmax,'continuous_vmax_rad_s':float(vmax),'continuous_amax_rad_s2':float(amax),'rates':rates,'passed':bool(minimum>=.008 and pmax<=.0005 and rmax<=.002),'archive':str(archive),'sha256':sha(archive)}
  result['stages'].append(row);save(caseout/'robot-review.json',result);print(name,stage['name'],'rigid',minimum,'rate',vmax,amax,flush=True);start=target.copy()
 # Independent raw finite opening check, both paired and measured unequal fingers.
 planner= captures[-1][0];qs=captures[-1][2];minimum=np.inf;worst=None
 for f in np.linspace(0,1,401):
  paired={side:float(state['q_'+side][7:].mean()) for side in qs};paired['right']=(1-f)*paired['right']+f*.04
  for kind in ['paired_mean','actual_individual']:
   if kind=='actual_individual':
    for side in qs:
     for j,v in enumerate(state['q_'+side][7:],1):planner.d.qpos[planner.m.joint(f'{side}_finger_joint{j}').qposadr[0]]=(1-f)*v+f*.04 if side=='right' else v
   raw=planner.clearance(qs,paired if kind=='paired_mean' else None);assert set(raw)==required and np.isfinite([v['distance'] for v in raw.values()]).all()
   for key,v in raw.items():
    if v['distance']<minimum:minimum=v['distance'];worst={'fraction':float(f),'finger_kind':kind,'category':key,**v}
 result['release_opening']={'sample_count':401,'minimum_rigid_gap_m':minimum,'worst':worst,'passed':minimum>=.008}
 result['all_robot_checks_passed']=all(x['passed'] for x in result['stages']) and minimum>=.008
 result['scope_note']='Same captured report paths retimed continuously to .35/.5; raw401 paired+unequal rigid and FK. This is measured-state planning, not exact previous command-origin replay or a native execution. Future physical observation remains mandatory.'
 save(caseout/'robot-review.json',result);return result
for name,statefile in [('new_initial','orient_teddy_postlift.npz'),('actual_transport_shape','transport_teddy.npz')]:
 caseout=OUT/name;caseout.mkdir(exist_ok=True);state=dict(np.load(RUN/statefile));captures.clear();casecfg=copy.deepcopy(cfg)
 if name=='actual_transport_shape':
  casecfg['task']['contact_trial']['recenter_offsets_xy_m']=[[.015,-.00375]]
  selected=json.loads((RUN/'guarded_contact_candidate.json').read_text())
  choice=json.loads((caseout/'candidate.json').read_text()) if (caseout/'candidate.json').exists() else ci.recenter_guarded_contact_trial(rest,state,casecfg,selected,attempt_index=1,cumulative_xy_m=0,output=caseout/'candidate.json')
 else:choice=json.loads((caseout/'candidate.json').read_text()) if (caseout/'candidate.json').exists() else ci.plan_guarded_contact_trial(rest,state,casecfg,output=caseout/'candidate.json')
 if not captures and choice['status']=='candidate':
  planner=Planner(casecfg);qs={side:state['q_'+side][:7].copy() for side in ['left','right']};finger={side:float(state['q_'+side][7:].mean()) for side in qs};R=planner.fk('right',qs['right'])[1]
  bases={'height_at_current_xy':2.5,'transport_y_at_current_x':3.,'translate_outside_mouth':3.,'insert_candidate':6.,'withdraw_after_release':3.}
  for st in [*choice['route']['stages'],choice['route']['withdraw_stage']]:
   if st['name']=='withdraw_after_release':finger={**finger,'right':.04}
   pp=planner.trajectory(qs,{'right':np.array(st['right_tcp_m'])},rotations={'right':R},samples=45,validate_samples=101,duration=bases[st['name']],finger_start=finger,finger_end=finger,clearance_margin=.008)
   assert abs(pp.duration-st['duration_s'])<1e-10
   qs=pp.end_q()
 save(caseout/'diagnostic-config.json',casecfg);np.savez_compressed(caseout/'actual-source-state.npz',**state)
 print(name,'candidate',choice['status'],choice['report']['rejections'],flush=True)
 result=audit_case(name,state,choice,caseout);report['cases'].append(result);save(OUT/'review.json',report)
report['input_hashes_unchanged']=all(sha(Path(p))==h for p,h in before.items());assert report['input_hashes_unchanged']
report['all_robot_preflight_passed']=all(c.get('all_robot_checks_passed',False) for c in report['cases'])
report['cloth_continuous_review']='Separate held-front descent and both handles pre-rim review required; this report does not certify handles/body or actual insertion.'
save(OUT/'review.json',report);print(json.dumps({'path':str(OUT/'review.json'),'sha256':sha(OUT/'review.json'),'all_robot_preflight_passed':report['all_robot_preflight_passed']},indent=2))
