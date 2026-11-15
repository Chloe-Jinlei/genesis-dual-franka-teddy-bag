"""One no-yaw frozen candidate only; no native execution or production edits."""
from pathlib import Path
import copy, hashlib, json, sys, time
import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation, Slerp
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'runs/two_panel_full_25'; OUT=RUN/'independent-no-yaw-positive-y30p75-review'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
from contact_insertion import plan_guarded_contact_trial,reobserve_guarded_contact_trial,_serial
from motion import Planner,minimum_jerk
from postlift_yaw import _clearance_rows,_finite_tree
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=[RUN/n for n in ['config.json','rest.npz','lift_teddy.npz','postlift_yaw_plan.npz','postlift_yaw_candidate.json','input_sha256.json']]+sorted((RUN/'source').glob('*.py'))+sorted((RUN/'source/assets').iterdir())
before={str(p):sha(p) for p in inputs if p.is_file()}; production={str(p):sha(p) for p in [*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'config_gravity_seating_trial.json']}
cfg=json.loads((RUN/'config.json').read_text()); config=copy.deepcopy(cfg)
del config['task']['contact_trial']['postlift_yaw']; config['task']['contact_trial']['center_offsets_xy_m']=[[0.,.03075]]
config['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
for k in ['npz_file','metadata']: config['bag'][k]=str(RUN/'source/assets'/Path(config['bag'][k]).name)
(OUT/'diagnostic_config.json').write_text(json.dumps(config,indent=2)+'\n')
rest=dict(np.load(RUN/'rest.npz',allow_pickle=False)); actual=dict(np.load(RUN/'lift_teddy.npz',allow_pickle=False)); archive=dict(np.load(RUN/'postlift_yaw_plan.npz',allow_pickle=False)); saved=json.loads((RUN/'postlift_yaw_candidate.json').read_text())
commands={s:PPoly(archive[s+'_c'],archive[s+'_x'])(0.) for s in ['left','right']}
assert abs(float(actual['time'])-saved['observed_time_s'])<1e-9
assert all(np.array_equal(commands[s],saved['command_source_poses'][s]['q']) for s in commands)
np.savez_compressed(OUT/'actual_lift_source.npz',**actual); np.savez_compressed(OUT/'rest.npz',**rest)
report=dict(kind='one_no_yaw_actual_full25_lift_frozen_route',physical_success=False,native_execution_performed=False,status='preparation_pending',source_time_s=float(actual['time']),input_sha256=before,production_sha256=production,semantic_config_changes={'postlift_yaw':'removed, no helper call','center_offsets_xy_m':[[0.,.03075]]},frozen_resource_rebindings=['robots.mjcf_path','bag.npz_file','bag.metadata'],command_source='Archived full25 native postlift curve first knot, cross-checked to candidate command_source_poses; this equals previous lift commanded endpoint, not actual q.',command_qs=commands,actual_lift_extent_m=np.ptp(actual['bear'],axis=0),proxy_stages=[],limitations=['Only actual_lift_source is native. Frozen rigid grasp keeps soft shape and bag unchanged; no force/slip/cloth prediction.','Original preparation and fresh full-depth checks are retained. Gravity shortened endpoint and actual release eligibility are not evaluated.','Command proxy uses frozen measured finger openings at every stage; real native would remeasure them. Robot clearance and FK are sampled, reference joint/rate extrema continuous numerical.','No handle crossing or dynamic final containment is certified; next independent handle check is required.'])
def save():
 report['inputs_unchanged']=all(sha(p)==h for p,h in before.items());report['production_unchanged']=all(sha(p)==h for p,h in production.items());(OUT/'review.json').write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')
save();started=time.monotonic(); choice=plan_guarded_contact_trial(rest,actual,config,output=OUT/'preparation.json');report['preparation_status']=choice['status']; report['preparation_rejections']=choice['report'].get('rejections',[]);report['preparation_cpu_s']=time.monotonic()-started
report['robot_attempts']=[{k:r.get(k) for k in ['rejection','stages_completed']} for r in choice['report'].get('robot_centering_attempts',[])]
report['status']='preparation_candidate' if choice['status']=='candidate' else 'rejected_preparation';save();print('PREP_RESULT '+json.dumps({'status':choice['status'],'rejections':report['preparation_rejections'],'robot_attempts':report['robot_attempts'],'path':str(OUT/'preparation.json')}),flush=True)
if choice['status']!='candidate':raise SystemExit(0)
planner=Planner(config);p0,R0=planner.fk('right',actual['q_right'][:7]); fingers={s:float(np.clip(actual['q_'+s][7:].mean(),0.,.04)) for s in commands}; qs={s:q.copy() for s,q in commands.items()};dt=float(cfg['simulation']['dt']);steps=0
try:
 for i,stage in enumerate(choice['stages_to_execute']):
  start,Rs=planner.fk('right',qs['right']);target=np.asarray(stage['right_tcp_m']);Rt=Rotation.from_quat(stage['rotation_xyzw']);sl=Slerp([0,1],Rotation.from_quat([Rs.as_quat(),Rt.as_quat()]))
  plan=planner.trajectory(qs,{'right':target},duration=stage['duration_s'],rotations={'right':Rt},finger_start=fingers,finger_end=fingers,clearance_margin=.008,validate_samples=201,max_joint_velocity=.35,max_joint_acceleration=.5)
  rates=plan.continuous_rate_extrema(); assert _finite_tree(rates)
  vmax=max(float(np.max(r['maximum_velocity_rad_s'])) for r in rates.values());amax=max(float(np.max(r['maximum_acceleration_rad_s2'])) for r in rates.values());assert vmax<=.35 and amax<=.5
  low=float('inf');worst=None;pmax=rmax=0.
  for t in np.linspace(0.,plan.duration,401):
   sample=plan.sample(float(t));f=float(minimum_jerk(t/plan.duration)[0])
   for kind in ['paired_mean','actual_individual']:
    if kind=='actual_individual':
     for side in qs:
      for j,v in enumerate(actual['q_'+side][7:],1):planner.d.qpos[planner.m.joint(f'{side}_finger_joint{j}').qposadr[0]]=v
    raw=planner.clearance(sample['q'],sample['finger'] if kind=='paired_mean' else None);cat,row=_clearance_rows(raw,stage['name']+' '+kind)
    if row['distance']<low:low=row['distance'];worst=dict(time_s=float(t),finger_mode=kind,category=cat,**row)
   p,R=planner.fk('right',sample['q']['right']);pe=float(np.linalg.norm(p-((1-f)*start+f*target)));re=float((R*sl(f).inv()).magnitude());assert np.isfinite([pe,re]).all();pmax=max(pmax,pe);rmax=max(rmax,re)
  assert pmax<=.0005 and rmax<=.002
  n=int(np.ceil(plan.duration/dt));steps+=n
  np.savez_compressed(OUT/f'proxy_stage_{i}.npz',duration=plan.duration,**{s+'_x':plan.curves[s].x for s in qs},**{s+'_c':plan.curves[s].c for s in qs})
  report['proxy_stages'].append(dict(name=stage['name'],duration_s=plan.duration,integer_steps=n,start_command_q=qs,first_knot_delta_from_command={s:plan.curves[s](0)-qs[s] for s in qs},rates=rates,minimum_rigid_clearance_m=low,worst=worst,max_FK_position_error_m=pmax,max_FK_Slerp_error_rad=rmax,raw_samples=401,finger_modes=['paired_mean','actual_individual']))
  qs=plan.end_q();save()
except Exception as exc:
 report.update(status='rejected_command_proxy',exception_type=type(exc).__name__,rejection=str(exc));save();print('PROXY_REJECTED '+str(exc),flush=True);raise SystemExit(0)
hyp={k:np.array(actual[k],copy=True) for k in ['bag','bear','q_left','q_right','tcp_left','tcp_right','time']}
for s in qs:hyp['q_'+s][:7]=qs[s];hyp['tcp_'+s]=planner.fk(s,qs[s])[0]
pend,Rend=planner.fk('right',qs['right']);hyp['bear']=(Rend*R0.inv()).apply(actual['bear']-actual['tcp_right'])+pend;hyp['time']=np.array(float(actual['time'])+steps*dt)
np.savez_compressed(OUT/'hypothetical_transport_endpoint.npz',**hyp)
report['hypothetical_construction']=dict(formula='(endFK_R * inverse(actualLiftFK_R)).apply(actualBear - actualLiftTCP) + endFK_TCP',integer_preparation_steps=steps,clock_increment_s=steps*dt,time_s=float(hyp['time']),bag_unchanged=np.array_equal(hyp['bag'],actual['bag']),fingers_unchanged=all(np.array_equal(hyp['q_'+s][7:],actual['q_'+s][7:]) for s in qs),full_rotation_error_from_actual_source_rad=float((Rend*R0.inv()).magnitude()),no_force_arrays=True)
report['status']='hypothetical_fresh_pending';save();fresh=reobserve_guarded_contact_trial(rest,hyp,config,choice,output=OUT/'hypothetical_fresh_full_depth.json')
report.update(status='candidate_frozen_only' if fresh['status']=='candidate' else 'rejected_hypothetical_fresh',hypothetical_fresh_status=fresh['status'],hypothetical_fresh_rejections=fresh['report'].get('rejections',[]),fresh_robot_attempts=[{k:r.get(k) for k in ['rejection','stages_completed']} for r in fresh['report'].get('robot_centering_attempts',[])])
save();print('FINAL '+json.dumps({'status':report['status'],'rejections':report['hypothetical_fresh_rejections'],'path':str(OUT/'review.json'),'sha256':sha(OUT/'review.json')}),flush=True)
