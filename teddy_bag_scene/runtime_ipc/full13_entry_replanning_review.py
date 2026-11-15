"""CPU replay of the saved full13 command prefix; never initializes Genesis."""
from pathlib import Path
import sys,json,hashlib,re
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parents[1];run=HERE/'runs/two_panel_full_13'
sys.path.insert(0,str(run/'source'))
from motion import Planner,DOWN,minimum_jerk
from contact_grasp import cloth_grasp_frame
config=json.loads((run/'config.json').read_text());events=json.loads((run/'events.json').read_text())
choice=json.loads((run/'guarded_contact_candidate.json').read_text())
entry=json.loads((run/'staged_contact_entry.json').read_text())
planner=Planner(config);initial=np.load(run/'initial.npz');rest=np.load(run/'rest.npz')
qs={s:initial['q_'+s][:7].copy() for s in ('left','right')}
finger={s:float(initial['q_'+s][7:].mean()) for s in qs};rotations={}
asset=np.load(run/'source/assets'/config['bag']['npz_file'])
for side in qs:
 spec=config['task']['handle_grasps'][side]
 frame=cloth_grasp_frame(asset['points'],asset['faces'],asset[spec['group']],shell_radius=config['bag']['thickness'],preferred_approach=spec['preferred_approach'])
 rotations[side]=frame.rotation
left_hold=cloth_grasp_frame(asset['points'],asset['faces'],asset[config['task']['handle_grasps']['left']['group']],shell_radius=config['bag']['thickness'],preferred_approach=config['task']['left_hold_preferred_approach']).rotation
clear=np.load(run/'right_clear_handle.npz');center=np.array(config['task']['bear_grasp_local']);local=rest['bear']-np.array(config['bear']['position'])
ids=np.where(np.linalg.norm(local-center,axis=1)<.032)[0]
target=np.array(config['bear']['position'])+center+(clear['bear'][ids]-rest['bear'][ids]).mean(0)
target[2]=max(target[2],config['table']['height']+.025)
guarded=dict(zip(('adjust_teddy_height_contact_trial','transport_teddy_along_y','transport_teddy'),choice['stages_to_execute']))
prefix=[];plans={};previous='initial'
for event in events:
 label=event['phase']
 if label in ('close_both_handles','drape_under_gravity','settle_oriented_bag','release_right_handle','close_teddy_gripper'):
  if label=='close_both_handles':finger=dict(left=0.,right=0.)
  if label=='release_right_handle':finger['right']=.04
  if label=='close_teddy_gripper':finger['right']=config['task']['bear_finger_target']
  previous=label;continue
 if label=='lower_teddy_to_rim':break
 start=np.load(run/(previous+'.npz'))
 measured={s:float(np.clip(start['q_'+s][7:].mean(),0.,.04)) for s in qs}
 if label=='orient_held_bag':
  plan=planner.trajectory(qs,{'left':config['task']['open_targets']['left']},duration=config['task']['left_hold_rotation_duration_s'],rotations={'left':left_hold},finger_start=measured,finger_end=measured,clearance_margin=.008,max_joint_velocity=.25,max_joint_acceleration=.4)
  rotations['left']=left_hold
 elif label in guarded:
  stage=guarded[label];r=Rotation.from_quat(stage['rotation_xyzw'])
  plan=planner.trajectory(qs,{'right':stage['right_tcp_m']},duration=stage['duration_s'],rotations={'right':r},finger_start=measured,finger_end=measured,clearance_margin=.008,max_joint_velocity=.35,max_joint_acceleration=.5)
  rotations['right']=r
 else:
  if label=='spread_bag':targets=config['task']['open_targets'];duration=2.5
  elif label=='right_clear_handle':targets={'right':[.12,.20,1.32]};duration=2.;rotations['right']=DOWN
  elif label=='teddy_pregrasp':targets={'right':target+[0,0,.15]};duration=3.;rotations['right']=DOWN
  elif label=='teddy_grasp_approach':targets={'right':target};duration=2.;rotations['right']=DOWN
  elif label=='lift_teddy':targets={'right':[target[0],target[1],1.37]};duration=4.
  else:raise ValueError(label)
  speed=config['task']['motion_speed_scale']
  plan=planner.trajectory(qs,targets,duration=duration/speed,rotations={s:rotations[s] for s in targets},finger_start=finger,finger_end=finger,clearance_margin=.008,max_joint_velocity=.55*speed,max_joint_acceleration=speed**2)
 prefix.append(dict(phase=label,reconstructed_duration_s=plan.duration,recorded_duration_s=event['duration'],difference_s=plan.duration-event['duration']))
 assert abs(plan.duration-event['duration'])<1e-8,(label,prefix[-1])
 qs=plan.end_q();plans[label]=plan;previous=label
state=np.load(run/'transport_teddy.npz');stage=entry['stage'];target=np.array(stage['right_tcp_m']);r=Rotation.from_quat(stage['rotation_xyzw'])
measured={s:float(np.clip(state['q_'+s][7:].mean(),0.,.04)) for s in qs}
runtime=planner.trajectory(qs,{'right':target},duration=stage['duration_s'],rotations={'right':r},finger_start=measured,finger_end=measured,clearance_margin=.008,max_joint_velocity=.35,max_joint_acceleration=.5)
actual_qs={s:state['q_'+s][:7].copy() for s in qs}
preflight=planner.trajectory(actual_qs,{'right':target},duration=6.,rotations={'right':r},finger_start=measured,finger_end=measured,clearance_margin=.008,max_joint_velocity=.35,max_joint_acceleration=.5)
assert abs(preflight.duration-stage['duration_s'])<1e-8
start=np.array(state['tcp_right']);records=[]
for f in np.linspace(0.,1.,201):
 a=runtime.sample(float(f*runtime.duration));b=preflight.sample(float(f*preflight.duration))
 ap,ar=planner.fk('right',a['q']['right']);bp,br=planner.fk('right',b['q']['right'])
 line=start+float(minimum_jerk(f)[0])*(target-start)
 records.append(dict(fraction=float(f),runtime_tcp_m=ap.tolist(),preflight_tcp_m=bp.tolist(),
  runtime_error_from_actual_start_line_m=float(np.linalg.norm(ap-line)),
  runtime_preflight_tcp_difference_m=float(np.linalg.norm(ap-bp)),
  runtime_orientation_error_rad=float((ar*r.inv()).magnitude()),
  runtime_preflight_joint_max_difference_rad=float(np.max(abs(a['q']['right']-b['q']['right']))),
  maximum_joint_speed_rad_s=float(max(abs(a['qd'][s]).max() for s in qs)),
  maximum_joint_acceleration_rad_s2=float(max(abs(a['qdd'][s]).max() for s in qs)),
  minimum_rigid_clearance_m=float(min(v['distance'] for v in planner.clearance(a['q'],measured).values()))))
log=(HERE/'runs/two_panel_full_13.log').read_text()
recorded=float(re.findall(r'PHASE lower_teddy_to_rim duration=([0-9.]+)s',log)[-1])
checks={'full_command_prefix_duration_matches':True,'runtime_duration_matches_log_to_print_precision':abs(runtime.duration-recorded)<=.000501,
 'preflight_duration_reproduced':True,'same_target_and_complete_rotation':True,
 'runtime_sampled_straight_line_error_within_half_mm':max(v['runtime_error_from_actual_start_line_m'] for v in records)<=.0005,
 'runtime_sampled_rotation_error_below_2mrad':max(v['runtime_orientation_error_rad'] for v in records)<=.002,
 'runtime_sampled_rigid_clearance_above_8mm':min(v['minimum_rigid_clearance_m'] for v in records)>=.008}
result={'kind':'independent_actual_prefix_entry_replanning_review','passed':all(checks.values()),'checks':checks,'source_sha256':{n:hashlib.sha256((run/'source'/n).read_bytes()).hexdigest() for n in ('run.py','motion.py','staged_contact_insertion.py')},
 'prefix_reconstruction':prefix,'runtime_duration_s':runtime.duration,'preflight_duration_s':preflight.duration,'runtime_recorded_rounded_duration_s':recorded,
 'source_command_minus_measured_q_rad':{s:(qs[s]-actual_qs[s]).tolist() for s in qs},
 'summary':{key:(min(v[key] for v in records) if key=='minimum_rigid_clearance_m' else max(v[key] for v in records)) for key in records[0] if key not in ('fraction','runtime_tcp_m','preflight_tcp_m')},
 'limitations':['This reconstructs issued joint targets, not actual dynamics. Every-step contact/soft/grasp monitors remain authoritative.',
  'The runtime IK is recomputed from previous commanded q; preflight IK starts from measured q. Thus this is not literally only time scaling of the identical joint spline.',
  'Cartesian/rotation/rigid errors are sampled at 201 path fractions, not an independent continuous FK bound.',
  'A slower path does not guarantee reduced quasistatic cloth contact loads or successful passage.'], 'samples':records}
(run/'independent-entry-replanning-review.json').write_text(json.dumps(result,indent=2,default=lambda value:value.item())+'\n')
print(json.dumps({k:result[k] for k in ('passed','checks','runtime_duration_s','preflight_duration_s','source_command_minus_measured_q_rad','summary')},indent=2,default=lambda value:value.item()))
