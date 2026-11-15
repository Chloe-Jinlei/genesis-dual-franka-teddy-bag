"""One explicitly bounded full19 actual-entry CPU-only yaw contingency. No runtime edits or GPU."""
from pathlib import Path
import hashlib, json, tempfile, time, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_19'
ORIGINAL=ROOT/'runtime_ipc/full17_yaw_contingency.py'
source=ORIGINAL.read_text().replace('runs/two_panel_full_17', 'runs/two_panel_full_19');prefix=source.split('\nresults = []')[0]
prefix=prefix.replace('        category, value = min(checks.items(), key=lambda item: item[1]["distance"])', '        raw_distances=np.asarray([item["distance"] for item in checks.values()]);\n        if not raw_distances.size or not np.isfinite(raw_distances).all(): raise ValueError("Non-finite/empty raw rigid distances")\n        category, value = min(checks.items(), key=lambda item: item[1]["distance"])')
ns={'__file__':str(ORIGINAL),'__name__':'independent_small_yaw_helper_snapshot'}
exec(compile(prefix,str(ORIGINAL),'exec'),ns)
Rotation=ns['Rotation'];state=ns['state'];cfg=ns['cfg'];rest=ns['rest'];planner=ns['planner']
original=ns['original'];tcp=ns['tcp0'];fingers=ns['fingers'];qs=ns['initial_q'];rim=ns['rim']
sys.path.insert(0,str(ROOT/'runtime_ipc'))
import diagnose_low_region_winding as low
import body_alignment_ranking as ranking

output_dir=RUN/'independent-single-low-yaw-contingency';output_dir.mkdir(exist_ok=True)
offsets=[[-.02,-.06]]
inputs=[ORIGINAL,RUN/'config.json',RUN/'rest.npz',RUN/'lower_teddy_to_rim.npz',RUN/'guarded_contact_candidate.json',
 RUN/'transport_teddy.npz',RUN/'source/motion.py',RUN/'source/contact_robot_geometry.py',RUN/'source/validate.py',
 ROOT/'runtime_ipc/diagnose_low_region_winding.py',ROOT/'runtime_ipc/body_alignment_ranking.py',Path(__file__).resolve()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
state_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in state.items()}
actual_tcp,actual_rotation=planner.fk('right',qs['right'])
transport=dict(np.load(RUN/'transport_teddy.npz'));cumulative_before=float(np.linalg.norm(tcp[:2]-transport['tcp_right'][:2]))
budget=cfg['task']['contact_trial']['staged']
report=dict(kind='full19_single_actual_entry_world_z_yaw_contingency',physical_success=False,runtime_execution_performed=False,
 active_runtime_or_configuration_modified=False,actual_entry_time_s=float(state['time']),fixed_tcp_world_m=tcp.tolist(),
 actual_full_rotation_xyzw=actual_rotation.as_quat().tolist(),baseline_commanded_full_rotation_xyzw=original.as_quat().tolist(),
 actual_vs_command_rotation_error_rad=float((original*actual_rotation.inv()).magnitude()),
 tcp_measurement_vs_fk_error_m=float(np.linalg.norm(tcp-actual_tcp)),
 input_sha256=input_hashes,declared_yaw_degrees=[15.],declared_center_offsets_xy_m=offsets,results=[],
 minimum_rigid_margin_m=.008,minimum_frozen_teddy_left_robot_sweep_margin_m=.005,
 stored_trajectory_scope='Exact float64 CubicSpline knots/coefficients and minimum-jerk time law for hypothetical command plans, not executed robot trajectories.',
 finger_geometry=dict(individual_left_fingers_m=state['q_left'][7:].tolist(),individual_right_fingers_m=state['q_right'][7:].tolist(),
  rigid_path_half_opening_approximations_m=fingers,frozen_bear_left_sweep_uses_actual_individual_fingers=True),
 limitations=['Pure intended yaw uses WORLD-Z left multiplication of the original full commanded rotation, preserving its complete orientation relation.',
  'Existing staged runtime requires original wrist orientation; these nonzero yaws are future offline contingencies, not currently executable runtime-policy records.',
  'Frozen shape arcs/translations cover full tetrahedral solids against measured nonbase left collision solids. They do not predict grasp retention, deformation or native affine tracking.',
  'Rigid clearance and Cartesian/Slerp tracking are sampled at 401 points; joint position extrema are analytic, retimed rates sampled at 501 points.',
  'This uses the full19 actual entry state only; no cross-run shape reproducibility is assumed. All later real stages must use fresh actual observations and unchanged native/final physical gates.'])
out=RUN/'independent-single-low-yaw-contingency.json'
def serial(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v))
def save():out.write_text(json.dumps(report,indent=2,allow_nan=False,default=serial)+'\n')
def store_plan(plan,label):
 path=output_dir/(label+'.npz');arrays={'duration_s':np.array(plan.duration),
  'finger_start_left_right_m':np.array([plan.finger_start[s] for s in ['left','right']]),
  'finger_end_left_right_m':np.array([plan.finger_end[s] for s in ['left','right']])}
 for side in ['left','right']:
  arrays[side+'_spline_knots_fraction']=plan.curves[side].x.copy();arrays[side+'_spline_coefficients']=plan.curves[side].c.copy()
 np.savez_compressed(path,**arrays)
 return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),coefficient_order='c[0:4] multiply local fraction powers 3,2,1,0; axes coefficient,segment,joint',
  time_law='fraction = 10u^3 - 15u^4 + 6u^5, u=clip(t/duration_s,0,1)',
  exactly_reconstructs_offline_joint_command=True,does_not_reconstruct_actual_physics=True)
def path(start,target,rotation,duration,label):
 evidence,plan=ns['checked_path'](start,np.asarray(target),rotation,fingers,duration)
 evidence['trajectory_archive']=store_plan(plan,label)
 return evidence,plan

start_time=time.monotonic();save()
for degrees in [15.]:
 angle=np.deg2rad(degrees);yaw=Rotation.from_euler('z',angle);target_rotation=yaw*original
 row=dict(world_z_yaw_degrees=degrees,physical_success=False,runtime_execution_performed=False,center_remainders=[])
 report['results'].append(row)
 print(json.dumps(dict(event='rotation_begin',degrees=degrees)),flush=True)
 try:
  row['rotation'],rp=path(qs,tcp,target_rotation,6.,f'yaw{int(degrees):02d}_rotation')
  frozen_bear=ns['yawed_bear'](angle)
  exact_endpoint=target_rotation.apply(actual_rotation.inv().apply(state['bear']-actual_tcp))+tcp
  row['pure_yaw_vs_actual_rigid_grasp_endpoint_max_vertex_difference_m']=float(np.linalg.norm(exact_endpoint-frozen_bear,axis=1).max())
  row['existing_no_yaw_runtime_orientation_gate_would_accept']=bool((target_rotation*original.inv()).magnitude()<=.015)
  if row['rotation']['status']!='geometric_candidate':save();continue
  row['frozen_bear_rotation_sweep']=ns['rotation_sweep'](angle)
  if row['frozen_bear_rotation_sweep']['status']!='clear':save();continue
  for index,delta_xy in enumerate(offsets):
   delta=np.r_[delta_xy,0.];tag=f'yaw{int(degrees):02d}_candidate{index}'
   item=dict(index=index,center_offset_xy_m=delta_xy,all_frozen_geometric_gates_pass=False,
    individual_xy_m=float(np.linalg.norm(delta_xy)),cumulative_xy_before_m=cumulative_before,
    cumulative_xy_after_m=cumulative_before+float(np.linalg.norm(delta_xy)),
    maximum_individual_xy_m=budget['maximum_individual_xy_m'],maximum_cumulative_xy_m=budget['maximum_cumulative_xy_m'],
    full_orientation_world_left_multiply=True,target_rotation_xyzw=target_rotation.as_quat().tolist())
   item['xy_budget_pass']=bool(item['individual_xy_m']<=item['maximum_individual_xy_m'] and item['cumulative_xy_after_m']<=item['maximum_cumulative_xy_m'])
   row['center_remainders'].append(item)
   print(json.dumps(dict(event='candidate_begin',degrees=degrees,index=index,offset=delta_xy)),flush=True)
   try:
    if not item['xy_budget_pass']:raise ValueError('Declared XY budget exceeded')
    item['com_contact_envelope']=ns['envelope'](frozen_bear,delta)
    target=tcp+delta
    item['center'],cp=path(rp.end_q(),target,target_rotation,4.,tag+'_center')
    if item['center']['status']!='geometric_candidate':raise ValueError('Center rigid or tracking gate rejected')
    item['center_bear_left_sweep']=ns['sweep'](frozen_bear,np.zeros(3),delta)
    if item['center_bear_left_sweep']['status']!='clear':raise ValueError('Center frozen solid sweep rejected')
    dz=cfg['table']['height']+.01-frozen_bear[:,2].min()
    item['insert'],ip=path(cp.end_q(),target+np.array([0.,0.,dz]),target_rotation,6.,tag+'_insert')
    if item['insert']['status']!='geometric_candidate':raise ValueError('Insert rigid or tracking gate rejected')
    item['insert_bear_left_sweep']=ns['sweep'](frozen_bear,delta,np.array([0.,0.,dz]))
    if item['insert_bear_left_sweep']['status']!='clear':raise ValueError('Insert frozen solid sweep rejected')
    final_bear=frozen_bear+delta+np.array([0.,0.,dz])
    item['whole_frozen_bear_below_lowest_rim_by_2mm']=bool(final_bear[:,2].max()<=state['bag'][rim,2].min()-.002)
    opening=[]
    for fraction in np.linspace(0.,1.,21):
     ff={**fingers,'right':float((1-fraction)*fingers['right']+fraction*.04)}
     checks=planner.clearance(ip.end_q(),ff);raw_distances=np.asarray([v['distance'] for v in checks.values()])
     if not raw_distances.size or not np.isfinite(raw_distances).all(): raise ValueError('Non-finite/empty release distances')
     category,value=min(checks.items(),key=lambda e:e[1]['distance'])
     opening.append(dict(fraction=float(fraction),category=category,**value))
    minimum=min(opening,key=lambda e:e['distance'])
    item['release']=dict(status='geometric_candidate' if np.isfinite(minimum['distance']) and minimum['distance']>=.008 else 'rejected',
     minimum_sampled_rigid_clearance_m=minimum['distance'],worst_sample=minimum,sample_count=21,
     samples=opening,fixed_joint_pose_left_right={k:v.tolist() for k,v in ip.end_q().items()})
    opened={**fingers,'right':.04}
    evidence,wp=ns['checked_path'](ip.end_q(),target,target_rotation,opened,4.)
    evidence['trajectory_archive']=store_plan(wp,tag+'_withdraw');item['withdraw']=evidence
    item['all_frozen_geometric_gates_pass']=bool(item['com_contact_envelope']['at_least_2mm_inside'] and
     item['whole_frozen_bear_below_lowest_rim_by_2mm'] and item['release']['status']=='geometric_candidate' and item['withdraw']['status']=='geometric_candidate')
    item['minimum_complete_route_sampled_rigid_clearance_m']=min(row['rotation']['minimum_sampled_rigid_clearance_m'],*[item[k]['minimum_sampled_rigid_clearance_m']for k in ['center','insert','release','withdraw']])
    if item['all_frozen_geometric_gates_pass']:
     with tempfile.TemporaryDirectory(prefix='small-yaw-frozen-query-') as tmp:
      query=Path(tmp)/'explicit_frozen_yaw_final_hypothesis.npz';np.savez(query,time=state['time'],bag=state['bag'],bear=final_bear)
      diagnostic=low.diagnose(RUN,query)
     item['body_metrics']=ranking.metrics_from_diagnostic(diagnostic,len(final_bear))
     item['frozen_body_diagnostic']={k:diagnostic.get(k) for k in ['height_precondition','actual_topology_orientation','actual_spatial_rim','cap_winding','physical_cloth_query_distance','reasons']}
     item['body_metric_scope']='Frozen hypothetical final points only, no physical rotation/retention/passage or whole-tetrahedron containment claim'
   except (ValueError,KeyError,TypeError,RuntimeError) as exc:
    item['rejection_reason']=str(exc)
   save();print(json.dumps(dict(event='candidate_complete',degrees=degrees,index=index,passed=item['all_frozen_geometric_gates_pass'],reason=item.get('rejection_reason'),minimum_rigid=item.get('minimum_complete_route_sampled_rigid_clearance_m'),body_metrics=item.get('body_metrics'))),flush=True)
 except (ValueError,KeyError,TypeError,RuntimeError) as exc:row['rotation_rejection_reason']=str(exc)
 save()
report['elapsed_wall_s']=time.monotonic()-start_time
report['all_input_and_source_hashes_unchanged']=input_hashes=={str(p.relative_to(ROOT)):sha(p)for p in inputs}
report['actual_entry_arrays_unchanged']=state_hashes=={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in state.items()}
report['finite_candidate_count']=sum(len(r['center_remainders'])for r in report['results'])
report['full_frozen_route_pass_count']=sum(i['all_frozen_geometric_gates_pass']for r in report['results']for i in r['center_remainders'])
save();print(json.dumps({k:report[k]for k in ['elapsed_wall_s','all_input_and_source_hashes_unchanged','actual_entry_arrays_unchanged','finite_candidate_count','full_frozen_route_pass_count']}),flush=True);print(out,flush=True)
