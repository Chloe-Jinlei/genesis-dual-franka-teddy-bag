"""CPU-only review of the completed entry, without assigning unknown contact pairs."""
from pathlib import Path
import json,sys,hashlib,argparse
import numpy as np
HERE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('run',nargs='?',type=Path,default=HERE/'runs/two_panel_full_13')
RUN=parser.parse_args().run.resolve()
sys.path.insert(0,str(RUN/'source'))
from motion import Planner
from validate import signed_volumes,mouth_structure
config=json.loads((RUN/'config.json').read_text());report=json.loads((RUN/'report.json').read_text())
validation=json.loads((RUN/'validation.json').read_text());guards=json.loads((RUN/'contact_guard_trace.json').read_text())
rest=np.load(RUN/'rest.npz');states=np.load(RUN/'states.npz');metrics=np.load(RUN/'soft_metrics.npy')
event=next(e for e in report['events'] if e['phase']=='lower_teddy_to_rim');start=float(event['start']);end=float(report['simulated_seconds'])
time=np.array([r['time_s'] for r in guards]);mask=time>=start-1e-9;rows=[r for r,yes in zip(guards,mask) if yes]
def array(key):return np.array([r[key] for r in rows],float)
def vector_summary(key):
 a=array(key);norm=np.linalg.norm(a,axis=-1);i=int(np.argmax(norm))
 return dict(first=a[0].tolist(),last=a[-1].tolist(),maximum_norm=float(norm[i]),maximum_time_s=float(rows[i]['time_s']),at_maximum=a[i].tolist())
raw={key:vector_summary(key) for key in ('force_world_n','torque_at_aim_tcp_world_nm','left_all_link_net_force_world_n','left_all_link_torque_at_aim_tcp_world_nm')}
perlink={}
for side in ('left','right'):
 force=array(side+'_link_force_world_n');torque=array(side+'_link_torque_at_'+('left_' if side=='left' else '')+'aim_tcp_world_nm')
 perlink[side]=[]
 for i,name in enumerate(rows[0][side+'_link_names']):
  norms=np.linalg.norm(force[:,i],axis=1);k=int(np.argmax(norms));tnorm=np.linalg.norm(torque[:,i],axis=1);j=int(np.argmax(tnorm))
  perlink[side].append(dict(link=name,first_force_world_n=force[0,i].tolist(),last_force_world_n=force[-1,i].tolist(),
   maximum_force_norm_n=float(norms[k]),maximum_force_time_s=float(rows[k]['time_s']),force_at_peak_world_n=force[k,i].tolist(),
   first_torque_at_tcp_world_nm=torque[0,i].tolist(),last_torque_at_tcp_world_nm=torque[-1,i].tolist(),
   maximum_torque_norm_nm=float(tnorm[j]),maximum_torque_time_s=float(rows[j]['time_s'])))
selected=metrics[metrics[:,0]>=start-1e-9];j=int(np.argmin(selected[:,1]));k=int(np.argmax(selected[:,4]))
physics=dict(rows_including_start=len(selected),minimum_tetra_volume_ratio=float(selected[j,1]),minimum_J_time_s=float(selected[j,0]),
 max_bag_edge_stretch=float(selected[k,4]),max_stretch_time_s=float(selected[k,0]),
 maximum_table_penetration_m={'bag':float(selected[:,2].max()),'bear':float(selected[:,3].max())},all_states_finite=bool(np.all(selected[:,5]==1)))
holding=dict(left_material_max_distance_m=float(array('left_designated_material_distance_m').max()),
 right_material_max_distance_m=float(array('right_nearest_teddy_distance_m').max()),
 right_two_sided_required_every_row=all(r['right_two_sided_contact_required'] for r in rows),
 left_minimum_opening_reactions_n=array('left_finger_opening_projection_n').min(0).tolist(),
 right_minimum_opening_reactions_n=array('right_finger_opening_projection_n').min(0).tolist(),
 right_maximum_finger_force_norm_n=array('right_finger_force_norm_n').max(0).tolist(),
 no_raw_guard_stop_reason=all(r['stop_reason'] is None for r in rows))
tets=rest['bear_tets'];weights=np.abs(signed_volumes(rest['bear'],tets))*config['bear']['rho'];mass=float(weights.sum())
body,rim=mouth_structure(rest['bag'],rest['bag_faces'],config)[:2];planner=Planner(config)
stages=[]
for name in ('lift_teddy','transport_teddy','lower_teddy_to_rim'):
 s=np.load(RUN/(name+'.npz'));ts=float(s['time']);g=min(guards,key=lambda r:abs(r['time_s']-ts))
 com=np.einsum('n,ni->i',weights,s['bear'][tets].mean(1))/mass;rimz=s['bag'][rim,2];tcp=s['tcp_right']
 gravity=np.array([0.,0.,-9.81*mass]);moment=np.cross(com-np.array(g['aim_tcp_world_m']),gravity)
 stages.append(dict(phase=name,time_s=ts,bear_bounds_world_m=[s['bear'].min(0).tolist(),s['bear'].max(0).tolist()],
  bear_center_of_mass_world_m=com.tolist(),tcp_world_m=tcp.tolist(),bear_com_relative_tcp_m=(com-tcp).tolist(),
  right_finger_q_m=s['q_right'][-2:].tolist(),right_finger_pd_command_force_n=s['force_right'][-2:].tolist(),
  body_rim_z_range_m=[float(rimz.min()),float(rimz.max())],
  vertex_fraction_below_highest_rim=float(np.mean(s['bear'][:,2]<rimz.max())),
  vertex_fraction_below_lowest_rim=float(np.mean(s['bear'][:,2]<rimz.min())),
  gravity_moment_about_aim_tcp_nm=moment.tolist(),right_force_n=g['force_world_n'],right_moment_nm=g['torque_at_aim_tcp_world_nm'],
  right_moment_minus_static_gravity_nm=(np.array(g['torque_at_aim_tcp_world_nm'])-moment).tolist()))
# The staged monitor was not activated: center was rejected before its call.
# Bound the unrecorded palm affine height using its saved pose-error summaries.
last=np.load(RUN/'lower_teddy_to_rim.npz');diag=json.loads((RUN/'latest_contact_diagnostics.json').read_text());ref=np.load(RUN/'staged_native_hand_reference.npz')
planner.fk('left',last['q_left'][:7]);idx=planner.m.body('left_link7').id
R=planner.d.xmat[idx].reshape(3,3);p=planner.d.xpos[idx];local=ref['left_link7'];rigid_min=float((local@R[2]+p[2]).min())
d=next(r for r in diag['links'] if r['entity']=='left_franka' and r['link']=='link7')
error=float(d['translation_error_m']+(d['affine_singular_value_deviation']+2*np.sin(d['rotation_error_rad']/2))*np.linalg.norm(local,axis=1).max())
finger_minima={}
for name in ('left_finger','right_finger'):
 d=next(r for r in diag['links'] if r['entity']=='left_franka' and r['link']==name);T=np.array(d['ipc_transform'])
 finger_minima[name]=float((ref['left_'+name]@T[2,:3]+T[2,3]).min())
top=float(last['bear'][:,2].max());hand_lower=min(rigid_min-error,*finger_minima.values())
height=dict(time_s=end,complete_bear_maximum_z_m=top,true_palm_measured_q_minimum_z_m=rigid_min,
 conservative_palm_world_vertex_error_bound_m=error,native_palm_minimum_z_lower_bound_m=rigid_min-error,
 exact_saved_native_finger_minimum_z_m=finger_minima,complete_bear_to_native_hand_gap_lower_bound_m=hand_lower-top,
 endpoint_gap_above_5mm=bool(hand_lower-top>=.005),native_per_step_hand_monitor_rows=len(json.loads((RUN/'staged_hand_trace.json').read_text())),
 method='Palm minZ from verified native palm-reference vertices and measured q; subtract delta_p+(max|singular_value-1|+2sin(polar_angle_error/2))*max||local_vertex||. Fingers use saved full native transforms.',
 scope='Endpoint-only independent conservative reconstruction; no every-step hand-height claim because center never began.')
out=dict(kind='independent_completed_entry_physics_review',run=RUN.name,task_success=False,completed_downward_entry=True,
 no_center_insert_release_executed=not any(e['phase'] in ('center_teddy_at_rim','insert_teddy','release_teddy') for e in report['events']),
 actual_stop=report['error'],entry_start_s=start,entry_end_s=end,entry_guard_rows_including_start=len(rows),
 entry_physics=physics,entry_holding=holding,entry_raw_wrenches=raw,entry_per_link_wrenches=perlink,
 mass_kg=mass,stage_snapshots=stages,endpoint_native_hand_clearance=height,
 final_geometry_interpretation=dict(virtual_closed_body_valid=validation['final_containment']['virtual_closed_body_valid'],
  actual_complete_containment_proven=False,inside_fraction='undefined for this invalid virtual closure; validator zero is a fail-closed sentinel, not measured zero occupancy',
  final_contact_sampling=validation['final_contact']),
 limitations=['IPC STC wrenches include soft contacts and proxy dynamics, not pair-identified contact forces or Genesis rigid-contact force totals.',
  'A nonzero palm reaction alone does not identify teddy versus cloth. A small reaction cannot prove zero contact.',
  'Soft metrics/force guards are checked at each saved 5 ms integration state; full geometry and per-element principal stretch only at saved frames.',
  'The final bag-bear penetration check samples cloth vertices/edge midpoints/face centroids; it is not an independent continuous triangle-triangle guarantee.',
  'Material stiffness, friction and engineering thresholds remain uncalibrated. No successful insertion or release occurred.'])
(RUN/'independent-entry-physics-review.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:out[k] for k in ('actual_stop','entry_start_s','entry_end_s','entry_guard_rows_including_start','entry_physics','entry_holding','entry_raw_wrenches','endpoint_native_hand_clearance','stage_snapshots')},indent=2))
