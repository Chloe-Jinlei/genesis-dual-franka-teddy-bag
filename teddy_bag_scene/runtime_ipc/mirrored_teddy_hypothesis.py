"""Nine-point CPU geometry study, explicitly hypothetical and not a physical grasp result."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from contact_insertion import _mouth,_policy,_section_envelopes,_evaluate_offset
from contact_robot_geometry import check_translation_sweep
from insertion_candidates import projected_inside,polygon_centroid
from validate import signed_volumes
from motion import Planner

RUN=HERE/'runs/two_panel_full_10'
rest=dict(np.load(RUN/'rest.npz'));state=dict(np.load(RUN/'lift_teddy.npz'))
config=json.loads((HERE/'config_guarded_detour_trial.json').read_text());policy=_policy(config)
original=state['bear'].copy();tcp=state['tcp_right'].copy();rotation=np.diag([-1.,-1.,1.])
bear=(original-tcp)@rotation.T+tcp
assert np.max(abs(bear[:,2]-original[:,2]))<1e-14
assert np.max(abs(signed_volumes(bear,rest['bear_tets'])-signed_volumes(original,rest['bear_tets'])))<1e-14
polygon,triangles,mouth=_mouth(rest,state,config);anchor=polygon_centroid(polygon)
weights=np.abs(signed_volumes(rest['bear'],rest['bear_tets']));com=np.average(bear[rest['bear_tets']].mean(1),axis=0,weights=weights)
sections,envelopes=_section_envelopes(bear,rest['bear_tets']);planner=Planner(config);saved_q=planner.d.qpos.copy()
rel=bear-tcp;outside_z=max(float(tcp[2]),mouth['rim_z_max_m']+policy['planned_outside_clearance_m']-float(rel[:,2].min()))
end_z=config['table']['height']+policy['bottom_clearance_m']-float(rel[:,2].min())
rows=[]
for x in (0.,.012,.024):
 for y in (0.,.020,.040):
  offset=np.array([x,y]);target=anchor+offset;inside,dist=projected_inside(target[None],polygon)
  delta=target-com[:2];translation=np.r_[delta,outside_z-tcp[2]]
  sweep=check_translation_sweep(bear,rest['bear_tets'],state['q_left'],config,
      translation_m=translation,displacement_m=[0.,0.,end_z-outside_z],required_gap_m=.005,planner=planner)
  evaluation=_evaluate_offset(envelopes,delta,polygon,triangles,mouth,policy)
  summary={k:v for k,v in evaluation.items() if k!='sections'}
  row=dict(offset_from_actual_mouth_centroid_xy_m=offset.tolist(),target_com_xy_m=target.tolist(),
      com_inside=bool(inside[0]),com_boundary_distance_m=float(dist[0]),
      within_existing35mm_initial_offset_radius=bool(np.linalg.norm(offset)<=.035),
      hypothetical_outside_tcp_world_m=np.r_[tcp[:2]+delta,outside_z].tolist(),
      hypothetical_insert_tcp_world_m=np.r_[tcp[:2]+delta,end_z].tolist(),
      vertical_left_robot_sweep=sweep,section_geometry=summary,
      satisfies_frozen_left_gap_com_perimeter=bool(sweep['status']=='clear' and inside[0] and dist[0]>=.002 and evaluation['perimeter_margin_m']>=0),
      satisfies_current_declared_section_heuristics=not evaluation['section_guard_rejections'])
  rows.append(row)
  print(json.dumps(dict(offset_mm=(offset*1000).tolist(),sweep=sweep['status'],gap_mm=1000*sweep['conservative_gap_lower_bound_m'],perimeter_spare_mm=1000*evaluation['perimeter_margin_m'],overlap=evaluation['minimum_section_slab_overlap_fraction'],outside_mm=1000*evaluation['maximum_outside_excursion_bound_m'],cominside=bool(inside[0])),allow_nan=False),flush=True)
assert np.array_equal(saved_q,planner.d.qpos)
assert np.array_equal(state['bear'],original)
report=dict(kind='hypothetical_world_z_180_teddy_frozen_geometry_review',physical_success=False,simulation_run=False,
    template='full10 actual lift soft shape, rotated by exactly 180 degrees around world Z through its actual TCP; no changed dimensions, topology or constitutive parameters',
    actual_mirrored_table_grasp_validated=False,actual_mirrored_teddy_shape_observed=False,right_robot_pose_or_grasp_path_validated=False,
    original_template_time_s=float(state['time']),world_rotation_matrix=rotation.tolist(),rotation_center_tcp_world_m=tcp.tolist(),
    template_input_sha256={name:hashlib.sha256((RUN/name).read_bytes()).hexdigest() for name in ('lift_teddy.npz','rest.npz','config.json')},
    helper_source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('contact_insertion.py','contact_robot_geometry.py')},
    mouth_area_centroid_xy_m=anchor.tolist(),mouth=mouth,rotated_template_com_world_m=com.tolist(),
    fixed_measured_left_q9=state['q_left'].tolist(),engineering_geometry_policy=policy,
    vertical_capacity_margin_m=float(mouth['rim_z_min_m']-.002-(config['table']['height']+policy['bottom_clearance_m'])-np.ptp(bear[:,2])),
    bounded_candidate_count=len(rows),rows=rows,input_arrays_and_planner_qpos_unchanged=True,
    any_frozen_left_gap_com_perimeter_channel=any(r['satisfies_frozen_left_gap_com_perimeter'] for r in rows),
    any_channel_also_meets_current_section_heuristics=any(r['satisfies_frozen_left_gap_com_perimeter'] and r['satisfies_current_declared_section_heuristics'] for r in rows),
    limitations=['This counterfactual rotation does not model the actual mirrored upper-arm grasp, settling, shape changes, inertia or frictional retention.',
      'Frozen vertical teddy sweep versus all22 measured left collision geoms is certified only where clear; horizontal approach, right robot IK/rigid routes and future left cloth motion are not checked.',
      'Rim perimeter and section overlap are finite engineering trial screens, not proof of nonplanar deformable passage or final bag containment.',
      'Some requested offsets exceed the existing35mm initial COM radius and are explicitly marked; no runtime configuration was changed.',
      'A different actual grasp layout requires a fresh physical prefix and repeated actual-shape/force/torque/grasp/deformation checks.'])
(RUN/'hypothetical_z180_teddy_geometry_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('SAVED',str(RUN/'hypothetical_z180_teddy_geometry_review.json'),flush=True)
