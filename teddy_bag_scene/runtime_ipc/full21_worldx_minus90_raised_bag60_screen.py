"""Single unexecuted raised-bag hypothesis; stop at necessary global depth bound."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_21';sys.path.insert(0,str(RUN/'source'))
from validate import mouth_structure,signed_volumes
from motion import Planner
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[RUN/'config.json',RUN/'rest.npz',RUN/'orient_teddy_postlift.npz',RUN/'source/validate.py',RUN/'source/motion.py',RUN/'source/assets/panda_tactile.xml',RUN/'source/assets/two_panel_bag_wide_tall_coarse.npz',Path(__file__)]
hashes={str(p):sha(p) for p in inputs}
cfg=json.loads((RUN/'config.json').read_text());cfg['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
for k in ('npz_file','metadata'):cfg['bag'][k]=str(RUN/'source/assets'/cfg['bag'][k])
rest=dict(np.load(RUN/'rest.npz'));actual=dict(np.load(RUN/'orient_teddy_postlift.npz'))
faces,rim,topology=mouth_structure(rest['bag'],rest['bag_faces'],cfg);assert topology['two_panels_directly_sewn_along_u'] and len(rim)==74
body_ids=np.unique(faces);bag=actual['bag'];bear=actual['bear'];tcp=actual['tcp_right'];pl=Planner(cfg)
fk_tcp,R=pl.fk('right',actual['q_right'][:7]);G=Rotation.from_rotvec([-np.pi/2,0.,0.]);target=G*R
turned=G.apply(bear-tcp)+tcp
weights=np.abs(signed_volumes(rest['bear'],rest['bear_tets']));com=np.sum(bear[rest['bear_tets']].mean(1)*weights[:,None],axis=0)/weights.sum();turned_com=G.apply(com-tcp)+tcp
shift=np.array([0.,0.,.060]);raised=bag+shift
margin=.002;table_floor=float(cfg['table']['height'])+.010
old_rim_min=float(bag[rim,2].min());new_rim_min=float(raised[rim,2].min());old_body_min=float(bag[body_ids,2].min());new_body_min=float(raised[body_ids,2].min())
old_top=old_rim_min-margin;new_top=new_rim_min-margin;H=float(np.ptp(turned[:,2]));body_capacity=new_top-new_body_min;table_capacity=new_top-table_floor
assert body_capacity < H and table_capacity>H
z_max_at_top=new_top-float(turned[:,2].max());z_min_at_body=new_body_min-float(turned[:,2].min());z_min_at_table=table_floor-float(turned[:,2].min())
report=dict(kind='full21_single_worldX_minus90_and_rigid_bag_plus60mm_necessary_geometry_screen',status='rejected_necessary_lowest_rim_vs_body_global_depth',physical_success=False,native_executed=False,gpu_used=False,production_modified=False,
 source_actual_state=str(RUN/'orient_teddy_postlift.npz'),source_time_s=float(actual['time']),input_sha256=hashes,
 hypothetical_transform=dict(bear_rotation_world_axis=[1.,0.,0.],bear_world_rotation_degrees=-90.,bear_rotation_about_measured_tcp_m=tcp.tolist(),actual_full_hand_rotation_xyzw=R.as_quat().tolist(),target_full_hand_rotation_xyzw=target.as_quat().tolist(),target_times_actual_inverse_rotvec_rad=(target*R.inv()).as_rotvec().tolist(),actual_fk_tcp_vs_measured_error_m=float(np.linalg.norm(fk_tcp-tcp)),
  whole_bag_rigid_translation_m=shift.tolist(),left_tcp_translation_hypothesis_m=shift.tolist(),hypothetical_left_tcp_m=(actual['tcp_left']+shift).tolist(),left_orientation_unchanged=True,left_robot_joint_solution_assessed=False,
  comment='Both transformations exist only in diagnostic array copies. The bag translation is not a simulated cloth response, and the left pose is not assigned to a robot.'),
 geometry=dict(actual_body_face_count=len(faces),actual_body_vertex_count=len(body_ids),actual_rim_vertex_count=len(rim),actual_bear_dimensions_m=np.ptp(bear,axis=0).tolist(),rotated_bear_dimensions_m=np.ptp(turned,axis=0).tolist(),rotated_bear_bounds_at_original_tcp_m=[turned.min(0).tolist(),turned.max(0).tolist()],rotated_mass_com_at_original_tcp_m=turned_com.tolist(),
  original_body_min_z_m=old_body_min,original_body_max_z_m=float(bag[body_ids,2].max()),original_rim_min_z_m=old_rim_min,original_rim_max_z_m=float(bag[rim,2].max()),
  raised_body_min_z_m=new_body_min,raised_body_max_z_m=float(raised[body_ids,2].max()),raised_rim_min_z_m=new_rim_min,raised_rim_max_z_m=float(raised[rim,2].max()),
  raised_whole_cloth_min_z_m=float(raised[:,2].min()),lowest_body_vs_lowest_rim_span_unchanged_m=new_rim_min-new_body_min),
 unchanged_requirements=dict(endpoint_below_lowest_rim_m=margin,bottom_clearance_above_table_m=.010,table_height_m=cfg['table']['height'],minimum_bear_bottom_from_table_m=table_floor,maximum_bear_top_m=new_top,
  no_penetration_or_compression_tolerance_relaxed=True,no_finger_or_material_or_base_or_scale_change=True),
 necessary_conditions=dict(frozen_bear_height_m=H,table_to_lowest_rim_available_height_m=table_capacity,table_to_rim_margin_m=table_capacity-H,table_and_rim_height_condition_passed=True,
  global_body_min_to_lowest_rim_available_height_m=body_capacity,global_body_depth_deficit_m=H-body_capacity,global_body_depth_condition_passed=False,
  minimum_vertical_translation_to_keep_bear_above_global_body_min_m=z_min_at_body,maximum_vertical_translation_to_keep_bear_below_rim_m=z_max_at_top,minimum_vertical_translation_from_table_m=z_min_at_table,
  allowed_vertical_translation_interval_nonempty=False,
  bear_lowest_point_at_highest_rim_compliant_placement_m=float(turned[:,2].min()+z_max_at_top),distance_below_all_body_triangles_at_that_placement_m=H-body_capacity,
  bear_highest_point_at_lowest_global_body_compliant_placement_m=float(turned[:,2].max()+z_min_at_body),excess_above_allowed_top_at_that_placement_m=H-body_capacity,
  body_translation_cannot_change_depth_deficit=True),
 rigid_rotation_signed_tet_volume_max_abs_error_m3=float(np.max(np.abs(signed_volumes(turned,rest['bear_tets'])-signed_volumes(bear,rest['bear_tets'])))),
 right_wrist_ik_assessed=False,right_robot_rigid_path_assessed=False,whole_bear_left_robot_sweep_assessed=False,
 reason_not_running_ik='A necessary geometry condition already fails for every XYZ placement of this frozen orientation under the unchanged lowest-rim rule, so IK or collision planning cannot make this exact rigidly shifted hypothesis contain the whole bear.',
 mathematical_scope=['For a triangle surface every point lies within the coordinate extrema of its vertices. Any enclosed bag-body region, closed at its actual rim by a cap at or above the lowest rim, has no interior below the minimum body vertex z. This is a necessary global bound, independent of XY alignment; it is not a sufficiency/embedding claim.',
  'Whole containment plus the unchanged 2 mm lowest-rim endpoint rule therefore requires bear_height <= rim_min - 2 mm - body_min. Rigidly translating all cloth preserves this right-hand side.',
  'The bag-global minimum is already a generous lower bound, not an assertion of a flat usable floor at that height. Cross-section, handle and self-contact constraints can only make frozen placement harder.',
  'The unchanged table rule alone now passes. It does not establish geometric containment in the raised cloth body.',
  'Actual native raising could unfold, tilt, compress or otherwise deform both objects. This report rejects only the requested frozen-shape, rigid-cloth-translation hypothesis, not all physically possible higher holds.',
  'No claim about right-arm IK or native collision feasibility is made because those tests were intentionally not run after the necessary-condition rejection.'])
report['inputs_unchanged']=all(sha(Path(p))==h for p,h in hashes.items())
out=RUN/'independent-worldx-minus90-raised-bag60-screen.json';out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(path=str(out),sha256=sha(out),status=report['status'],bear_height_mm=H*1000,available_body_height_mm=body_capacity*1000,body_deficit_mm=(H-body_capacity)*1000,table_available_height_mm=table_capacity*1000,inputs_unchanged=report['inputs_unchanged']),indent=2))
