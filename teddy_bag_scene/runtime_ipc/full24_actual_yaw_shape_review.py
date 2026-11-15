"""Bounded fresh actual-yaw physical/COM/route-contract inspection; no planning."""
from pathlib import Path
import sys,hashlib,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_24';OLD=ROOT/'runs/two_panel_full_23'
sys.path.insert(0,str(RUN/'source'))
from contact_insertion import _state_context,_policy,_mouth,_serial
from insertion_candidates import polygon_centroid
from validate import signed_volumes
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[RUN/'config.json',RUN/'rest.npz',RUN/'orient_teddy_postlift.npz',RUN/'guarded_contact_candidate.json',OLD/'config.json',OLD/'orient_teddy_postlift.npz',RUN/'source/contact_insertion.py',RUN/'source/motion.py',ROOT/'config_guarded_inward_15mm_trial.json']
before={str(p):sha(p) for p in paths};cfg=json.loads((RUN/'config.json').read_text());oldcfg=json.loads((OLD/'config.json').read_text());state=dict(np.load(RUN/'orient_teddy_postlift.npz'));old=dict(np.load(OLD/'orient_teddy_postlift.npz'));rest=dict(np.load(RUN/'rest.npz'));c=json.loads((RUN/'guarded_contact_candidate.json').read_text());r=c['report'];checks={}
def check(k,v):checks[k]=bool(v)
expected=copy.deepcopy(oldcfg);expected['task']['contact_trial']['center_offsets_xy_m']=[[.015,.03075]]
check('only_config_change_from_full23_is_declared_initial_offset',cfg==expected)
check('frozen_config_equals_authorized_candidate',cfg==json.loads((ROOT/'config_guarded_inward_15mm_trial.json').read_text()))
check('source_state_filename_and_candidate_time_match',float(state['time'])==c['observed_time_s']==r['observed_time_s'])
check('candidate_is_current_actual_source_not_old_file',sha(RUN/'orient_teddy_postlift.npz')!=sha(OLD/'orient_teddy_postlift.npz') and not np.array_equal(state['bag'],old['bag']) and not np.array_equal(state['bear'],old['bear']))
check('rest_topology_unchanged',all(np.array_equal(v,np.load(OLD/'rest.npz')[k]) for k,v in rest.items()))
ctx=_state_context(rest,state,cfg,_policy(cfg));polygon,triangles,mouth=_mouth(rest,state,cfg)
check('fresh_physical_context_has_no_rejections',not ctx['reasons']);check('full_active_policy_matches_frozen_code',r['policy']==_policy(cfg))
for k in ['minimum_tetra_volume_ratio','maximum_bag_edge_stretch_ratio','bag_table_penetration_m','bear_table_penetration_m','left_designated_material_distance_m','right_nearest_teddy_distance_m']:
 check('physical_'+k+'_recomputed',abs(ctx['physical'][k]-r['physical_state'][k])<1e-12)
weights=np.abs(signed_volumes(rest['bear'],rest['bear_tets']));com=np.average(state['bear'][rest['bear_tets']].mean(1),axis=0,weights=weights);centroid=polygon_centroid(polygon);offset=np.array([.015,.03075]);targetcom=centroid+offset
check('actual_rest_volume_COM_matches',np.max(abs(com-r['observed_center_of_mass_world_m']))<1e-12)
check('actual_mouth_centroid_matches',np.max(abs(centroid-r['observed_mouth_area_centroid_xy_m']))<1e-12)
check('sole_declared_offset_exact',r['declared_com_offsets_from_mouth_area_centroid_m']==[[.015,.03075]] and len(r['centering_candidates'])==len(r['robot_centering_attempts'])==1)
selection=r['selected_sections'];check('selected_COM_exact_centroid_plus_offset',np.max(abs(targetcom-selection['center_of_mass_xy_target_m']))<1e-12)
check('selected_translation_exact',np.max(abs(targetcom-com[:2]-selection['translation_xy_m']))<1e-12)
check('active_policy_section_only_explicit',selection['active_rim_perimeter_policy']=='deformable_rim_contact' and selection['active_rim_perimeter_criterion']=='buffered_section_only')
check('old_union_negative_remains_diagnostic',selection['perimeter_margin_m']<0 and bool(selection['mouth_union_perimeter_rejections']) and selection['mouth_union_perimeter_is_diagnostic_only'])
check('active_margin_algebraic',abs(selection['active_perimeter_margin_m']-(selection['available_rim_perimeter_m']-selection['maximum_buffered_section_envelope_perimeter_m']))<1e-12)
check('active_and_overlap_lateral_COM_height_gates_pass',not selection['section_guard_rejections'] and selection['active_perimeter_margin_m']>=0 and selection['minimum_section_slab_overlap_fraction']>=.4 and selection['maximum_outside_excursion_bound_m']<=selection['allowed_outside_excursion_m'] and selection['com_boundary_distance_m']>=.002 and r['vertical_capacity']['height_margin_m']>=0)
stages=c['route']['stages'];names=[s['name'] for s in stages];expectednames=['height_at_current_xy','transport_y_at_current_x','translate_outside_mouth','insert_candidate']
check('route_exact_original_stage_order_no_detour',names==expectednames);check('executable_stages_only_preparation',c['stages_to_execute']==stages[:-1] and c['next_observation']=='before_insertion')
check('recenter_disabled',cfg['task']['contact_trial']['recenter_enabled'] is False)
check('all_route_R_exact_actual_fullFK',all(np.max(abs(np.asarray(s['rotation_xyzw'])-ctx['rotation'].as_quat()))<1e-12 for s in stages))
tcp=state['tcp_right'];targets=np.array([s['right_tcp_m'] for s in stages]);endxy=tcp[:2]+targetcom-com[:2]
check('height_stage_keeps_actual_XY',np.max(abs(targets[0,:2]-tcp[:2]))<1e-12)
check('first_transport_only_Y',abs(targets[1,0]-targets[0,0])<1e-12 and abs(targets[1,2]-targets[0,2])<1e-12 and abs(targets[1,1]-endxy[1])<1e-12)
check('second_transport_only_X_to_selected_COM',abs(targets[2,1]-targets[1,1])<1e-12 and abs(targets[2,2]-targets[1,2])<1e-12 and np.max(abs(targets[2,:2]-endxy))<1e-12)
check('provisional_insert_strict_vertical',np.max(abs(targets[3,:2]-targets[2,:2]))<1e-12 and targets[3,2]<targets[2,2])
check('original_robot_and_soft_margins_unchanged',all(s['rigid_clearance_required_m']==.008 for s in stages) and r['policy']['left_robot_soft_sweep_clearance_m']==.005)
a=r['robot_centering_attempts'][0];soft=a['left_robot_teddy_sweep_segments'];check('recorded_wholebear_left_all_four_segments_clear',a['rejection'] is None and len(soft)==4 and all(x['status']=='clear' and x['conservative_gap_lower_bound_m']>=.005 for x in soft))
check('candidate_accepted_with_no_physical_success_claim',c['status']=='candidate' and not r['rejections'] and c['physical_success'] is False and c['runtime_execution_performed'] is False)
check('all_read_inputs_unchanged',all(sha(Path(p))==h for p,h in before.items()))
result={'scope':'Bounded read-only current full24 actual postlift-yaw inspection. No new planning, full-history audit, new candidate or production mutation. Future route is predicted from this actual state; native transport/insert not certified.','physical_success':False,'status':'available_checks_passed' if all(checks.values()) else 'rejected','checks':checks,'check_count':len(checks),'failed_checks':[k for k,v in checks.items() if not v],'actual_time_s':float(state['time']),'actual_state_sha256':sha(RUN/'orient_teddy_postlift.npz'),'input_sha256':before,'actual_physical_recomputed':ctx['physical'],'actual_bear_bbox_min_m':state['bear'].min(0).tolist(),'actual_bear_bbox_max_m':state['bear'].max(0).tolist(),'actual_bear_extents_m':np.ptp(state['bear'],axis=0).tolist(),'actual_COM_world_m':com.tolist(),'actual_mouth_area_centroid_xy_m':centroid.tolist(),'declared_COM_offset_m':offset.tolist(),'selected_COM_world_xy_m':targetcom.tolist(),'source_shape_difference_from_full23_same_phase':{k:{'maximum_material_node_world_difference_m':float(np.linalg.norm(state[k]-old[k],axis=1).max()),'mean_material_node_world_difference_m':float(np.linalg.norm(state[k]-old[k],axis=1).mean())} for k in ['bag','bear']},'vertical_capacity':r['vertical_capacity'],'section_guard_summary':{k:v for k,v in selection.items() if k!='sections'},'route_stage_names':names,'route_TCP_targets_m':targets.tolist(),'recorded_wholebear_left_frozen_lower_bounds_m':{x['stage']:x['conservative_gap_lower_bound_m'] for x in soft},'not_certified':['This limited review recomputes current state physics/COM and checks route contract; it does not redo complete IK or raw401 history.','Reported full-route robot/soft certificates remain frozen-shape planning, not native dynamic collision or cloth passage.','Saved force arrays are PD/control values and cannot replace native IPC contact guard trace; no force/torque or every-step coverage claim is made.','Full24 actual transport must be freshly reobserved before insertion. Active section perimeter only is a contact-trial condition, not proof of 3D bag fit.']}
out=RUN/'independent-actual-yaw-shape-route-review.json';out.write_text(json.dumps(_serial(result),indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(out),'sha256':sha(out),'status':result['status'],'checks':len(checks),'failed':result['failed_checks'],'COM':com.tolist(),'shape_differences':result['source_shape_difference_from_full23_same_phase'],'soft':result['recorded_wholebear_left_frozen_lower_bounds_m']},indent=2))
