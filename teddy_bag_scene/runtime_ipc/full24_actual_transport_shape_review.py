"""Current transport only: physical/shape/route and prior foot material alignment."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_24';OLD=ROOT/'runs/two_panel_full_23'
sys.path.insert(0,str(RUN/'source'))
from contact_insertion import _state_context,_policy,_mouth,_serial
from validate import mouth_structure
from insertion_candidates import projected_inside,polygon_centroid
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[RUN/'config.json',RUN/'rest.npz',RUN/'transport_teddy.npz',RUN/'guarded_contact_reobserved.json',RUN/'guarded_contact_candidate.json',RUN/'orient_teddy_postlift.npz',OLD/'transport_teddy.npz',OLD/'rest.npz',OLD/'config.json',OLD/'states.npz',RUN/'source/contact_insertion.py']
before={str(p):sha(p) for p in paths};cfg=json.loads((RUN/'config.json').read_text());rest=dict(np.load(RUN/'rest.npz'));state=dict(np.load(RUN/'transport_teddy.npz'));old=dict(np.load(OLD/'transport_teddy.npz'));fresh=json.loads((RUN/'guarded_contact_reobserved.json').read_text());initial=json.loads((RUN/'guarded_contact_candidate.json').read_text());report=fresh['report'];ctx=_state_context(rest,state,cfg,_policy(cfg));polygon,_,mouth=_mouth(rest,state,cfg);body,rim,topology=mouth_structure(rest['bag'],rest['bag_faces'],cfg)
checks={}
def check(k,v):checks[k]=bool(v)
check('new_actual_transport_and_fresh_times_match',float(state['time'])==fresh['observed_time_s']==report['observed_time_s'] and float(state['time'])>initial['observed_time_s'])
check('new_actual_shape_not_full23_reuse',not np.array_equal(state['bear'],old['bear']) and not np.array_equal(state['bag'],old['bag']))
check('fresh_policy_unchanged_from_initial_and_code',report['policy']==initial['report']['policy']==_policy(cfg))
check('fresh_physical_context_clear',not ctx['reasons'])
for k in ['minimum_tetra_volume_ratio','maximum_bag_edge_stretch_ratio','bag_table_penetration_m','bear_table_penetration_m','left_designated_material_distance_m','right_nearest_teddy_distance_m']:
 check('recomputed_'+k,abs(ctx['physical'][k]-report['physical_state'][k])<1e-12)
check('fresh_actual_COM_matches_restvolume_weighted',np.max(abs(ctx['com']-report['observed_center_of_mass_world_m']))<1e-12)
check('original_single_initial_offset',cfg['task']['contact_trial']['center_offsets_xy_m']==[[.015,.03075]] and cfg['task']['contact_trial']['recenter_enabled'] is False)
check('fresh_no_XY_correction_or_extra_stage',report['selected_sections']['translation_xy_m']==[0.,0.] and [s['name'] for s in fresh['route']['stages']]==['insert_candidate'])
stage=fresh['route']['stages'][0]
check('fresh_insert_uses_actual_XY',np.max(abs(np.asarray(stage['right_tcp_m'])[:2]-state['tcp_right'][:2]))<1e-12)
check('fresh_full_R_from_actual_FK',np.max(abs(np.asarray(stage['rotation_xyzw'])-ctx['rotation'].as_quat()))<1e-12 and report['fresh_rotation_error_rad']<=.015)
check('fresh_original_vertical_clearance_passes',report['observed_clearance_above_highest_rim_m']>=.04 and report['vertical_capacity']['height_margin_m']>=0)
check('fresh_no_rejections_or_claimed_physical_success',fresh['status']=='candidate' and not report['rejections'] and not fresh['physical_success'] and not fresh['runtime_execution_performed'])

def local_row(label,s):
 bag=s['bag'];bear=s['bear'];rp=bag[rim];chosen=[66,92,60];p=bear[chosen];inside,dist=projected_inside(p[:,:2],rp[:,:2]);mesh=trimesh.Trimesh(bag,body,process=False);closest,d,face=trimesh.proximity.closest_point(mesh,p)
 rows=[]
 for j,nid in enumerate(chosen):
  v=rp[:,:2];w=np.roll(v,-1,axis=0);ab=w-v;frac=np.clip(np.sum((p[j,:2]-v)*ab,axis=1)/np.sum(ab*ab,axis=1),0,1);pr=v+frac[:,None]*ab;edge=int(np.argmin(np.linalg.norm(pr-p[j,:2],axis=1)));tri=bag[body[face[j]]];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal);worldface=np.flatnonzero(np.all(rest['bag_faces']==body[face[j]],axis=1))[0]
  rows.append({'material_node':nid,'actual_world_m':p[j].tolist(),'actual_relative_TCP_m':(p[j]-s['tcp_right']).tolist(),'projected_inside_current_rim':bool(inside[j]),'projected_rim_distance_m':float(dist[j]),'closest_projected_rim_edge_material_ids':[int(rim[edge]),int(rim[(edge+1)%len(rim)])],'closest_projected_rim_xy_m':pr[edge].tolist(),'rim_to_foot_projected_vector_m':(p[j,:2]-pr[edge]).tolist(),'exact_node_to_body_triangle_distance_m':float(d[j]),'body_nearest_global_face':int(worldface),'body_nearest_point_world_m':closest[j].tolist(),'body_face_oriented_normal':normal.tolist()})
 face1407=bag[rest['bag_faces'][1407]];normal=np.cross(face1407[1]-face1407[0],face1407[2]-face1407[0]);normal/=np.linalg.norm(normal)
 return {'label':label,'actual_time_s':float(s['time']),'foot_points':rows,'full_bear_bbox_min_m':bear.min(0).tolist(),'full_bear_bbox_max_m':bear.max(0).tolist(),'whole_bear_minZ_above_highest_rim_m':float(bear[:,2].min()-rp[:,2].max()),'original_first_contact_front_face1407_oriented_normal':normal.tolist(),'original_face1407_material_world_vertices_m':face1407.tolist()}
rows=[local_row('full23_original_transport',old),local_row('full24_current_actual_transport',state)]
hist=dict(np.load(OLD/'states.npz'));i=int(np.argmin(abs(hist['time']-43.03)));rows.append(local_row('full23_first_observed_body_near_contact',{'time':hist['time'][i],'bag':hist['bag'][i],'bear':hist['bear'][i],'tcp_right':hist['tcp_right'][i]}))
check('inputs_unchanged',all(sha(Path(p))==h for p,h in before.items()))
result={'scope':'Only current actual full24 transport versus full23 material points involved in first observed front-panel contact. No planning, future frames, synthetic soft mutation or active-run change.','physical_success':False,'status':'available_checks_passed' if all(checks.values()) else 'rejected','check_count':len(checks),'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],'input_sha256':before,'actual_state_sha256':sha(RUN/'transport_teddy.npz'),'actual_time_s':float(state['time']),'actual_physical_recomputed':ctx['physical'],'actual_COM_world_m':ctx['com'].tolist(),'actual_COM_offset_from_current_mouth_centroid_m':(ctx['com'][:2]-polygon_centroid(polygon)).tolist(),'current_actual_COM_minus_original_selected_COM_xy_m':(ctx['com'][:2]-np.asarray(initial['report']['selected_sections']['center_of_mass_xy_target_m'])).tolist(),'actual_TCP_difference_from_full23_transport_m':(state['tcp_right']-old['tcp_right']).tolist(),'relative_to_TCP_material_shape_difference_from_full23':{'maximum_node_difference_m':float(np.linalg.norm((state['bear']-state['tcp_right'])-(old['bear']-old['tcp_right']),axis=1).max()),'mean_node_difference_m':float(np.linalg.norm((state['bear']-state['tcp_right'])-(old['bear']-old['tcp_right']),axis=1).mean())},'vertical_capacity':report['vertical_capacity'],'section_summary':{k:v for k,v in report['selected_sections'].items() if k!='sections'},'actual_foot_alignment_comparison':rows,'limits':['Horizontal point inclusion is local alignment evidence, not whole-object body passage or dynamic clearance.','Full23 first near-contact is an earlier run after lowering, not a matched-time prediction of full24.','Bag and teddy continue changing after transport; future body contact, rear-handle contact, force/torque and final containment remain native verification tasks.','Saved force fields are control forces, not a substitute for native IPC contact reaction traces.']}
out=RUN/'independent-actual-transport-shape-alignment-review.json';out.write_text(json.dumps(_serial(result),indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(out),'sha256':sha(out),'checks':len(checks),'failed':result['failed_checks'],'current_offset':result['actual_COM_offset_from_current_mouth_centroid_m'],'TCP_delta':result['actual_TCP_difference_from_full23_transport_m'],'relative_shape':result['relative_to_TCP_material_shape_difference_from_full23'],'foot':[{ 'label':x['label'],'time':x['actual_time_s'],'points':[{k:p[k] for k in ['material_node','projected_inside_current_rim','projected_rim_distance_m','exact_node_to_body_triangle_distance_m','closest_projected_rim_edge_material_ids']} for p in x['foot_points']]} for x in rows]},indent=2))
