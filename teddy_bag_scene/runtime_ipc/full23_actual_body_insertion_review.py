"""Saved native body geometry only. Low-region points, not a replacement validator."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_23'
sys.path.insert(0,str(RUN/'source'));sys.path.insert(1,str(ROOT/'runtime_ipc'))
from validate import mouth_structure,mesh_edges,signed_volumes,winding,proper_segment_crossings,tetra_boundary,final_contact_audit
from diagnose_low_region_winding import capped_surface,spatial_rim
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=['rest.npz','config.json','states.npz','failed_soft_state.npz','soft_metrics.npy','contact_guard_trace.json','source/validate.py']
hashes={f:sha(RUN/f) for f in files}
rest=dict(np.load(RUN/'rest.npz'));states=dict(np.load(RUN/'states.npz'));fail=dict(np.load(RUN/'failed_soft_state.npz'))
config=json.loads((RUN/'config.json').read_text());meta=dict(np.load(RUN/'source/assets'/config['bag']['npz_file']))
body_faces,rim,topology=mouth_structure(rest['bag'],rest['bag_faces'],config);_,_,boundary=mesh_edges(body_faces)
weights=np.abs(signed_volumes(rest['bear'],rest['bear_tets']));trace=json.loads((RUN/'contact_guard_trace.json').read_text());tt=np.array([x['time_s'] for x in trace])
body_face_sets={name:set(meta[name].tolist()) for name in ['panel_front_face_indices','panel_back_face_indices']}
face_lookup={tuple(sorted(f)):i for i,f in enumerate(rest['bag_faces'])};body_global=np.array([face_lookup[tuple(sorted(f))] for f in body_faces])
node_sets={name:set(meta[name].tolist()) for name in ['panel_front','panel_back','rim','u_seam']}
rows=[]
indices=[]
for target in [40.28,42.48,42.98,43.48,43.73,43.98,44.23,44.48]:
 i=int(np.argmin(abs(states['time']-target)))
 if i not in indices:indices.append(i)
observations=[(float(states['time'][i]),states['bag'][i],states['bear'][i],'states.npz',i) for i in indices]
observations.append((float(fail['time']),fail['bag'],fail['bear'],'failed_soft_state.npz',None))
for timestamp,bag,bear,source,index in observations:
 rp=bag[rim];h=float(rp[:,2].min());com=np.average(bear[rest['bear_tets']].mean(1),axis=0,weights=weights);queries=np.vstack([bear,com]);low=queries[:,2]<=h-.002
 body=trimesh.Trimesh(bag,body_faces,process=False);distance=[];closest=[];triids=[]
 for start in range(0,len(queries),256):
  cp,dd,ti=trimesh.proximity.closest_point(body,queries[start:start+256]);closest.extend(cp);distance.extend(dd);triids.extend(ti)
 distance=np.array(distance);closest=np.array(closest);triids=np.array(triids)
 apices=[rp.mean(0),rp.mean(0)+np.array([.031,-.023,.15])];ww=[]
 for apex in apices:
  vv,ff=capped_surface(bag,body_faces,boundary,apex);ww.append(winding(queries,vv,ff))
 wa,wb=ww;res=np.maximum(abs(wa-np.rint(wa)),abs(wb-np.rint(wb)));agreement=abs(wa-wb)<=1e-8;multiple=(abs(np.rint(wa))>1)|(abs(np.rint(wb))>1)
 area2=np.linalg.norm(np.cross(bag[body_faces[:,1]]-bag[body_faces[:,0]],bag[body_faces[:,2]]-bag[body_faces[:,0]]),axis=1)
 spatial=spatial_rim(rp,rest['bag'][rim],config['bag']['thickness'])
 topology_ok=body.is_winding_consistent and area2.min()>1e-14 and spatial['passed']
 guarded=low & agreement & (res<=1e-6)&(~multiple)&(distance>1e-6)&topology_ok
 inside=guarded&(abs(abs(wa)-1)<=1e-6)&(abs(abs(wb)-1)<=1e-6)
 outside=guarded&(abs(wa)<=1e-6)&(abs(wb)<=1e-6)
 min_rim_id=int(rim[np.argmin(rp[:,2])]);ratio=signed_volumes(bear,rest['bear_tets'])/signed_volumes(rest['bear'],rest['bear_tets']);tetid=int(np.argmin(ratio));worst_tet=rest['bear_tets'][tetid]
 force=trace[int(np.argmin(abs(tt-timestamp)))];near=int(np.argmin(distance[:-1]));faceid=int(body_global[triids[near]])
 region=['front' if faceid in body_face_sets['panel_front_face_indices'] else 'back' if faceid in body_face_sets['panel_back_face_indices'] else 'other'][0]
 row={
  'time_s':timestamp,'source':source,'state_index':index,'bear_node_count':len(bear),
  'rim_z_min_max_m':[h,float(rp[:,2].max())],'rim_z_span_m':float(np.ptp(rp[:,2])),
  'lowest_rim_node':min_rim_id,'lowest_rim_world_m':bag[min_rim_id].tolist(),'lowest_rim_rest_local_m':(rest['bag'][min_rim_id]-config['bag']['position']).tolist(),
  'lowest_rim_groups':[name for name,s in node_sets.items() if min_rim_id in s],
  'bear_bbox_min_m':bear.min(0).tolist(),'bear_bbox_max_m':bear.max(0).tolist(),'bear_com_m':com.tolist(),
  'all_nodes_low_region_applicable':bool(low[:-1].all()),'low_node_count':int(low[:-1].sum()),
  'guarded_low_unit_winding_node_count':int(inside[:-1].sum()),'guarded_low_zero_winding_node_count':int(outside[:-1].sum()),
  'low_ambiguous_node_count':int((low&~(inside|outside))[:-1].sum()),
  'higher_node_containment_unknown_count':int((~low[:-1]).sum()),
  'low_zero_winding_node_ids':np.flatnonzero(outside[:-1]).tolist(),
  'guarded_low_unit_fraction_of_all_nodes_not_volume':float(inside[:-1].mean()),
  'guarded_low_outside_fraction_of_all_nodes_not_volume':float(outside[:-1].mean()),
  'COM':{'low_region_applicable':bool(low[-1]),'body_distance_m':float(distance[-1]),'low_unit':bool(inside[-1]) if low[-1] else None,'low_zero':bool(outside[-1]) if low[-1] else None,'raw_two_cap_winding_diagnostic_only':[float(wa[-1]),float(wb[-1])]},
  'all_node_raw_mean_cap_unit_count_diagnostic_only':int((abs(abs(wa[:-1])-1)<=1e-6).sum()),
  'all_node_raw_high_cap_unit_count_diagnostic_only':int((abs(abs(wb[:-1])-1)<=1e-6).sum()),
  'all_node_two_cap_disagreement_count':int((~agreement[:-1]).sum()),
  'low_max_cap_difference':float(abs(wa[low]-wb[low]).max()) if low.any() else None,
  'low_max_integer_residual':float(res[low].max()) if low.any() else None,
  'multiple_winding_count':int(multiple.sum()),'minimum_body_triangle_twice_area_m2':float(area2.min()),'body_orientation_consistent':bool(body.is_winding_consistent),'spatial_rim':spatial,
  'minimum_bear_node_body_distance':{'distance_m':float(distance[near]),'bear_node':near,'bear_world_m':bear[near].tolist(),'cloth_closest_world_m':closest[near].tolist(),'global_cloth_face':faceid,'panel':region},
  'bear_nodes_within_2mm_body_count':int((distance[:-1]<.002).sum()),
  'current_minimum_J':float(ratio.min()),'minimum_J_tet':tetid,'minimum_J_rest_local_center_m':(rest['bear'][worst_tet].mean(0)-config['bear']['position']).tolist(),
  'minimum_J_actual_center_m':bear[worst_tet].mean(0).tolist(),
  'same_time_trace_or_nearest':{'time_s':force['time_s'],'time_error_s':abs(force['time_s']-timestamp),'tcp_world_m':force['aim_tcp_world_m'],'net_force_n':force['net_force_norm_n'],'tcp_torque_nm':force['tcp_torque_norm_nm'],'delta_force_world_n':force['delta_force_world_n'],'right_finger_force_norm_n':force['right_finger_force_norm_n']},
  'physical_success':False}
 rows.append(row)
 print(json.dumps({k:row[k] for k in ['time_s','low_node_count','guarded_low_unit_winding_node_count','guarded_low_zero_winding_node_count','higher_node_containment_unknown_count','current_minimum_J','rim_z_min_max_m']},ensure_ascii=False),flush=True)
# Recompute existing discrete physical cloth-to-teddy audit on the actual failure geometry,
# because original validation uses the last regular frame 30ms earlier.
final_contact=final_contact_audit(fail['bag'],fail['bear'],rest['bag_faces'],rest['bear_tets'])
result={'scope':'Actual saved native state geometry; failed_soft_state is the true 44.510s endpoint. Prior states are saved frames, not continuous contact timing.',
 'rows':rows,'actual_failure_discrete_cloth_contact_audit':final_contact,
 'limitations':['Unit/zero degree is only cap-independent for query points at least 2mm below every actual rim vertex. Above that level the counts and COM cannot be promoted from two arbitrary fan caps to physical containment.',
 'Pointwise counts are not volume fractions or tetrahedron-wide containment. Original body embedding/self-intersection is not independently certified by topology, normal orientation and rim separation.',
 'Closest sampled bear nodes and discrete cloth samples do not locate exact continuous onset or establish which contact caused a native reaction. Peer strap and raw per-link force reviews are separate.',
 'This run stopped while fingers still held the teddy, before release; no successful final containment or stability is claimed.'],
 'input_sha256_before':hashes,'input_sha256_after':{f:sha(RUN/f) for f in files},'physical_success':False}
assert result['input_sha256_before']==result['input_sha256_after']
out=RUN/'independent-actual-body-insertion-review.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SAVED',str(out),sha(out),flush=True)
