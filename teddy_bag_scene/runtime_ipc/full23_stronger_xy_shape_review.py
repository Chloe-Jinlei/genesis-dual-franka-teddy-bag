"""Bounded frozen actual-transport shape diagnostics for the sole stronger XY backup."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_23';OUT=RUN/'independent-stronger-xy-backup/actual_transport_shape'
sys.path.insert(0,str(RUN/'source'));sys.path.insert(1,str(ROOT/'runtime_ipc'))
from validate import mouth_structure
from insertion_candidates import projected_inside
from diagnose_low_region_winding import diagnose,serialize
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=[RUN/'transport_teddy.npz',RUN/'rest.npz',RUN/'config.json',RUN/'states.npz',OUT/'candidate.json'];before={str(p):sha(p) for p in sources}
state=dict(np.load(RUN/'transport_teddy.npz')); rest=dict(np.load(RUN/'rest.npz'));cfg=json.loads((RUN/'config.json').read_text());hist=dict(np.load(RUN/'states.npz'));c=json.loads((OUT/'candidate.json').read_text());assert c['status']=='candidate'
target=np.asarray(c['route']['insert_tcp_m']);delta=target-state['tcp_right'];query=OUT/'hypothetical-frozen-insert-queries.npz';np.savez_compressed(query,time=state['time'],bag=state['bag'],bear=state['bear']+delta)
r=diagnose(RUN,query)
r.update(observation_kind='hypothetical_frozen_translation_queries_not_a_native_state',time_is_source_observation_only=True,source_actual_time_s=float(state['time']),source_path=str(RUN/'transport_teddy.npz'),source_sha256=sha(RUN/'transport_teddy.npz'),translation_world_m=delta.tolist(),scope='Actual transport bear/cloth held fixed, bear translated exactly to the sole counterfactual planned endpoint. Not actual descent, deformed response, force feasibility or whole-tet containment. This is only final node/COM low-region evidence. Keys named actual in the generic diagnostic refer to supplied frozen query data, not a native endpoint.')
(OUT/'hypothetical-final-pointwise-body.json').write_text(json.dumps(r,indent=2,default=serialize,allow_nan=False)+'\n')
_,rim,_=mouth_structure(rest['bag'],rest['bag_faces'],cfg);shift=np.array([.015,-.00375,0.]);points=[]
for t in [40.28,42.98,43.03,43.08]:
 i=int(np.argmin(abs(hist['time']-t)));bag=hist['bag'][i];bear=hist['bear'][i];p=bear[66];inside,d=projected_inside(np.array([p[:2],(p+shift)[:2]]),bag[rim,:2]);points.append({'actual_time_s':float(hist['time'][i]),'material_foot_node':66,'actual_world_point_m':p.tolist(),'unshifted_projected_inside':bool(inside[0]),'unshifted_boundary_distance_m':float(d[0]),'counterfactual_shifted_projected_inside':bool(inside[1]),'counterfactual_boundary_distance_m':float(d[1])})
w=r.get('cap_winding',{});non=w.get('non_unit_winding_query_indices',[]);n=len(state['bear'])
summary={'physical_success':False,'scope':'One declared +15X/-3.75Y counterfactual only. Point projection is not a whole-object collision certificate; same old saved cloth does not predict future cloth dynamics.','actual_transport_candidate':str(OUT/'candidate.json'),'section_and_height':{'vertical_capacity':c['report']['vertical_capacity'],'selected_sections':{k:v for k,v in c['report']['selected_sections'].items() if k!='sections'}},'frozen_endpoint_pointwise_status':r['status'],'frozen_endpoint_all_nodes_inside':r['all_bear_vertices_inside_by_guarded_unit_winding'],'frozen_endpoint_com_inside':r['bear_com_inside_by_guarded_unit_winding'],'frozen_endpoint_raw_unit_node_count':n-sum(i<n for i in non) if w else None,'denominator_nodes':n,'node_count_is_not_volume_fraction_or_tet_containment':True,'frozen_endpoint_reasons':r['reasons'],'point_direction_checks':points,'input_sha256':before,'source_before_after_identical':all(sha(Path(p))==h for p,h in before.items())}
(OUT/'shape-review.json').write_text(json.dumps(summary,indent=2,default=serialize,allow_nan=False)+'\n');print(json.dumps(summary,indent=2,default=serialize))
