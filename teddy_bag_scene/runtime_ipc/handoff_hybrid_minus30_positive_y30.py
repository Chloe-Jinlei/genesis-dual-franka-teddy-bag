"""One authorized COM offset [0,+30 mm], reusing saved minus30 hybrid endpoint; CPU only."""
from pathlib import Path
import copy,hashlib,json,sys,time
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'runs/two_panel_handoff_upright_offset_01'; SOURCE=RUN/'source'; PRIOR=RUN/'independent-hybrid-lift-worldz-minus30-path';OUT=RUN/'independent-hybrid-minus30-positive-y30';OUT.mkdir(exist_ok=True);sys.path.insert(0,str(SOURCE))
from contact_insertion import plan_guarded_contact_trial,_state_context,_policy,_mouth,_serial,_section_envelopes,_evaluate_offset
from insertion_candidates import polygon_centroid,projected_inside
from validate import tetra_boundary,mesh_edges
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[PRIOR/'hybrid_hypothetical_after_yaw.npz',PRIOR/'rest.npz',PRIOR/'review.json',RUN/'config.json',*[SOURCE/n for n in ('contact_insertion.py','motion.py','contact_robot_geometry.py','postlift_yaw.py')],Path(__file__)];hashes={str(p):sha(p) for p in paths}
state={k:v.copy() for k,v in dict(np.load(paths[0])).items()};rest=dict(np.load(paths[1]));prior=json.loads(paths[2].read_text());assert prior['yaw_preflight']['status']=='candidate';assert prior['total_world_z_rotation_degrees']==-30
cfg=json.loads((RUN/'config.json').read_text());params=copy.deepcopy(cfg);params['task']['contact_trial']['center_offsets_xy_m']=[[0.,.030]];params['task']['contact_trial']['postlift_yaw']={'world_z_degrees':-30.,'duration_s':6.}
(OUT/'diagnostic_config.json').write_text(json.dumps(params,indent=2)+'\n');np.savez_compressed(OUT/'hybrid_hypothetical_observation.npz',**state);np.savez_compressed(OUT/'rest.npz',**rest)
params['robots']['mjcf_path']=str(SOURCE/'assets/panda_tactile.xml')
for key in ('npz_file','metadata'):params['bag'][key]=str(SOURCE/'assets'/cfg['bag'][key])
report=dict(kind='single_hybrid_minus30_positive_y30_frozen_route_screen',physical_success=False,native_execution=False,not_a_joint_native_snapshot=True,status='initial_screen',source_provenance_path=str(PRIOR/'review.json'),saved_minus30_endpoint_reused_exactly=True,input_sha256=hashes,only_com_offset_m=[0.,.030],within_unchanged_35mm_initial_offset_radius=True,frozen_source=str(SOURCE),frozen_postlift_policy_degrees=15.,production_source_not_imported=True,limitations=['Uses the previously audited minus30 hypothetical endpoint, not a native simultaneous observation.','Frozen probe source is used; current production helper edits are intentionally not mixed into this diagnostic.','No source/config/assets mutation, native execution, particle manipulation or physical-success claim.','Static no-hit checks are only early rejection screens; no continuous cloth separation claim.'])
def save():
 report['inputs_unchanged']=all(sha(Path(p))==h for p,h in hashes.items());(OUT/'review.json').write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')
policy=_policy(params);context=_state_context(rest,state,params,policy);polygon,triangles,mouth=_mouth(rest,state,params);turned=state['bear'];tcp=state['tcp_right'];com=context['com'];anchor=polygon_centroid(polygon);target=anchor+[0.,.030]
inside,distance=projected_inside(np.array([target]),polygon);sections,envelopes=_section_envelopes(turned,rest['bear_tets']);evaluation=_evaluate_offset(envelopes,target-com[:2],polygon,triangles,mouth,policy)
heightmargin=mouth['rim_z_min_m']-policy['endpoint_below_lowest_rim_m']-(cfg['table']['height']+policy['bottom_clearance_m'])-float(np.ptp(turned[:,2]))
report.update(context_rejections=context['reasons'],context_physical=context['physical'],initial_section_screen=dict(target_com_xy_m=target.tolist(),translation_xy_m=(target-com[:2]).tolist(),com_boundary_distance_m=float(distance[0]),com_inside=bool(inside[0]),height_margin_m=heightmargin,evaluation={k:v for k,v in evaluation.items() if k!='sections'}))
rejections=list(context['reasons'])+list(evaluation['section_guard_rejections'])
if heightmargin<0:rejections.append('original height gate failed')
if not inside[0] or distance[0]<policy['com_boundary_clearance_m']:rejections.append('original 2mm COM margin failed')
if rejections:
 report.update(status='rejected_initial_context_or_section',rejections=rejections);save();print('INITIAL_REJECTED',json.dumps(report['initial_section_screen']),flush=True);raise SystemExit(0)
print('INITIAL_SECTION_PASSED',json.dumps(report['initial_section_screen']),flush=True)
asset=dict(np.load(SOURCE/'assets'/cfg['bag']['npz_file']));error,mapping=cKDTree(rest['bag']).query(asset['points']+cfg['bag']['position']);assert error.max()<1e-7 and len(np.unique(mapping))==len(mapping)
bf=tetra_boundary(rest['bear'],rest['bear_tets']);tri=turned[bf];e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0];handles={}
for name in ('front','back'):
 ids=np.flatnonzero(np.isin(asset['faces'],asset['handle_'+name]).all(axis=1));faces=mapping[asset['faces'][ids]];edges,_,_=mesh_edges(faces);assert len(faces)==352;handles[name]=edges
outside=tcp.copy();outside[:2]+=target-com[:2];outside[2]=max(tcp[2],mouth['rim_z_max_m']+policy['planned_outside_clearance_m']-(turned[:,2]-tcp[2]).min())
report['ideal_transport_tcp_m']=outside.tolist();report['early_handle_surface_checks']=[]
for depth in (0.,.005,.025,.050):
 delta=outside-tcp-[0.,0.,depth];row=dict(depth_below_transport_m=depth,translation_from_saved_shape_m=delta.tolist(),handles={})
 for name,edges in handles.items():
  origins=state['bag'][edges[:,0]]-delta;dirs=state['bag'][edges[:,1]]-state['bag'][edges[:,0]];count=0;hits=[]
  for first in range(0,len(edges),64):
   O=origins[first:first+64];D=dirs[first:first+64];H=np.cross(D[:,None,:],e2[None]);det=np.einsum('eji,ji->ej',H,e1);good=np.abs(det)>1e-12;inv=np.divide(1.,det,out=np.zeros_like(det),where=good);S=O[:,None,:]-tri[None,:,0];u=np.einsum('eji,eji->ej',S,H)*inv;Q=np.cross(S,e1[None]);v=np.einsum('ei,eji->ej',D,Q)*inv;t=np.einsum('ji,eji->ej',e2,Q)*inv
   valid=good&(u>1e-8)&(v>1e-8)&(u+v<1-1e-8)&(t>1e-8)&(t<1-1e-8);pairs=np.argwhere(valid);count+=len(pairs)
   for ee,ff in pairs[:max(0,3-len(hits))]:
    edge=edges[first+ee];bary=np.array([1-u[ee,ff]-v[ee,ff],u[ee,ff],v[ee,ff]]);point=state['bag'][edge[0]]+t[ee,ff]*dirs[first+ee];res=float(np.linalg.norm(point-(bary@tri[ff]+delta)));assert res<1e-10
    hits.append(dict(handle_edge=edge.tolist(),bear_surface_face=int(ff),bear_face_vertices=bf[ff].tolist(),edge_fraction=float(t[ee,ff]),bear_barycentric=bary.tolist(),point_world_m=point.tolist(),residual_m=res))
  row['handles'][name]=dict(strict_crossing_count=count,first_witnesses=hits)
 report['early_handle_surface_checks'].append(row)
 if any(h['strict_crossing_count'] for h in row['handles'].values()):
  report.update(status='rejected_early_frozen_handle_crossing',route_assessed=False);save();print('EARLY_HANDLES_REJECTED',json.dumps(row),flush=True);raise SystemExit(0)
report.update(status='static_screen_passed_planning_route',route_assessed=True);save();print('FOUR_STATIC_POSES_NO_KNOWN_CROSSING',flush=True)
t=time.monotonic();choice=plan_guarded_contact_trial(rest,state,params);choice.update(not_a_joint_native_snapshot=True,hybrid_provenance_path=str(PRIOR/'review.json'),diagnostic_review_path=str(OUT/'review.json'));p=OUT/'guarded_candidate.json';p.write_text(json.dumps(_serial(choice),indent=2,allow_nan=False)+'\n')
report.update(status='bounded_screen_complete',route_wall_s=time.monotonic()-t,route_candidate_path=str(p),route_candidate_status=choice['status'],route_rejections=choice['report'].get('rejections',[]),route_attempts=[dict(center=a['center_of_mass_xy_target_m'],rejection=a['rejection'],stages_completed=a['stages_completed']) for a in choice['report'].get('robot_centering_attempts',[])])
save();print('DONE',json.dumps({k:report[k] for k in ('status','route_candidate_status','route_rejections','route_attempts')}),flush=True)
