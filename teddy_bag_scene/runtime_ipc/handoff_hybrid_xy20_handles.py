"""Bounded actual-bag/frozen-bear handle surface screen for the sole accepted hybrid route."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np,trimesh
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_handoff_upright_offset_01';OUT=RUN/'independent-hybrid-postlift-route';sys.path.insert(0,str(RUN/'source'))
from validate import tetra_boundary,mesh_edges
from contact_insertion import _serial
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
state=dict(np.load(OUT/'hybrid_hypothetical_observation.npz'));rest=dict(np.load(OUT/'rest.npz'));cfg=json.loads((OUT/'config.json').read_text());candidate=json.loads((OUT/'positive_xy20_guarded_candidate.json').read_text());assert candidate['status']=='candidate'
assetpath=RUN/'source/assets'/cfg['bag']['npz_file'];asset=dict(np.load(assetpath));error,mapping=cKDTree(rest['bag']).query(asset['points']+cfg['bag']['position']);assert error.max()<1e-7 and len(np.unique(mapping))==len(mapping)
bf=tetra_boundary(rest['bear'],rest['bear_tets']);bearmesh=trimesh.Trimesh(state['bear'],bf,process=False)
def samples(points,faces):
 edges,_,_=mesh_edges(faces);return np.vstack([points[np.unique(faces)],points[edges].mean(1),points[faces].mean(1)])
bearsamples=samples(state['bear'],bf);tri=state['bear'][bf];e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
handles={}
for name in ('front','back'):
 ids=np.flatnonzero(np.isin(asset['faces'],asset['handle_'+name]).all(axis=1));faces=mapping[asset['faces'][ids]];edges,_,_=mesh_edges(faces)
 assert len(faces)==352
 handles[name]=dict(original_faces=ids,faces=faces,edges=edges,mesh=trimesh.Trimesh(state['bag'],faces,process=False),samples=samples(state['bag'],faces))

def crossings(delta,h):
 origins=state['bag'][h['edges'][:,0]]-delta;directions=state['bag'][h['edges'][:,1]]-state['bag'][h['edges'][:,0]];hits=[];count=0
 for first in range(0,len(origins),64):
  origin=origins[first:first+64];direction=directions[first:first+64]
  H=np.cross(direction[:,None,:],e2[None]);det=np.einsum('eji,ji->ej',H,e1);good=np.abs(det)>1e-12;inverse=np.divide(1.,det,out=np.zeros_like(det),where=good)
  S=origin[:,None,:]-tri[None,:,0];u=np.einsum('eji,eji->ej',S,H)*inverse;Q=np.cross(S,e1[None]);v=np.einsum('ei,eji->ej',direction,Q)*inverse;t=np.einsum('ji,eji->ej',e2,Q)*inverse
  valid=good&(u>1e-8)&(v>1e-8)&(u+v<1.-1e-8)&(t>1e-8)&(t<1.-1e-8);indices=np.argwhere(valid);count+=len(indices)
  for ee,ff in indices[:max(0,3-len(hits))]:
   edge=h['edges'][first+ee];bary=np.array([1-u[ee,ff]-v[ee,ff],u[ee,ff],v[ee,ff]]);point=state['bag'][edge[0]]+t[ee,ff]*directions[first+ee];res=float(np.linalg.norm(point-(bary@tri[ff]+delta)));assert res<1e-10
   hits.append(dict(handle_material_edge=edge,whole_bear_surface_face=int(ff),whole_bear_face_vertices=bf[ff],segment_fraction=float(t[ee,ff]),bear_triangle_barycentric=bary,world_point=point,reconstruction_residual_m=res))
 return dict(proper_edge_triangle_crossing_count=count,first_strict_crossing_witnesses=hits)

def distances(delta,h):
 pb,db,ib=trimesh.proximity.closest_point(bearmesh,h['samples']-delta)
 ph,dh,ih=trimesh.proximity.closest_point(h['mesh'],bearsamples+delta)
 i=int(np.argmin(db));j=int(np.argmin(dh));return dict(sampled_minimum_distance_m=min(float(db[i]),float(dh[j])),handle_samples_to_bear_minimum_m=float(db[i]),bear_samples_to_handle_minimum_m=float(dh[j]),nearest_handle_sample_world_m=h['samples'][i],nearest_bear_triangle_point_world_m=pb[i]+delta,nearest_bear_surface_face_index=int(ib[i]),nearest_bear_sample_world_m=bearsamples[j]+delta,nearest_handle_triangle_point_world_m=ph[j],nearest_original_handle_face_index=int(h['original_faces'][ih[j]]))
inputpaths=[OUT/'hybrid_hypothetical_observation.npz',OUT/'rest.npz',OUT/'positive_xy20_guarded_candidate.json',assetpath,Path(__file__)];hashes={str(p):sha(p) for p in inputpaths}
report=dict(kind='single_hybrid_plus20_plus20_two_handles_frozen_surface_screen',physical_success=False,native_execution=False,not_a_joint_native_snapshot=True,provenance_path=str(OUT/'provenance.json'),candidate_path=str(OUT/'positive_xy20_guarded_candidate.json'),input_sha256=hashes,scope=['Both the retained front strap and released back strap use all352 original physical triangles. Whole1450-node/5486-tet teddy boundary is used; no patch omitted.','Frozen measured bear is translated with full postyaw orientation unchanged; actual probe final bag remains fixed. No physical object or production file is modified.','Strict interior handle-edge through bear-face hits prove surface crossing only in the explicitly hypothetical static pose. They do not prove native penetration or force.','No-hit is not a surface-disjointness certificate: coplanar/tangent cases and reverse-edge configurations are not exhaustively certified.','Finite point-triangle distances using vertices,edge midpoints,face centroids in both directions are upper bounds on exact surface distance. Positive values do not certify clearance.','Only bounded discrete declared route fractions are tested, not continuous handle clearance. Body contact/real grasp/native deformations need original physical gates.'],rows=[])
out=OUT/'positive_xy20_two_handles_surface_review.json'
def save():
 report['inputs_unchanged']=all(sha(Path(p))==v for p,v in hashes.items());out.write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')
start=state['tcp_right'].copy();started=time.monotonic()
for stage in candidate['route']['stages']:
 target=np.array(stage['right_tcp_m']);name=stage['name'];length=float(np.linalg.norm(target-start))
 if length<1e-12:fractions=np.array([0.])
 elif name=='insert_candidate':
  depth=float(start[2]-target[2]);depths=np.unique(np.r_[np.arange(0.,min(depth,.10)+1e-9,.005),np.linspace(min(depth,.10),depth,13)])
  fractions=depths/depth
 else:fractions=np.linspace(0.,1.,11)
 for f in fractions:
  tcp=(1-f)*start+f*target;delta=tcp-state['tcp_right'];row=dict(stage=name,segment_fraction=float(f),right_tcp_m=tcp,translation_m=delta,depth_below_transport_m=float(start[2]-tcp[2]) if name=='insert_candidate' else None,handles={})
  for label,h in handles.items():row['handles'][label]={**distances(delta,h),**crossings(delta,h)}
  report['rows'].append(row);save()
  print(name,round(float(f),5),[(k,round(v['sampled_minimum_distance_m']*1000,4),v['proper_edge_triangle_crossing_count']) for k,v in row['handles'].items()],flush=True)
 start=target
report['summary']={}
for label in handles:
 rows=report['rows'];hits=[r for r in rows if r['handles'][label]['proper_edge_triangle_crossing_count']]
 nearest=min(rows,key=lambda r:r['handles'][label]['sampled_minimum_distance_m'])
 report['summary'][label]=dict(minimum_sampled_distance_m=nearest['handles'][label]['sampled_minimum_distance_m'],minimum_at_stage=nearest['stage'],minimum_at_fraction=nearest['segment_fraction'],number_of_static_poses_with_strict_crossings=len(hits),first_crossing_stage=hits[0]['stage'] if hits else None,first_crossing_fraction=hits[0]['segment_fraction'] if hits else None,first_crossing_depth_m=hits[0]['depth_below_transport_m'] if hits else None)
report['elapsed_s']=time.monotonic()-started;report['status']='complete_bounded_sampled_surface_screen';save();print('DONE',out,sha(out),json.dumps(_serial(report['summary'])),flush=True)
