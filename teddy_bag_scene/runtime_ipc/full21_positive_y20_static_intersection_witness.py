"""Four declared frozen poses; strict segment/triangle crossing witnesses."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_21';OUT=RUN/'independent-positive-y20-route-review'
sys.path.insert(0,str(RUN/'source'))
from validate import tetra_boundary,mesh_edges
from postlift_yaw import _serial
cfg=json.loads((RUN/'config.json').read_text());rest=dict(np.load(RUN/'rest.npz'));state=dict(np.load(RUN/'orient_teddy_postlift.npz'))
asset=dict(np.load(RUN/'source/assets'/cfg['bag']['npz_file']))
error,mapping=cKDTree(rest['bag']).query(asset['points']+cfg['bag']['position']);assert error.max()<1e-7
front_ids=np.flatnonzero(np.isin(asset['faces'],asset['handle_front']).all(axis=1));frontfaces=mapping[asset['faces'][front_ids]]
edges,_,_=mesh_edges(frontfaces);bf=tetra_boundary(rest['bear'],rest['bear_tets']);tri=state['bear'][bf]
e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
rows=[]
for name,depth,file in [('zero',0.,'guarded_contact_candidate.json'),('positive_y20',0.,'independent-positive-y20-original-base.json'),('zero',.005,'guarded_contact_candidate.json'),('positive_y20',.025,'independent-positive-y20-original-base.json')]:
    candidate=json.loads((RUN/file).read_text());delta=np.asarray(candidate['route']['outside_tcp_m'])-state['tcp_right']-[0.,0.,depth]
    origins=state['bag'][edges[:,0]]-delta;directions=state['bag'][edges[:,1]]-state['bag'][edges[:,0]]
    hits=[]
    for first in range(0,len(edges),64):
        origin=origins[first:first+64];direction=directions[first:first+64]
        h=np.cross(direction[:,None,:],e2[None]);det=np.einsum('eji,ji->ej',h,e1)
        good=np.abs(det)>1e-12;inverse=np.divide(1.,det,out=np.zeros_like(det),where=good)
        ss=origin[:,None,:]-tri[None,:,0];u=np.einsum('eji,eji->ej',ss,h)*inverse
        q=np.cross(ss,e1[None]);v=np.einsum('ei,eji->ej',direction,q)*inverse
        t=np.einsum('ji,eji->ej',e2,q)*inverse
        valid=good&(u>1e-8)&(v>1e-8)&(u+v<1.-1e-8)&(t>1e-8)&(t<1.-1e-8)
        for ee,ff in np.argwhere(valid):
            edge_index=first+int(ee);face=int(ff);fraction=float(t[ee,ff]);bary=np.array([1-u[ee,ff]-v[ee,ff],u[ee,ff],v[ee,ff]])
            point_edge=state['bag'][edges[edge_index,0]]+fraction*directions[edge_index]
            point_tri=bary@tri[face]+delta;residual=float(np.linalg.norm(point_edge-point_tri))
            assert residual<1e-10
            hits.append(dict(frontstrap_material_edge=edges[edge_index],bear_surface_face_index=face,bear_surface_face_vertices=bf[face],
                segment_fraction=fraction,triangle_barycentric=bary,determinant=float(det[ee,ff]),world_point=point_edge,
                reconstruction_residual_m=residual))
    rows.append(dict(case=name,descent_depth_m=depth,proper_segment_triangle_crossings=len(hits),first_witnesses=hits[:12],
        no_crossing_result_is_not_general_disjointness_certificate=True))
result=dict(kind='frozen_postyaw_frontstrap_edge_vs_bear_surface_strict_crossing_witnesses',physical_success=False,native_executed=False,
    source_state=str(RUN/'orient_teddy_postlift.npz'),rows=rows,
    limits=['Strict interior segment and triangle crossings at the specified static translated poses are sufficient frozen surface-intersection witnesses.',
        'This is not native cloth penetration or force evidence; real shapes deform and the cloth can move.',
        'Only frontstrap edges through bear boundary faces are tested; no-hit does not prove triangle meshes disjoint. No extra offsets or heights were searched.'])
path=OUT/'frontstrap-static-intersection-witnesses.json';path.write_text(json.dumps(_serial(result),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=[{k:v for k,v in r.items() if k!='first_witnesses'} for r in rows]),indent=2))
