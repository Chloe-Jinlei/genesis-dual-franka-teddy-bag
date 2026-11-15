"""One authorized offset: all saved descent frames, unchanged soft shapes, independent IK poses."""
from pathlib import Path
import sys,json,hashlib,xml.etree.ElementTree as ET
import numpy as np,trimesh,mujoco
from scipy.spatial import cKDTree,ConvexHull
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_29';sys.path.insert(0,str(RUN/'source'))
from motion import Planner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((RUN/'config.json').read_text());rest=dict(np.load(RUN/'rest.npz'));ss=dict(np.load(RUN/'states.npz'));failed=dict(np.load(RUN/'failed_soft_state.npz'))
trace=np.load(RUN/'joint_trace.npy');events=json.loads((RUN/'events.json').read_text());cp=json.loads((RUN/'coupling_trace.json').read_text())
sourcefile=RUN/'source/assets'/cfg['bag']['npz_file'];asset=dict(np.load(sourcefile));err,mp=cKDTree(rest['bag']).query(asset['points']+cfg['bag']['position']);assert err.max()<1e-7
heldfaces=asset['faces'][np.all(np.isin(asset['faces'],asset['handle_front']),axis=1)];faceids=np.flatnonzero(np.all(np.isin(asset['faces'],asset['handle_front']),axis=1));ff=mp[heldfaces]
sourceverts=np.unique(ff);ee=np.unique(np.sort(np.concatenate([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]]),axis=1),axis=0)
pc=json.loads(json.dumps(cfg));pc['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml');p=Planner(pc);m,d=p.m,p.d
gid=next(i for i in range(m.ngeom) if p.body_name(i)=='right_hand' and m.geom_contype[i]);mid=m.geom_dataid[gid];va,vn=m.mesh_vertadr[mid],m.mesh_vertnum[mid];fa,fn=m.mesh_faceadr[mid],m.mesh_facenum[mid]
raw=trimesh.Trimesh(m.mesh_vert[va:va+vn].astype(float),m.mesh_face[fa:fa+fn],process=False);h=raw.convex_hull;assert len(h.faces)<=500
he=h.edges_unique;delta=np.array([-.015,0,0]);eps=1e-10
paths=[RUN/n for n in ['config.json','rest.npz','states.npz','failed_soft_state.npz','joint_trace.npy','coupling_trace.json','events.json','source/motion.py','source/assets/panda_tactile.xml']]+[sourcefile]
for mesh in ET.parse(paths[-2] if paths[-2].suffix=='.xml' else RUN/'source/assets/panda_tactile.xml').getroot().findall('./asset/mesh'):
    if mesh.get('name')=='hand_c':paths.append(Path(mesh.get('file')))
hashes={str(x):sha(x) for x in paths}
start=next(e['start'] for e in events if e['phase']=='insert_teddy');ix=np.flatnonzero(ss['time']>=start-1e-8)
observations=[({k:v[i] for k,v in ss.items()},int(i),'states.npz') for i in ix]
observations.append((failed,None,'failed_soft_state.npz'))

def clip(poly,planes,pad=0.):
    poly=np.asarray(poly,float)
    for pl in planes:
        if not len(poly):break
        q=[];last=poly[-1];dl=last@pl[:3]+pl[3]-pad
        for cur in poly:
            dc=cur@pl[:3]+pl[3]-pad
            if (dc<=0)!=(dl<=0):q.append(last+(cur-last)*(dl/(dl-dc)))
            if dc<=0:q.append(cur)
            last=cur;dl=dc
        if q:
            poly=np.array(q);keep=np.r_[True,np.linalg.norm(np.diff(poly,axis=0),axis=1)>1e-13];poly=poly[keep]
            if len(poly)>1 and np.linalg.norm(poly[-1]-poly[0])<=1e-13:poly=poly[:-1]
        else:poly=np.empty((0,3))
    return poly

def segment_candidates(A,B,C,D):
    u=B-A;v=D-C;w=A-C;a=np.einsum('ij,ij->i',u,u);b=np.einsum('ij,ij->i',u,v);c=np.einsum('ij,ij->i',v,v);dd=np.einsum('ij,ij->i',u,w);e=np.einsum('ij,ij->i',v,w)
    assert np.all(a>0) and np.all(c>0), 'Degenerate segment needs explicit treatment'
    uv=np.cross(u,v);den=np.einsum('ij,ij->i',uv,uv);sol=[]
    for s,t in [(np.zeros(len(A)),np.clip(e/c,0,1)),(np.ones(len(A)),np.clip((e+b)/c,0,1)),(np.clip(-dd/a,0,1),np.zeros(len(A))),(np.clip((b-dd)/a,0,1),np.ones(len(A)))]:
        pa=A+s[:,None]*u;pb=C+t[:,None]*v;sol.append((np.sum((pa-pb)**2,axis=1),pa,pb))
    good=den>0;rhs=C-A;s=np.divide(np.einsum('ij,ij->i',np.cross(rhs,v),uv),den,out=np.zeros_like(a),where=good);t=np.divide(np.einsum('ij,ij->i',np.cross(rhs,u),uv),den,out=np.zeros_like(a),where=good);good&=(s>=0)&(s<=1)&(t>=0)&(t<=1)
    pa=A+s[:,None]*u;pb=C+t[:,None]*v;d2=np.sum((pa-pb)**2,axis=1);d2[~good]=np.inf;sol.append((d2,pa,pb))
    values=np.stack([s[0] for s in sol]);which,j=np.unravel_index(values.argmin(),values.shape);return float(np.sqrt(values[which,j])),sol[which][1][j],sol[which][2][j],int(j)

def exact_frozen_gap(hv,bag):
    mesh=trimesh.Trimesh(hv,h.faces,process=False);cloth=trimesh.Trimesh(bag,ff,process=False);tri=bag[ff];planes=ConvexHull(hv).equations
    gap=np.maximum(0,np.maximum(tri.min(1)-hv.max(0),hv.min(0)-tri.max(1)));possible=np.flatnonzero(np.linalg.norm(gap,axis=1)<=eps)
    uncertain=[]
    for f in possible:
        poly=clip(tri[f],planes)
        if len(poly):
            point=poly.mean(0);depth=float(-(planes[:,:3]@point+planes[:,3]).max())
            return dict(solid_triangle_gap_m=0.,intersection=True,strict_interior_witness=depth>eps,interior_plane_slack_m=depth,source_cloth_face_id=int(faceids[f]),witness_world_m=point.tolist(),numerical_near_tangent=depth<=eps)
        if len(clip(tri[f],planes,eps)):uncertain.append(int(faceids[f]))
    # For disjoint polyhedral surfaces, vertex/face and edge/edge cover all closest-feature pairs.
    q,ds,fi=trimesh.proximity.closest_point(mesh,bag[sourceverts]);j=int(ds.argmin());best=float(ds[j]);pa=q[j];pb=bag[sourceverts[j]];kind='cloth_vertex_to_hand_face';extra=dict(cloth_vertex_id=int(sourceverts[j]),hand_face_id=int(fi[j]))
    q,ds,fi=trimesh.proximity.closest_point(cloth,hv);j=int(ds.argmin())
    if ds[j]<best:best=float(ds[j]);pa=hv[j];pb=q[j];kind='hand_vertex_to_cloth_face';extra=dict(hand_vertex_id=j,source_cloth_face_id=int(faceids[fi[j]]))
    ha,hb=hv[he[:,0]],hv[he[:,1]];ca,cb=bag[ee[:,0]],bag[ee[:,1]]
    sep=np.maximum(0,np.maximum(np.minimum(ca,cb)[None]-np.maximum(ha,hb)[:,None],np.minimum(ha,hb)[:,None]-np.maximum(ca,cb)[None]))
    ai,ci=np.where(np.sum(sep*sep,axis=2)<=(best+eps)**2)
    if len(ai):
        sd,x,y,j=segment_candidates(ha[ai],hb[ai],ca[ci],cb[ci])
        if sd<best:best=sd;pa=x;pb=y;kind='hand_edge_to_cloth_edge';extra=dict(hand_edge_ids=he[ai[j]].tolist(),cloth_edge_native_ids=ee[ci[j]].tolist())
    return dict(solid_triangle_gap_m=best,intersection=False,strict_interior_witness=False,closest_feature_pair=kind,hand_point_world_m=pa.tolist(),cloth_point_world_m=pb.tolist(),
        gap_direction_hand_to_cloth_world=((pb-pa)/max(best,1e-20)).tolist(),closest_feature=extra,numerical_near_tangent=bool(uncertain),uncertain_face_ids=uncertain,
        algorithm='All vertices versus opposite complete surface, all AABB-eligible edge pairs, and convex-solid/triangle clipping. Numerical doubles, not directed-rounding interval arithmetic.')

rows=[]
for number,(s,index,name) in enumerate(observations):
    t=float(s['time']);ti=int(np.argmin(abs(trace[:,0]-t)));assert abs(trace[ti,0]-t)<1e-8
    qr=trace[ti,10:19];ql=trace[ti,1:10]
    for side,q in [('left',ql),('right',qr)]:
        d.qpos[p.qindices[side]]=q[:7]
        for j in(1,2):d.qpos[m.joint(side+'_finger_joint'+str(j)).qposadr[0]]=q[6+j]
    mujoco.mj_forward(m,d);hv=h.vertices@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
    actual=exact_frozen_gap(hv,s['bag']);tcp,R=p.fk('right',qr[:7]);target=tcp+delta
    qnew,pe,re=p.solve('right',target,R,qr[:7]);mujoco.mj_forward(m,d)
    newhv=h.vertices@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid];proposed=exact_frozen_gap(newhv,s['bag'])
    error=float(np.linalg.norm(newhv-hv-delta,axis=1).max())
    row=dict(time_s=t,source=name,state_index=index,actual_TCP_m=tcp.tolist(),candidate_target_TCP_m=target.tolist(),actual_full_R_xyzw=R.as_quat().tolist(),
        candidate_IK_q_rad=qnew.tolist(),candidate_IK_position_error_m=pe,candidate_IK_rotation_error_rad=re,
        candidate_hand_max_deviation_from_exact15mm_translation_m=error,original_frozen_gap=actual,candidate_frozen_gap=proposed)
    rows.append(row)
    if number%20==0 or number==len(observations)-1:print(json.dumps(dict(frame=number,total=len(observations),t=t,original=actual['solid_triangle_gap_m'],candidate=proposed['solid_triangle_gap_m'],hit=proposed['intersection'],ik=pe)),flush=True)

minrow=min(rows,key=lambda r:r['candidate_frozen_gap']['solid_triangle_gap_m']);oldmin=min(rows,key=lambda r:r['original_frozen_gap']['solid_triangle_gap_m'])
out=dict(kind='independent_full29_sole_minusX15mm_palm_heldfront_saved_frame_diagnostic',physical_task_success=False,native_execution=False,soft_geometry_modified=False,
    supersedes_numerical_method_report_sha256='43bfdac1ca1d0c7d400f0bceb355a0520cd77cf2271941ef1e7e2b1b4e0dd47a',
    edge_distance_numerical_method='Cross-product denominator avoids near-parallel dot-product cancellation; all four boundary projections retained.',
    original_COM_offset_m=[.005,.03075],candidate_COM_offset_m=[-.01,.03075],declared_TCP_translation_m=delta.tolist(),rotation_policy='Preserve each actual frame complete world rotation, not a reconstructed yaw-only pose.',
    saved_native_insert_start_s=start,true_failed_soft_time_s=float(failed['time']),frame_count=len(rows),input_sha256=hashes,inputs_unchanged=all(sha(k)==v for k,v in hashes.items()),
    hand_mesh=dict(vertices=len(h.vertices),faces=len(h.faces),convexified=True,decimated=False),held_front_triangles=len(ff),row_results=rows,
    summary=dict(original_minimum_gap_m=oldmin['original_frozen_gap']['solid_triangle_gap_m'],original_minimum_time_s=oldmin['time_s'],
        candidate_minimum_gap_m=minrow['candidate_frozen_gap']['solid_triangle_gap_m'],candidate_minimum_time_s=minrow['time_s'],
        original_intersection_frame_count=sum(r['original_frozen_gap']['intersection'] for r in rows),candidate_intersection_frame_count=sum(r['candidate_frozen_gap']['intersection'] for r in rows),
        candidate_strict_intersection_frame_count=sum(r['candidate_frozen_gap']['strict_interior_witness'] for r in rows),
        numerical_ambiguous_frame_count=sum(r['candidate_frozen_gap']['numerical_near_tangent'] for r in rows),
        candidate_maximum_IK_position_error_m=max(r['candidate_IK_position_error_m'] for r in rows),candidate_maximum_IK_rotation_error_rad=max(r['candidate_IK_rotation_error_rad'] for r in rows)),
    limitations=['Each actual saved frame keeps the original cloth and bear; the candidate robot poses are counterfactual and are not a simultaneous new native state.',
      'No guarantee between saved frames, no proposed continuous joint command, and no full robot8mm/teddy5mm route clearance is claimed.',
      'The right hand collision mesh follows the configured convexify/no-decimation path, but it is placed by actual-joint FK, not an unavailable link7 IPC affine matrix.',
      'Other contacts and force balance change in a real altered path; already deformed old cloth is not the cloth that the new path will generate.',
      'Solid-triangle gap is zero for boundary crossing or full cloth-triangle containment inside the hand. Positive gap is a floating-point geometric result, not a physical success or interval-certified clearance.'])
dest=RUN/'independent-minus-x15-palm-held-front-review-v2.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(path=str(dest),sha256=sha(dest),summary=out['summary'])),flush=True)
