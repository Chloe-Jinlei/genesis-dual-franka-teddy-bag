"""Actual full29 saved soft shapes; same-time actual-q FK palm geometry, no native replay."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import trimesh,mujoco
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_29'
sys.path.insert(0,str(RUN/'source'))
from motion import Planner
from validate import signed_volumes,tetra_boundary
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((RUN/'config.json').read_text());rest=dict(np.load(RUN/'rest.npz'))
states=dict(np.load(RUN/'states.npz'));failed=dict(np.load(RUN/'failed_soft_state.npz'))
trace=np.load(RUN/'joint_trace.npy');guards=json.loads((RUN/'contact_guard_trace.json').read_text())
coupling=json.loads((RUN/'coupling_trace.json').read_text())
sp=RUN/'source/assets'/cfg['bag']['npz_file'];asset=dict(np.load(sp))
paths=[RUN/n for n in ['config.json','rest.npz','states.npz','failed_soft_state.npz','joint_trace.npy','contact_guard_trace.json','coupling_trace.json','source/motion.py','source/physics_config.py','source/assets/panda_tactile.xml']]+[sp]
before={str(p):sha(p) for p in paths}
distance,mapping=cKDTree(rest['bag']).query(asset['points']+cfg['bag']['position'])
assert distance.max()<1e-7 and len(np.unique(mapping))==len(mapping)
faces=mapping[asset['faces']];parts={k:asset['panel_'+k+'_face_indices'] for k in ['front','back']}
for side in ['front','back']:
    parts[side+'_handle']=np.flatnonzero(np.all(np.isin(asset['faces'],asset['handle_'+side]),axis=1))
assert sum(map(len,parts.values()))==len(faces)
pc=json.loads(json.dumps(cfg));pc['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
planner=Planner(pc);m,d=planner.m,planner.d
bf=tetra_boundary(rest['bear'],rest['bear_tets'])

def matrices(qleft,qright):
    for side,q in [('left',qleft),('right',qright)]:
        d.qpos[planner.qindices[side]]=q[:7]
        for i in (1,2):d.qpos[m.joint(side+'_finger_joint'+str(i)).qposadr[0]]=q[6+i]
    mujoco.mj_forward(m,d)

def body_geometries(bodyname):
    out=[]
    for gid in range(m.ngeom):
        if planner.body_name(gid)!=bodyname or not(m.geom_contype[gid] or m.geom_conaffinity[gid]):continue
        assert m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH
        mid=m.geom_dataid[gid];a,n=m.mesh_vertadr[mid],m.mesh_vertnum[mid];fa,fn=m.mesh_faceadr[mid],m.mesh_facenum[mid]
        vertices=m.mesh_vert[a:a+n].astype(float)@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
        mesh=trimesh.Trimesh(vertices,m.mesh_face[fa:fa+fn],process=False)
        out.append((gid,mesh))
    return out

def query(mesh,vertices,faceids=None):
    if faceids is None:
        pts=vertices;fi=None;samplekind=None
    else:
        tri=vertices[faces[faceids]]
        pts=np.concatenate([tri[:,0],tri[:,1],tri[:,2],(tri[:,0]+tri[:,1])/2,(tri[:,1]+tri[:,2])/2,(tri[:,2]+tri[:,0])/2,tri.mean(1)])
        fi=np.tile(faceids,7);samplekind=np.repeat(np.arange(7),len(faceids))
    near,ds,triid=trimesh.proximity.closest_point(mesh,pts);j=int(ds.argmin())
    out=dict(sampled_minimum_distance_m=float(ds[j]),soft_point_world_m=pts[j].tolist(),rigid_nearest_world_m=near[j].tolist(),
        rigid_triangle_id=int(triid[j]),sample_count=len(pts),gap_direction_rigid_to_soft_world=((pts[j]-near[j])/max(ds[j],1e-15)).tolist())
    if fi is not None:
        f=int(fi[j]);out.update(source_cloth_face_id=f,source_cloth_vertex_ids=asset['faces'][f].tolist(),native_cloth_vertex_ids=faces[f].tolist(),sample_kind=int(samplekind[j]),
          source_material_center_m=asset['points'][asset['faces'][f]].mean(0).tolist(),nearest_cloth_triangle_world_m=vertices[faces[f]].tolist())
    else:out.update(bear_node_id=j,bear_rest_local_m=(rest['bear'][j]-cfg['bear']['position']).tolist())
    return out

rows=[]
for requested in [46.155,46.655,47.005,47.155,float(failed['time'])]:
    if abs(requested-float(failed['time']))<1e-8:
        s=failed;source='failed_soft_state.npz';index=None
    else:
        index=int(np.argmin(abs(states['time']-requested)));s={k:v[index] for k,v in states.items()};source='states.npz'
    t=float(s['time']);ti=int(np.argmin(abs(trace[:,0]-t)));assert abs(trace[ti,0]-t)<1e-8
    qleft=trace[ti,1:10];qright=trace[ti,10:19];matrices(qleft,qright)
    guard=min(guards,key=lambda a:abs(a['time_s']-t));assert abs(guard['time_s']-t)<1e-8
    cup=min(coupling,key=lambda a:abs(a['time']-t));same_cup=abs(cup['time']-t)<1e-8
    native_err=next(x for x in cup['links'] if x['entity']=='right_franka' and x['link']=='link7') if same_cup else None
    measured_tcp=s.get('tcp_right');tcp,R=planner.fk('right',qright[:7])
    row=dict(time_s=t,source=source,state_index=index,same_time_actual_joint_trace=True,actual_saved_TCP_m=None if measured_tcp is None else measured_tcp.tolist(),
        reconstructed_FK_TCP_m=tcp.tolist(),FK_to_saved_TCP_error_m=None if measured_tcp is None else float(np.linalg.norm(tcp-measured_tcp)),
        native_link7_IPC_deviation_from_current_Genesis_pose=native_err,nearest_coupling_time_s=cup['time'],same_time_native_coupling=same_cup,
        actual_minJ=float((signed_volumes(s['bear'],rest['bear_tets'])/signed_volumes(rest['bear'],rest['bear_tets'])).min()),
        distal_wrench=dict(net_force_N=guard['net_force_norm_n'],TCP_torque_Nm=guard['tcp_torque_norm_nm'],per_link_names=guard['right_link_names'],
            forces_world_N=guard['right_link_force_world_n'],torques_at_TCP_world_Nm=guard['right_link_torque_at_aim_tcp_world_nm']),geometries={})
    for body in ['right_link7','right_hand']:
        for gid,mesh in body_geometries(body):
            local_radius=np.linalg.norm(mesh.vertices-d.xpos[m.body('right_link7').id],axis=1).max()
            posebound=None if native_err is None else native_err['translation_error_m']+(2*np.sin(native_err['rotation_error_rad']/2)+native_err['affine_singular_value_deviation'])*local_radius
            near={part:query(mesh,s['bag'],idx) for part,idx in parts.items()};near['bear']=query(mesh,s['bear'])
            row['geometries'][body]=dict(geom_id=int(gid),mesh_vertex_count=len(mesh.vertices),mesh_face_count=len(mesh.faces),mesh_bounds_m=mesh.bounds.tolist(),
                maximum_radius_from_link7_m=float(local_radius),native_pose_point_displacement_upper_bound_m=posebound,
                source_surface_distances=near)
    rows.append(row)
    print(json.dumps(dict(time=t,torque=guard['tcp_torque_norm_nm'],FKerror=row['FK_to_saved_TCP_error_m'],
        contacts={b:{n:x['sampled_minimum_distance_m'] for n,x in a['source_surface_distances'].items()} for b,a in row['geometries'].items()})),flush=True)

out=dict(kind='independent_full29_link7_palm_contact_geometry',physical_task_success=False,native_execution_performed=False,
    source_sha256=before,inputs_unchanged=all(sha(p)==h for p,h in before.items()),rows=rows,
    source_geometry_contract=dict(hand_fixed_transform_from_link7=dict(position_m=[0,0,.107],quaternion_wxyz=[.9238795,0,0,-.3826834]),
        right_TCP_hand_local_m=[0,0,.1029],native_IPC_link7_includes_fixed_hand=True,
        pose_source='Frozen MuJoCo model, exact simultaneously saved actual joint values including both fingers; transformed original mesh vertices/faces, no whole-hand convex replacement.',
        runtime_binding='Genesis ipc_coupler/coupler.py lines403-489 merges fixed child hand collision geometry into link7 using the child transform.'),
    limitations=[
        'Native link7 IPC matrices are not saved. Same-time complete frames contain only translation/rotation/affine deviation magnitudes, so mesh distances use FK and report a pose-error bound. At true failed soft frame even this same-time deviation is unavailable; previous frame is15ms earlier.',
        'The frozen original collision mesh surface is used. Native imported rigid collision processing may convexify or decimate meshes; this report does not claim exact native IPC surface reconstruction.',
        'Distances query cloth triangle vertices, all edge midpoints and centroids and all bear vertices. Positive samples do not prove whole-surface separation, continuous clearance, or absence of narrow contact.',
        'Robot-link aggregate native wrench does not identify the colliding material pair. Geometric coincidence with a torque trend is evidence, not pairwise force attribution.',
        'No alternate trajectory, offset, release action, material, or protection threshold is changed or certified.'])
p=RUN/'independent-full29-palm-contact-geometry.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print('SAVED',p,sha(p),flush=True)
