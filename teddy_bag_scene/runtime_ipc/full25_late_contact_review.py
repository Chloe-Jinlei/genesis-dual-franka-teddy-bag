"""Five actual saved states only: material/body/handle proximity and pose. CPU only."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import trimesh
import mujoco,itertools
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1];P=B/'runs/two_panel_full_25';D=P/'independent-full25-late-contact-review'
sys.path.insert(0,str(P/'source'));from validate import tetra_boundary,signed_volumes
from motion import Planner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ser(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x))
def main():
 D.mkdir(exist_ok=True);files=[P/n for n in['states.npz','failed_soft_state.npz','rest.npz','config.json','transport_teddy.npz','contact_guard_trace.json']]+[Path(__file__).resolve()]
 ss=dict(np.load(files[0]));failed=dict(np.load(files[1]));rest=dict(np.load(files[2]));cfg=json.loads(files[3].read_text());ref=dict(np.load(files[4]));trace=json.loads(files[5].read_text());srcfile=P/'source/assets'/cfg['bag']['npz_file'];files += [srcfile,P/'joint_trace.npy',P/'source/motion.py',P/'source/assets/panda_tactile.xml'];src=dict(np.load(srcfile));hashes={str(p.relative_to(B)):sha(p)for p in files}
 e,mp=cKDTree(rest['bag']).query(src['points']+cfg['bag']['position']);assert e.max()<1e-6 and len(np.unique(mp))==len(mp)
 tets=rest['bear_tets'];bf=tetra_boundary(rest['bear'],tets);local=(rest['bear']-cfg['bear']['position'])/.85;vol=signed_volumes(rest['bear'],tets);weight=np.abs(vol);mass=weight.sum()*cfg['bear']['rho'];tt=np.array([r['time_s']for r in trace]);gref=trace[np.argmin(abs(tt-float(ref['time'])))];X0=ref['bear']-ref['bear'].mean(0)
 joint=np.load(P/'joint_trace.npy');assert abs(joint[-1,0]-float(failed['time']))<1e-9
 cfgp=json.loads(json.dumps(cfg));cfgp['robots']['mjcf_path']=str(P/'source/assets/panda_tactile.xml');planner=Planner(cfgp);bad=int(np.argmin(signed_volumes(failed['bear'],tets)/vol));badids=tets[bad]
 def jaw_reference(q,points):
  tcp,R=planner.fk('right',q[:7])
  for finger in[1,2]:planner.d.qpos[planner.m.joint(f'right_finger_joint{finger}').qposadr[0]]=q[6+finger]
  mujoco.mj_forward(planner.m,planner.d);out={}
  for name in['right_hand','right_left_finger','right_right_finger']:
   rows=[]
   for gid in planner.geoms['right']:
    if planner.body_name(gid)!=name:continue
    kind=planner.m.geom_type[gid]
    if kind==mujoco.mjtGeom.mjGEOM_BOX:lv=np.array(list(itertools.product([-1.,1.],repeat=3)))*planner.m.geom_size[gid]
    elif kind==mujoco.mjtGeom.mjGEOM_MESH:
     mi=planner.m.geom_dataid[gid];start=planner.m.mesh_vertadr[mi];count=planner.m.mesh_vertnum[mi];lv=planner.m.mesh_vert[start:start+count].astype(float)
    else:raise ValueError('Unsupported hand collision kind')
    vv=lv@planner.d.geom_xmat[gid].reshape(3,3).T+planner.d.geom_xpos[gid];mesh=trimesh.convex.convex_hull(vv);near,dd,ii=trimesh.proximity.closest_point(mesh,points);k=int(dd.argmin());rows.append(dict(geom_id=int(gid),geom_type='box'if kind==mujoco.mjtGeom.mjGEOM_BOX else'mesh_convex',sample_min_distance_m=float(dd[k]),query_index=k,query_point_m=points[k],nearest_point_m=near[k]))
   out[name]=min(rows,key=lambda r:r['sample_min_distance_m'])
  return dict(scope='Same-time recorded actual q with frozen MuJoCo collision/FK model; not late native IPC affine geometry',tcp_m=tcp,rotation_xyzw=R.as_quat(),badcell_vertices_and_centroid_hand_local_m=(points-tcp)@R.as_matrix(),individual_finger_q_m=q[7:],nearest_components=out)
 bary7=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[.5,0,.5],[0,.5,.5],[1/3,1/3,1/3]])
 groups={};arcs={};striprow={};bodyset=set(src['body_face_indices'].tolist())
 for name in['front','back']:
  ids=np.array([i for i,f in enumerate(src['faces'])if i not in bodyset and np.isin(f,src['handle_'+name]).all()]);assert len(ids)==352;groups[name]=ids
  nc=len(src['handle_'+name+'_root_left']);inner=np.setdiff1d(src['handle_'+name],src['body']);strip=np.vstack([src['handle_'+name+'_root_left'],inner.reshape(-1,nc),src['handle_'+name+'_root_right'][::-1]])
  expected=set()
  for j in range(len(strip)-1):
   for i in range(nc-1):
    a,b,c,d=strip[j,i],strip[j,i+1],strip[j+1,i],strip[j+1,i+1];expected.update([tuple(sorted([a,b,d])),tuple(sorted([a,d,c]))])
  assert expected==set(tuple(sorted(f))for f in src['faces'][ids])
  centers=src['points'][strip].mean(1);arcs[name]=np.r_[0,np.cumsum(np.linalg.norm(np.diff(centers,axis=0),axis=1))];striprow[name]={int(v):j for j,row in enumerate(strip)for v in row}
 shapes=[('torso',(0,-.025,.045),(.065,.075,.040)),('head',(0,.080,.051),(.058,.055,.047)),('left_ear',(-.047,.126,.057),(.023,.023,.022)),('right_ear',(.047,.126,.057),(.023,.023,.022)),('left_arm',(-.076,-.010,.040),(.040,.035,.028)),('right_arm',(.076,-.010,.040),(.040,.035,.028)),('left_foot',(-.040,-.102,.034),(.036,.041,.030)),('right_foot',(.040,-.102,.034),(.036,.041,.030)),('muzzle',(0,.068,.091),(.031,.024,.017)),('nose',(0,.083,.106),(.012,.008,.006))]
 def anatomy(p):return min(shapes,key=lambda v:(np.linalg.norm((p-np.array(v[1]))/np.array(v[2]))-1)*min(v[2]))[0]
 def observe(s,label,requested):
  bear=s['bear'];bag=s['bag'];t=float(s['time']);bm=trimesh.Trimesh(bear,bf,process=False);com=np.average(bear[tets].mean(1),weights=weight,axis=0);u,sv,vt=np.linalg.svd(X0.T@(bear-bear.mean(0)));R=u@vt
  if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@vt
  fit=X0@R;res=np.linalg.norm(bear-bear.mean(0)-fit,axis=1);g=trace[np.argmin(abs(tt-t))];rim=bag[mp[src['rim']]]
  badverts=bear[badids];lo=badverts.min(0);hi=badverts.max(0);ff=bag[mp[src['faces']]];gg=np.maximum(np.maximum(lo-ff.max(1),ff.min(1)-hi),0.);badclothlower=float(np.linalg.norm(gg,axis=1).min())-1e-7
  jaw=jaw_reference(s['q_right'] if 'q_right'in s else joint[-1,10:19],np.vstack([badverts,badverts.mean(0)]))
  row=dict(label=label,requested_time_s=requested,time_s=t,native_guard_time_s=g['time_s'],same_time_native_guard=bool(abs(g['time_s']-t)<1e-8),bear_COM_m=com,bear_min_m=bear.min(0),bear_max_m=bear.max(0),bear_bestfit_rotation_deg=float(np.degrees(Rotation.from_matrix(R.T).magnitude())),bear_bestfit_rotvec_deg=np.degrees(Rotation.from_matrix(R.T).as_rotvec()),bear_nonrigid_fit_RMS_m=float(np.sqrt(np.mean(res**2))),rim_min_z_m=float(rim[:,2].min()),rim_max_z_m=float(rim[:,2].max()),rim_mean_m=rim.mean(0),rim_mean_displacement_m=(bag-ref['bag'])[mp[src['rim']]].mean(0),body_mean_displacement_m=(bag-ref['bag'])[mp[src['body']]].mean(0),guard_aim_tcp_m=g['aim_tcp_world_m'],measured_tcp_m=s.get('tcp_right'),right_opening_projection_N=g['right_finger_opening_projection_n'],right_finger_forces_N=g['right_finger_force_norm_n'],right_net_robot_force_N=g['force_world_n'],left_net_robot_force_N=g['left_all_link_net_force_world_n'],left_net_robot_force_change_since_transport_N=np.array(g['left_all_link_net_force_world_n'])-gref['left_all_link_net_force_world_n'],right_distal_torque_Nm=g['tcp_torque_norm_nm'],body_panels={},handles={},roots={})
  row['failed_element_material_tracking']=dict(tet_index=bad,vertex_indices=badids,J=float(signed_volumes(bear,tets)[bad]/vol[bad]),original_material_center_m=local[badids].mean(0),original_material_region=anatomy(local[badids].mean(0)),actual_vertices_m=badverts,actual_center_m=badverts.mean(0),whole_tet_all_cloth_AABB_gap_lower_m=max(0.,badclothlower),rigid_hand_reference=jaw)
  for name in['front','back']:
   ids=src['panel_'+name+'_face_indices'];faces=mp[src['faces'][ids]];mesh=trimesh.Trimesh(bag,faces,process=False);near,dist,idx=trimesh.proximity.closest_point(mesh,bear);k=int(dist.argmin());fi=int(idx[k]);sf=int(ids[fi]);normal=mesh.face_normals[fi];selected=np.flatnonzero(dist<=.002)
   row['body_panels'][name]=dict(sampled_bear_node_distance_m=float(dist[k]),bear_node=k,bear_point_m=bear[k],bear_material_original_m=local[k],bear_region=anatomy(local[k]),source_face=sf,cloth_nearest_m=near[k],cloth_face_normal=normal,signed_bear_offset_along_face_normal_m=float((bear[k]-near[k])@normal),body_material_face_centroid_m=src['points'][src['faces'][sf]].mean(0),bear_node_count_within2mm=len(selected),bear_near2mm_nodes=selected,bear_near2mm_material_regions={name:sum(anatomy(local[k])==name for k in selected)for name,_,_ in shapes},panel_mean_displacement_m=(bag-ref['bag'])[mp[src['panel_'+name]]].mean(0))
   ids=groups[name];faces=mp[src['faces'][ids]];samples=np.einsum('kc,fcj->fkj',bary7,bag[faces]).reshape(-1,3);near,dist,idx=trimesh.proximity.closest_point(bm,samples);k=int(dist.argmin());fi,bi=divmod(k,7);bfi=int(idx[k]);wb=trimesh.triangles.points_to_barycentric(bear[bf[[bfi]]],near[[k]])[0];material=wb@local[bf[bfi]];sf=int(ids[fi]);rs=np.array([striprow[name][int(v)]for v in src['faces'][sf]]);arc=float(bary7[bi]@arcs[name][rs]);normal=bm.face_normals[bfi]
   # Whole-tetrahedra versus true triangle AABB lower bound. Only useful if >2mm.
   tetv=bear[tets];lo=tetv.min(1);hi=tetv.max(1);lower=np.inf
   for face in bag[faces]:
    gaps=np.maximum(np.maximum(lo-face.max(0),face.min(0)-hi),0.);lower=min(lower,float(np.linalg.norm(gaps,axis=1).min())-1e-7)
   row['handles'][name]=dict(sampled_handle_to_bear_surface_m=float(dist[k]),sample_count=len(samples),whole_tet_to_tri_AABB_gap_lower_m=max(0.,lower),whole_tet_static_2mm_clear_certified=bool(lower>=.002),handle_point_m=samples[k],bear_point_m=near[k],source_face=sf,material_strip_rows=rs,handle_material_arc_from_left_root_m=arc,handle_total_arc_m=float(arcs[name][-1]),bear_material_original_m=material,bear_region=anatomy(material),bear_outward_normal=normal)
   for side in['left','right']:
    ids=mp[src['handle_'+name+'_root_'+side]];row['roots'][name+'_'+side]=dict(mean_m=bag[ids].mean(0),displacement_m=(bag-ref['bag'])[ids].mean(0))
  print('frame',t,'body',[(n,r['sampled_bear_node_distance_m'],r['bear_region'])for n,r in row['body_panels'].items()],'handles',[(n,r['sampled_handle_to_bear_surface_m'],r['bear_region'])for n,r in row['handles'].items()],'rotation',row['bear_bestfit_rotation_deg'],flush=True);return row
 rows=[]
 for t in[48.5,48.7,48.9,49.0]:
  i=int(np.argmin(abs(ss['time']-t)));rows.append(observe({k:v[i]for k,v in ss.items()},'actual_complete_saved_frame',t))
 rows.append(observe(failed,'actual_failed_soft_frame_no_measured_TCP',float(failed['time'])))
 out=dict(scope='Only five actual native saved shapes near failed insertion; no hypothetical soft-state substitution',native_execution=False,production_edited=False,task_success=False,source_to_native_mapping_max_error_m=float(e.max()),rest_mass_kg=float(mass),weight_N=float(mass*9.81),transport_time_s=float(ref['time']),transport_guard_reference=gref,rows=rows,input_sha256=hashes,limitations=['Body proximity queries all1450 bear vertices to complete physical panel triangles; handle proximity queries triangle vertices,edge midpoints,centroids to exact bear boundary. These distances are upper bounds, not no-intersection proofs.','Explicit whole-tet static AABB lower bounds certify selected saved-frame handle separation only, never between-frame clearance.','Original procedural ellipsoids overlap; reported nearest component is an anatomical approximation.','Near2mm does not prove normal load, stable support or entanglement. Native recorded forces aggregate robot links; they do not identify bag-to-bear material pairs.','A positive change in left-arm load or right-hand unloading is not a quasi-static support certificate without bag/teddy force and acceleration decomposition.','No earlier-release action was planned, authorized or executed by this diagnostic. Containment is independently reviewed.','Exact failed soft frame lacks measured joint/TCP arrays; jaw reference uses same-time recorded actual joint trace and frozen FK, never labels it actual IPC affine geometry. No late coupling trace is available; latest_contact_diagnostics is stale39.265s.'])
 out['failure_element_index']=bad;out['actual_material_peer_contact_scope']='A bad-tet/cloth positive AABB gap excludes direct cloth contact only; cloth loading elsewhere can still transmit compression to the held arm.';out['input_hashes_unchanged']=all(sha(B/p)==h for p,h in hashes.items());path=D/'geometry-and-contact.json';assert not path.exists();path.write_text(json.dumps(out,indent=2,default=ser,allow_nan=False)+'\n');print('REPORT',path,sha(path),flush=True)
if __name__=='__main__':main()
