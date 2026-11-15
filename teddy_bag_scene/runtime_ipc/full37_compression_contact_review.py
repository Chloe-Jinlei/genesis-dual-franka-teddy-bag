"""Read-only bounded material-cell/contact comparison; no native calls."""
from pathlib import Path
import sys,json,hashlib,ast,itertools
import numpy as np,trimesh,mujoco
from scipy.spatial import cKDTree
P=Path(__file__).resolve().parents[1];R=P/'runs/two_panel_full_37';sys.path.insert(0,str(R/'source'))
from validate import signed_volumes
from motion import Planner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ser=lambda o:o.item() if isinstance(o,np.generic) else o.tolist()
fail=dict(np.load(R/'failed_soft_state.npz'));r0=dict(np.load(R/'rest.npz'));Jfail=signed_volumes(fail['bear'],r0['bear_tets'])/signed_volumes(r0['bear'],r0['bear_tets']);bad=int(Jfail.argmin());ids=r0['bear_tets'][bad];tfinal=float(fail['time']);allrows=[];hashes={};target_z=None
# Extract only the pure prior tet-point/cloth-triangle diagnostic.
tree=ast.parse((P/'runtime_ipc/full28_release_failure_review.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sampled_tet_distance');exec(compile(ast.Module([fn],type_ignores=[]),'prior-pure-distance','exec'));edges4=np.array(list(itertools.combinations(range(4),2)))
for no in [37,30,32]:
 run=P/'runs'/f'two_panel_full_{no}';cfg=json.loads((run/'config.json').read_text());rest=dict(np.load(run/'rest.npz'));ss=dict(np.load(run/'states.npz'));tr=np.load(run/'joint_trace.npy');cu=json.loads((run/'coupling_trace.json').read_text());gu=json.loads((run/'contact_guard_trace.json').read_text());events=json.loads((run/'events.json').read_text());source=dict(np.load(run/'source/assets'/cfg['bag']['npz_file']));table=cfg['table']['height'];dist,mp=cKDTree(rest['bag']).query(source['points']+cfg['bag']['position']);assert dist.max()<1e-7;faces=mp[source['faces']]
 assert np.array_equal(rest['bear_tets'],r0['bear_tets']) and np.array_equal(rest['bear'],r0['bear'])
 parts={n+'_panel':source['panel_'+n+'_face_indices'] for n in ['front','back']}
 for n in ['front','back']:parts[n+'_handle']=np.flatnonzero(np.all(np.isin(source['faces'],source['handle_'+n]),axis=1))
 pc=json.loads(json.dumps(cfg));pc['robots']['mjcf_path']=str(run/'source/assets/panda_tactile.xml');pl=Planner(pc);m,d=pl.m,pl.d
 for p in [run/f for f in ['config.json','rest.npz','states.npz','joint_trace.npy','coupling_trace.json','contact_guard_trace.json','source/motion.py','source/assets/panda_tactile.xml','report.json','events.json','source/assets/'+cfg['bag']['npz_file']]]:hashes[str(p)]=sha(p)
 selected=[]
 if no==37:
  for t in [tfinal-.4,tfinal-.2,tfinal-.075,float(ss['time'][-1])]:
   i=int(np.argmin(abs(ss['time']-t)));selected.append(({k:v[i] for k,v in ss.items()},'states.npz',i))
  selected.append((fail,'failed_soft_state.npz',None));hashes[str(R/'failed_soft_state.npz')]=sha(R/'failed_soft_state.npz');target_z=float(ss['tcp_right'][-1,2])
 elif no in [30,32]:
  start=next(e['start'] for e in events if e['phase']=='insert_teddy');end=next((e['start'] for e in events if e['phase']=='release_teddy'),float(json.loads((run/'report.json').read_text())['simulated_seconds'])+1e-8);mask=(ss['time']>=start)&(ss['time']<end);ix=np.flatnonzero(mask);i=int(ix[np.argmin(abs(ss['tcp_right'][ix,2]-target_z))]);selected.append(({k:v[i] for k,v in ss.items()},'states.npz',i))
 else:
  s=dict(np.load(run/'failed_soft_state.npz'));selected.append((s,'failed_soft_state.npz',None));hashes[str(run/'failed_soft_state.npz')]=sha(run/'failed_soft_state.npz')
 for s,file,index in selected:
  time=float(s['time']);qi=int(np.argmin(abs(tr[:,0]-time)));assert abs(tr[qi,0]-time)<1e-8;g=min(gu,key=lambda g:abs(g['time_s']-time));assert abs(g['time_s']-time)<1e-8;cou=min(cu,key=lambda c:abs(c['time']-time));samecu=abs(cou['time']-time)<1e-8
  for side,q in [('left',tr[qi,1:10]),('right',tr[qi,10:19])]:
   d.qpos[pl.qindices[side]]=q[:7]
   for j in [1,2]:d.qpos[m.joint(side+'_finger_joint'+str(j)).qposadr[0]]=q[6+j]
  mujoco.mj_forward(m,d);tcp,rot=pl.fk('right',tr[qi,10:17]);tet=s['bear'][ids];samples=np.vstack([tet,tet[edges4].mean(1),tet.mean(0)]);Dm=(rest['bear'][ids[1:]]-rest['bear'][ids[0]]).T;F=(tet[1:]-tet[0]).T@np.linalg.inv(Dm);U,S,Vt=np.linalg.svd(F);J=signed_volumes(s['bear'],rest['bear_tets'])/signed_volumes(rest['bear'],rest['bear_tets'])
  row={'run':no,'time_s':time,'source':file,'state_index':index,'observation_phase':'insert_teddy','comparison_selection':'full37 bounded failure trend' if no==37 else 'nearest saved TCP height within actual insert phase, not matched time or soft shape','same_time_native_finger_transform_saved':samecu,'nearest_native_coupling_time_s':cou['time'],'TCP_FK_m':tcp.tolist(),'actual_saved_TCP_m':s.get('tcp_right'),'failed_material_tet_J':float(J[bad]),'global_minJ':float(J.min()),'global_worst_tet':int(J.argmin()),'failed_tet_world_centroid_m':tet.mean(0),'failed_tet_world_vertices_m':tet,'principal_stretches':S,'minimum_stretch_spatial_direction':U[:,-1],'table_clearance_m':float(tet[:,2].min()-table),'physical_cloth_distances':{n:sampled_tet_distance(tet,s['bag'],fi) for n,fi in parts.items()},'jaw_contacts':{},'guard':{k:g[k] for k in ['net_force_norm_n','tcp_torque_norm_nm','right_finger_opening_projection_n','right_finger_force_norm_n','force_world_n','right_link_force_world_n','right_link_names','right_link_torque_at_aim_tcp_world_nm','aim_tcp_world_m']},'finger_q_m':tr[qi,17:19].tolist()}
  for bname,short in [('right_hand','link7'),('right_left_finger','left_finger'),('right_right_finger','right_finger')]:
   obj=[]
   for gid in pl.geoms['right']:
    if pl.body_name(gid)!=bname:continue
    typ=m.geom_type[gid]
    if typ==mujoco.mjtGeom.mjGEOM_BOX:lv=np.array(list(itertools.product([-1.,1.],repeat=3)))*m.geom_size[gid]
    elif typ==mujoco.mjtGeom.mjGEOM_MESH:
     mi=m.geom_dataid[gid];a,n=m.mesh_vertadr[mi],m.mesh_vertnum[mi];lv=m.mesh_vert[a:a+n].astype(float)
    else:raise ValueError('geometry type')
    world=lv@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid];bodyid=m.body(bname).id;frame='same-time actual-q FK reference, not saved IPC affine'
    if bname!='right_hand' and samecu:
     link=next(c for c in cou['links'] if c['entity']=='right_franka' and c['link']==short);T=np.array(link['ipc_transform']);local=(world-d.xpos[bodyid])@d.xmat[bodyid].reshape(3,3);world=local@T[:3,:3].T+T[:3,3];frame='same-time saved native IPC affine transform of finger with frozen collision components'
    mesh=trimesh.convex.convex_hull(world);near,dd,idx=trimesh.proximity.closest_point(mesh,samples);i=int(dd.argmin());radius=np.linalg.norm(world-world.mean(0),axis=1).max();gap=np.maximum(0,np.maximum(tet.min(0)-world.max(0),world.min(0)-tet.max(0)))
    obj.append({'geom_id':int(gid),'shape':'box' if typ==mujoco.mjtGeom.mjGEOM_BOX else 'mesh_convex','frame':frame,'tet11sample_distance_m':float(dd[i]),'tet_sample_index':i,'tet_sample_m':samples[i],'rigid_nearest_m':near[i],'whole_AABB_lower_m':float(np.linalg.norm(gap)),'robot_surface_face_normal':mesh.face_normals[idx[i]].tolist()})
   row['jaw_contacts'][bname]=min(obj,key=lambda x:x['tet11sample_distance_m'])
  allrows.append(row);print(json.dumps({'run':no,'t':time,'tetJ':row['failed_material_tet_J'],'tcpZ':float(tcp[2]),'cloth':{n:x['distance_m'] for n,x in row['physical_cloth_distances'].items()},'jaws':{n:x['tet11sample_distance_m'] for n,x in row['jaw_contacts'].items()}}),flush=True)
cfg=json.loads((R/'config.json').read_text());rt=r0['bear'][ids];vol=abs(signed_volumes(r0['bear'],r0['bear_tets']));meanratio=12*(3*vol[bad])**(2/3)/sum(np.sum((rt[j]-rt[i])**2) for i,j in edges4);met=np.load(R/'soft_metrics.npy');idx=np.flatnonzero(met[:,1]<.05);out={'kind':'independent_full37_localized_compression_review','physical_success':False,'native_simulation_performed':False,'failed_time_s':tfinal,'last_complete_time_s':float(np.load(R/'states.npz')['time'][-1]),'first_J_below_gate_time_s':float(met[idx[0],0]),'number_of_failing_metric_steps':len(idx),'failed_tet':{'index':bad,'vertex_ids':ids,'J':float(Jfail[bad]),'rest_unscaled_centroid_m':(rt.mean(0)-cfg['bear']['position'])/.663,'approximate_anatomy':'right arm/held grasp-side material from original procedural coordinates; full37 tet3255 differs from full32 tet895 in the same arm region','rest_mean_ratio_quality':float(meanratio),'rest_volume_m3':float(vol[bad]),'only_tets_below_005':np.flatnonzero(Jfail<.05)},'rows':allrows,'last_100ms_metrics':met[met[:,0]>=tfinal-.1-1e-8],'metric_columns':json.loads((R/'soft_metrics_columns.json').read_text()),'source_hashes':hashes,'inputs_unchanged':all(sha(p)==h for p,h in hashes.items()),'limitations':['11 tet samples give contact diagnostic upper-bound distances, not exact tetrahedron separation or force attribution; AABB lower bound applies whole tet.','Native finger affine transforms exist only for complete saved coupling frames; failed56.270 uses same-time actualjoint FK reference, never reused56.250 transform as current.','Conservative convex collision component hulls are used for imported finger meshes; sub500 face shape correspondence is not independently reproved here.','Whole robot link aggregate forces cannot resolve bag-to-bear contacts or friction/normal decomposition.','Full30 and full32 compare use nearest saved insertion TCP height; actual XY/R, loading history, material and soft shape differ. These are not causal counterfactuals.','No new material, motion, geometry, bear or gate edit/native run.']};dest=R/'independent-full37-compression-contact-review.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2,default=ser)+'\n');print('SAVED',dest,sha(dest))
