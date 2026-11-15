"""Bounded saved-state geometry comparison; no native steps or state mutation."""
from pathlib import Path
import sys,json,hashlib,ast
import numpy as np,trimesh,mujoco
from scipy.spatial import cKDTree
P=Path(__file__).resolve().parents[1];RUN=P/'runs/two_panel_full_31';sys.path.insert(0,str(RUN/'source'))
from motion import Planner
from validate import signed_volumes
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
tree=ast.parse((P/'runtime_ipc/full29_palm_contact_review.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='query');exec(compile(ast.Module([node],type_ignores=[]),'pure-query','exec'))
rows=[];hashes={};comparisons=[]
reference31={}
for no in [31,30]:
 run=P/'runs'/f'two_panel_full_{no}';cfg=json.loads((run/'config.json').read_text());rest=dict(np.load(run/'rest.npz'));states=dict(np.load(run/'states.npz'));tr=np.load(run/'joint_trace.npy');guards=json.loads((run/'contact_guard_trace.json').read_text());coupling=json.loads((run/'coupling_trace.json').read_text());ev=json.loads((run/'events.json').read_text())
 asset=dict(np.load(run/'source/assets'/cfg['bag']['npz_file']));dist,mp=cKDTree(rest['bag']).query(asset['points']+cfg['bag']['position']);assert dist.max()<1e-7 and len(np.unique(mp))==len(mp);faces=mp[asset['faces']]
 parts={n:asset['panel_'+n+'_face_indices'] for n in ['front','back']}
 for n in ['front','back']:parts[n+'_handle']=np.flatnonzero(np.all(np.isin(asset['faces'],asset['handle_'+n]),axis=1))
 assert sum(map(len,parts.values()))==len(faces)
 files=[run/n for n in ['config.json','rest.npz','states.npz','joint_trace.npy','contact_guard_trace.json','coupling_trace.json','source/motion.py','source/assets/panda_tactile.xml']]+[run/'source/assets'/cfg['bag']['npz_file']]
 hashes.update({str(p):sha(p) for p in files})
 pc=json.loads(json.dumps(cfg));pc['robots']['mjcf_path']=str(run/'source/assets/panda_tactile.xml');pl=Planner(pc);m,d=pl.m,pl.d
 t0=next(e['start'] for e in ev if e['phase']=='insert_teddy');t1=next((e['start'] for e in ev if e['phase']=='release_teddy'),float(states['time'][-1])+1)
 mask=(states['time']>=t0)&(states['time']<=t1);ii=np.flatnonzero(mask)
 selected=[]
 if no==31:
  for z in [1.06,.99,.95,float(states['tcp_right'][-1,2])]:
   i=int(ii[np.argmin(abs(states['tcp_right'][ii,2]-z))]);selected.append((str(z),{k:v[i] for k,v in states.items()},'states.npz',i));reference31[str(z)]=states['tcp_right'][i].copy()
  failed=dict(np.load(run/'failed_soft_state.npz'));selected.append(('failed',failed,'failed_soft_state.npz',None));hashes[str(run/'failed_soft_state.npz')]=sha(run/'failed_soft_state.npz')
 else:
  for label,tcp in reference31.items():
   i=int(ii[np.argmin(abs(states['tcp_right'][ii,2]-tcp[2]))]);selected.append((label,{k:v[i] for k,v in states.items()},'states.npz',i))
 for label,s,source,index in selected:
  t=float(s['time']);qi=int(np.argmin(abs(tr[:,0]-t)));assert abs(tr[qi,0]-t)<1e-8
  for side,q in [('left',tr[qi,1:10]),('right',tr[qi,10:19])]:
   d.qpos[pl.qindices[side]]=q[:7]
   for j in [1,2]:d.qpos[m.joint(side+'_finger_joint'+str(j)).qposadr[0]]=q[6+j]
  mujoco.mj_forward(m,d);tcp,R=pl.fk('right',tr[qi,10:17]);g=min(guards,key=lambda g:abs(g['time_s']-t));assert abs(g['time_s']-t)<1e-8
  cu=min(coupling,key=lambda c:abs(c['time']-t));err=next(c for c in cu['links'] if c['entity']=='right_franka' and c['link']=='link7') if abs(cu['time']-t)<1e-8 else None
  rr={'run':no,'comparison_label':label,'time_s':t,'state_file':source,'state_index':index,'tcp_actual_FK_m':tcp.tolist(),'FK_to_saved_TCP_m':None if 'tcp_right' not in s else float(np.linalg.norm(tcp-s['tcp_right'])),'native_link7_pose_deviation':err,'nearest_coupling_time':cu['time'],'q_actual_right':tr[qi,10:19].tolist(),'distal_torque_Nm':g['tcp_torque_norm_nm'],'link7_F_N':g['right_link_force_world_n'][0],'link7_tau_TCP_Nm':g['right_link_torque_at_aim_tcp_world_nm'][0],'minJ':float(np.min(signed_volumes(s['bear'],rest['bear_tets'])/signed_volumes(rest['bear'],rest['bear_tets']))),'geometries':{}}
  if no==30:rr['delta_TCP_from_matched_full31_m']=(tcp-reference31[label]).tolist()
  for body in ['right_hand','right_link7']:
   gid=next(i for i in range(m.ngeom) if pl.body_name(i)==body and m.geom_contype[i]);mid=m.geom_dataid[gid];va,vn=m.mesh_vertadr[mid],m.mesh_vertnum[mid];fa,fn=m.mesh_faceadr[mid],m.mesh_facenum[mid];local=trimesh.Trimesh(m.mesh_vert[va:va+vn].astype(float),m.mesh_face[fa:fa+fn],process=False);h=local.convex_hull;assert len(h.faces)<=500
   vs=h.vertices@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid];mesh=trimesh.Trimesh(vs,h.faces,process=False)
   near={p:query(mesh,s['bag'],ids) for p,ids in parts.items()};near['bear']=query(mesh,s['bear'])
   for part,n in near.items():
    if part=='bear':continue
    tri=np.array(n['nearest_cloth_triangle_world_m']);normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal);n['oriented_cloth_face_normal_world']=normal.tolist();n['palm_to_cloth_gap_dot_native_link7_force_N']=float(np.array(n['gap_direction_rigid_to_soft_world'])@rr['link7_F_N'])
    point=np.array(n['rigid_nearest_world_m']);F=np.array(rr['link7_F_N']);tau=np.array(rr['link7_tau_TCP_Nm']);pred=np.cross(point-np.array(g['aim_tcp_world_m']),F);n['single_point_resultant_tau_residual_Nm']=float(np.linalg.norm(tau-pred))
    if part.endswith('handle'):
     side=part.split('_')[0];nc=len(asset['handle_'+side+'_root_left']);inner=np.setdiff1d(asset['handle_'+side],asset['body']);strip=np.vstack([asset['handle_'+side+'_root_left'],inner.reshape(-1,nc),asset['handle_'+side+'_root_right'][::-1]])
     n['material_strip_rows']=[int(np.where(strip==v)[0][0]) for v in n['source_cloth_vertex_ids']];n['strip_row_count']=len(strip)
   radius=np.linalg.norm(vs-d.xpos[m.body('right_link7').id],axis=1).max();bound=None if err is None else err['translation_error_m']+(2*np.sin(err['rotation_error_rad']/2)+err['affine_singular_value_deviation'])*radius
   rr['geometries'][body]={'convex_mesh_vertices':len(h.vertices),'convex_mesh_faces':len(h.faces),'original_mesh_vertices':vn,'original_mesh_faces':fn,'native_affine_pose_error_bound_m':bound,'near':near}
  rows.append(rr);print(json.dumps({'run':no,'t':t,'z':float(tcp[2]),'tau':rr['distal_torque_Nm'],'hand':{n:v['sampled_minimum_distance_m'] for n,v in rr['geometries']['right_hand']['near'].items()}}),flush=True)
res={'kind':'independent_full31_material_change_palm_contact_review','native_run_started':False,'source_hashes':hashes,'source_unchanged':all(sha(p)==h for p,h in hashes.items()),'rows':rows,'limits':['Only saved states and simultaneous actual q/FK. Exact IPC link7 matrix is not archived; complete same-time coupling error bound shown, truefailure has only prior nearby observation.','Whole native-convex hand/link7 surfaces against cloth vertices/midpoints/centroids: positive sampled minima are not whole surface separation certificates.','Full30 matched by TCP height, with actual XYZ delta explicitly recorded. Materials change real cloth shape, and actual replanned route shifts slightly; comparison is not a counterfactual exact same-state test.','Near geometry and aggregate link wrench do not uniquely identify pairwise friction/normal forces.','No motion/assets/material/gate edits or new native run performed.'],'physical_success':False}
out=RUN/'independent-full31-material-palm-contact-review.json';assert not out.exists();out.write_text(json.dumps(res,indent=2,default=lambda o:o.item() if isinstance(o,np.generic) else o.tolist())+'\n');print('SAVED',out,sha(out))
