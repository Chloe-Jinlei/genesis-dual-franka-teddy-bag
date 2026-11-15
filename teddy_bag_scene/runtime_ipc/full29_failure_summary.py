"""Additional bounded native-mesh postprocess correspondence and wrench review."""
from pathlib import Path
import sys,json,hashlib,ast
import numpy as np,trimesh,mujoco
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_29';sys.path.insert(0,str(RUN/'source'))
from motion import Planner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((RUN/'config.json').read_text());rest=dict(np.load(RUN/'rest.npz'));states=dict(np.load(RUN/'states.npz'));failed=dict(np.load(RUN/'failed_soft_state.npz'))
trace=np.load(RUN/'joint_trace.npy');guards=json.loads((RUN/'contact_guard_trace.json').read_text());coupling=json.loads((RUN/'coupling_trace.json').read_text())
asset=dict(np.load(RUN/'source/assets'/cfg['bag']['npz_file']));_,mapping=cKDTree(rest['bag']).query(asset['points']+cfg['bag']['position']);faces=mapping[asset['faces']]
geometry=RUN/'independent-full29-palm-contact-geometry.json';earlier=json.loads(geometry.read_text())
pc=json.loads(json.dumps(cfg));pc['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml');planner=Planner(pc);m,d=planner.m,planner.d
# Reuse only the pure, already executed diagnostic query function, not the script's I/O or main loop.
tree=ast.parse((ROOT/'runtime_ipc/full29_palm_contact_review.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='query');exec(compile(ast.Module([node],type_ignores=[]),'query-only','exec'))
ids=np.flatnonzero(np.all(np.isin(asset['faces'],asset['handle_front']),axis=1));bodyids=asset['panel_front_face_indices']
observations=[]
for t in [47.155,float(failed['time'])]:
    if abs(t-float(failed['time']))<1e-8:s=failed;source='failed_soft_state.npz'
    else:
        i=int(np.argmin(abs(states['time']-t)));s={k:v[i] for k,v in states.items()};source='states.npz'
    t=float(s['time']);ii=int(np.argmin(abs(trace[:,0]-t)));q=trace[ii,10:19]
    for side,qq in [('left',trace[ii,1:10]),('right',q)]:
        d.qpos[planner.qindices[side]]=qq[:7]
        for j in(1,2):d.qpos[m.joint(side+'_finger_joint'+str(j)).qposadr[0]]=qq[6+j]
    mujoco.mj_forward(m,d)
    gid=next(i for i in range(m.ngeom) if planner.body_name(i)=='right_hand' and m.geom_contype[i])
    mid=m.geom_dataid[gid];va,vn=m.mesh_vertadr[mid],m.mesh_vertnum[mid];fa,fn=m.mesh_faceadr[mid],m.mesh_facenum[mid]
    local=trimesh.Trimesh(m.mesh_vert[va:va+vn].astype(float),m.mesh_face[fa:fa+fn],process=False)
    hull=local.convex_hull;assert len(hull.faces)<=500
    verts=hull.vertices@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
    mesh=trimesh.Trimesh(verts,hull.faces,process=False)
    near=query(mesh,s['bag'],ids);front=query(mesh,s['bag'],bodyids);bear=query(mesh,s['bear'])
    g=min(guards,key=lambda g:abs(g['time_s']-t));assert abs(g['time_s']-t)<1e-8
    F=np.array(g['right_link_force_world_n'][0]);tau=np.array(g['right_link_torque_at_aim_tcp_world_nm'][0]);p=np.array(near['rigid_nearest_world_m']);tcp=np.array(g['aim_tcp_world_m']);lever=p-tcp
    estimate=np.cross(lever,F);residual=tau-estimate;perp=np.cross(F,tau)/np.dot(F,F);on_line=tcp+perp+np.dot(lever-perp,F)*F/np.dot(F,F)
    hbid=m.body('right_hand').id;lbid=m.body('right_link7').id
    nc=len(asset['handle_front_root_left']);inner=np.setdiff1d(asset['handle_front'],asset['body']);strip=np.vstack([asset['handle_front_root_left'],inner.reshape(-1,nc),asset['handle_front_root_right'][::-1]])
    face=near['source_cloth_face_id'];material_rows=[int(np.where(strip==v)[0][0]) for v in asset['faces'][face]]
    original=next(r for r in earlier['rows'] if abs(r['time_s']-t)<1e-8)['geometries']['right_hand']['source_surface_distances']['front_handle']
    observations.append(dict(time_s=t,soft_geometry_source=source,native_postprocess_model=dict(original_vertices=len(local.vertices),original_faces=len(local.faces),convex_vertices=len(hull.vertices),convex_faces=len(hull.faces),decimation_applies=False,
        convex_volume_relative_difference=float(hull.volume/local.volume-1),scope='Convex hull of the same compiled collision mesh, matching configured default convexification; <=500 faces means the engine decimate method leaves it unchanged. Not an observation of the IPC affine pose.'),
        native_processed_reference_mesh_front_handle=near,native_processed_reference_mesh_front_panel=front,native_processed_reference_mesh_bear_nodes=bear,
        nearest_hand_local_point_m=((p-d.xpos[hbid])@d.xmat[hbid].reshape(3,3)).tolist(),nearest_link7_local_point_m=((p-d.xpos[lbid])@d.xmat[lbid].reshape(3,3)).tolist(),
        held_front_material_rows=material_rows,strip_row_count=len(strip),touches_right_sewn_root=bool(np.isin(asset['faces'][face],asset['handle_front_root_right']).any()),
        difference_from_original_mesh_sample_distance_m=near['sampled_minimum_distance_m']-original['sampled_minimum_distance_m'],
        force_moment_consistency=dict(link7_force_world_N=F.tolist(),link7_measured_TCP_moment_Nm=tau.tolist(),nearest_contact_lever_from_aim_TCP_m=lever.tolist(),
          point_force_predicted_TCP_moment_Nm=estimate.tolist(),residual_Nm=residual.tolist(),residual_norm_Nm=float(np.linalg.norm(residual)),
          torque_component_parallel_to_force_Nm=float(tau@F/np.linalg.norm(F)),equivalent_force_line_closest_point_world_m=on_line.tolist(),contact_to_equivalent_force_line_distance_m=float(np.linalg.norm(on_line-p)),
          scope='Compatibility test for one resultant force applied near this point; does not resolve multiple contact forces or prove a unique pair cause.')))

native_root=ROOT.parent/'genesis-world/genesis'
native_sources=[native_root/'engine/couplers/ipc_coupler/coupler.py',native_root/'engine/entities/rigid_entity/description.py',native_root/'engine/mesh.py',native_root/'utils/mjcf.py']
inputs={str(geometry):sha(geometry),**earlier['source_sha256'],**{str(p):sha(p) for p in native_sources}}
out=dict(kind='independent_full29_failure_review',physical_task_success=False,original_torque_limit_Nm=.15,true_failure_time_s=float(failed['time']),last_complete_state_time_s=float(states['time'][-1]),
    source_sha256=inputs,inputs_unchanged=all(sha(p)==h for p,h in inputs.items()),additional_observations=observations,
    conclusion='The held front handle right straight leg and its sewn mouth root are the closest identified physical material to the right palm. As the palm descends, near-contact migrates down the leg to the root. The measured link7 resultant moment closely matches the lever arm to this palm contact. The fixed hand is merged into IPC link7, so the link7 wrench label does not imply the wrist shell itself is contacting cloth.',
    root_question_positive_Y_offset='The evidence identifies a local front right strap/root obstruction; it does not establish positive-Y offset as its sole cause. The nearest contact is+X and−Y from the TCP. A negative-X path adjustment is a geometrically motivated candidate to move away from that right strap leg, but a new complete route and actual-state checks are required. Increasing descent depth along the unchanged axis continues loading this root.',
    physical_next_step='Keep all force, torque, soft deformation and final containment gates. A bounded lateral repositioning while the palm is still above the strap/root can be evaluated against the complete hand/strap surface and the teddy/left-arm route. Do not assume a final-pose offset or a faster/slower downward push resolves the caught strap. Alternatively redesign the approach/held-handle pose so the palm can descend without feeding the held strap underneath it.',
    limitations=earlier['limitations']+['Convexification correspondence removes the main original-surface ambiguity for this hand mesh; same-time native affine matrices remain unavailable.','No new offset, timing, orientation or control policy has been planned or tested here.','The scene-local aggregate wrench cannot distinguish held front strip from simultaneously approaching front panel at a few millimetres; near-contact at the stitched root may involve both.'])
p=RUN/'independent-full29-failure-review.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(path=str(p),sha256=sha(p),observations=observations)),flush=True)
