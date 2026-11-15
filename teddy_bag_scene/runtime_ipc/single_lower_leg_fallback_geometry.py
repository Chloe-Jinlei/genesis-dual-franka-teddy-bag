"""One declared lower-leg fallback geometry check; no adoption or native run."""
from pathlib import Path
import copy
import hashlib
import itertools
import json
import sys
import numpy as np
import mujoco
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/two_panel_full_23'
OUT=ROOT/'runtime_ipc/single-lower-leg-fallback-assessment'
OUT.mkdir(exist_ok=True)
upper_only=False
if upper_only:
    OUT=OUT/'authorized-upper-candidate'
    OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
from motion import Planner,TCP,DOWN
from contact_grasp import cloth_grasp_frame,track_reference_displacement,_clip
from contact_robot_geometry import _convex,_common_ball
from validate import tetra_boundary

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((RUN/'config.json').read_text())
cfg['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
rest=dict(np.load(RUN/'rest.npz'))
initial=dict(np.load(RUN/'initial.npz'))
settled=dict(np.load(RUN/'teddy_pregrasp.npz'))
origin=np.array(cfg['bear']['position'])
radius=float(cfg['task']['bear_grasp_tracking_radius_m'])
rotation=DOWN
R=rotation.as_matrix()
candidates=[[.034,-.10,.0255]]
if upper_only:
    candidates=[[0.,.07225,.0765]]
protected=[RUN/n for n in ('config.json','rest.npz','initial.npz','teddy_pregrasp.npz','source/motion.py','source/run.py','source/contact_robot_geometry.py','source/contact_grasp.py','source/assets/panda_tactile.xml','source/assets/bear_uniform_coarse_s85.obj')]
before={str(p):sha(p) for p in protected}
planner=Planner(cfg)
bag=np.load(RUN/'source/assets'/cfg['bag']['npz_file'])
leftspec=cfg['task']['handle_grasps']['left']
leftframe=cloth_grasp_frame(bag['points'],bag['faces'],bag[leftspec['group']],shell_radius=cfg['bag']['thickness'],preferred_approach=leftspec['preferred_approach'])
ql,el,al=planner.solve('left',np.array(cfg['bag']['position'])+leftframe.position,leftframe.rotation)
leftfinger=max(.001,leftframe.minimum_half_gap)
planner.set_fingers({'left':leftfinger,'right':.04})
mujoco.mj_forward(planner.m,planner.d)
hand=planner.bodies['right']; hr=planner.d.xmat[hand].reshape(3,3);tcp=planner.d.xpos[hand]+hr@TCP
signs=np.array(list(itertools.product((-1.,1.),repeat=3)))
components=[]
for gid in planner.geoms['right']:
    body=planner.body_name(gid)
    if not (body.endswith('_finger') or body.endswith('_hand')): continue
    kind=planner.m.geom_type[gid]
    if kind==mujoco.mjtGeom.mjGEOM_BOX:
        local=signs*planner.m.geom_size[gid]
    elif kind==mujoco.mjtGeom.mjGEOM_MESH:
        mid=planner.m.geom_dataid[gid];adr,num=planner.m.mesh_vertadr[mid],planner.m.mesh_vertnum[mid]
        local=planner.m.mesh_vert[adr:adr+num]
    else: raise ValueError('Unsupported hand collision component')
    world=local@planner.d.geom_xmat[gid].reshape(3,3).T+planner.d.geom_xpos[gid]
    handlocal=(world-tcp)@hr
    components.append(dict(geom_id=int(gid),body=body,local=handlocal))

def check_solid_against_teddy(vertices,points):
    solid=_convex(vertices); tets=points[rest['bear_tets']]
    lo,hi=solid['bounds']; ids=np.flatnonzero(np.all(tets.max(1)>=lo,axis=1)&np.all(tets.min(1)<=hi,axis=1))
    nlp=0;unknown=[];maxr=-np.inf
    for tid in ids:
        tet=tets[tid]
        # A strict separating hull face is an exact fast exclusion.
        vals=tet@solid['equations'][:,:3].T+solid['equations'][:,3]
        if np.any(vals.min(0)>1e-9):continue
        nlp+=1; w=_common_ball(solid,_convex(tet))
        if w is None:unknown.append(int(tid));continue
        maxr=max(maxr,w['radius_m'])
        if w['radius_m']>1e-7:
            return dict(status='blocked_conservative_convex_component',tetrahedron=int(tid),common_interior_ball=w,LP_count=nlp,overlap_is_physical_native_contact_proof=False)
        if w['radius_m']>=-1e-7:unknown.append(int(tid))
    return dict(status='indeterminate_near_boundary' if unknown else 'clear_conservative_solid_geometry',broadphase_tetrahedra=int(len(ids)),LP_count=nlp,maximum_common_ball_radius_m=None if not np.isfinite(maxr) else maxr,unknown_tetrahedra=unknown)

def evaluate_shape(label,state,target):
    points=state['bear'];faces=tetra_boundary(points,rest['bear_tets'])
    component_rows=[]
    for c in components:
        end=c['local']@R.T+target
        initial_geom=end+[0,0,.09]
        initcheck=check_solid_against_teddy(initial_geom,points)
        sweep=check_solid_against_teddy(np.vstack([end,initial_geom]),points)
        component_rows.append(dict(geom_id=c['geom_id'],body=c['body'],initial_90mm_above=initcheck,complete_90mm_descent=sweep))
    # Actual triangulated surface ray sections in the square 17mm pad footprint.
    mesh=trimesh.Trimesh(points,faces,process=False)
    n=21; grid=(np.arange(n)+.5)/n*.017-.0085
    xx,zz=np.meshgrid(grid,grid)
    starts=np.column_stack([xx.ravel(),np.full(n*n,.2),zz.ravel()])@R.T+target
    directions=np.tile(-R[:,1],(len(starts),1))
    loc,ri,fi=mesh.ray.intersects_location(starts,directions,multiple_hits=True)
    lh=(loc-target)@R; spans=[]
    for ray in range(len(starts)):
        ids=np.flatnonzero(ri==ray); ids=ids[np.argsort(-lh[ids,1])]
        for a,b in zip(ids[::2],ids[1::2]):
            high,low=lh[a,1],lh[b,1]
            if low<0<high and high<.04 and low>-.04:
                spans.append([high,low,abs(mesh.face_normals[fi[a]]@R[:,1]),abs(mesh.face_normals[fi[b]]@R[:,1])]);break
    spans=np.array(spans).reshape(-1,4)
    tri=(points[faces]-target)@R;corridor=[]
    for poly in tri:
        for axis,bound,sign in ((0,.0085,1),(0,-.0085,-1),(2,.009358,1),(2,-.09,-1)):
            poly=_clip(poly,axis,bound,sign)
        if len(poly):corridor.extend(poly[:,1])
    initial_clear=all(c['initial_90mm_above']['status']=='clear_conservative_solid_geometry' for c in component_rows)
    sweep_clear=all(c['complete_90mm_descent']['status']=='clear_conservative_solid_geometry' for c in component_rows)
    return dict(source_state=label,source_time_s=float(state['time']),tracked_target_world_m=target,components=component_rows,initial_open_hand_clear=initial_clear,full_open_hand_and_fingers_descent_clear=sweep_clear,pad_footprint_ray_count=n*n,central_two_sided_intervals_within_80mm=int(len(spans)),sampled_two_sided_projected_area_m2=float(len(spans)*(.017/n)**2),sampled_surface_touch_half_gaps_m=None if not len(spans) else dict(positive_side_range=[spans[:,0].min(),spans[:,0].max()],negative_side_abs_range=[(-spans[:,1]).min(),(-spans[:,1]).max()]),sampled_absolute_surface_normal_dot_closing_axis=None if not len(spans) else spans[:,2:].mean(0),full_descent_central_pad_corridor_required_half_opening_m=max(abs(np.array(corridor))) if corridor else None,geometric_closure_opportunity=bool(len(spans)),friction_grasp_success=False)

results=[]
for i,local in enumerate(candidates):
    ref=origin+local;roi=np.flatnonzero(np.linalg.norm(rest['bear']-ref,axis=1)<radius)
    initial_target=ref.copy()
    actual_target=track_reference_displacement(rest['bear'],settled['bear'],ref,radius=radius)
    actual_target[2]=max(actual_target[2],cfg['table']['height']+.025)
    shapes=[evaluate_shape('initial.npz',initial,initial_target),evaluate_shape('teddy_pregrasp.npz',settled,actual_target)]
    poses=[];seed=None
    for label,target in [('native_probe_initial',initial_target+[0,0,.09]),('frozen_settled_grasp',actual_target),('hypothetical_probe_lift',[actual_target[0],actual_target[1],1.17])]:
        q,pe,re=planner.solve('right',target,rotation,seed=seed);seed=q
        clear=planner.clearance({'left':ql,'right':q},{'left':leftfinger,'right':.04})
        finite=all(np.isfinite(v['distance']) for v in clear.values())
        poses.append(dict(label=label,target_world_m=target,q=q,position_error_m=pe,rotation_error_rad=re,IK_tolerance_pass=bool(pe<=.001 and re<=.005),all_raw_rigid_finite=bool(finite),minimum_raw_rigid_m=min(v['distance'] for v in clear.values()),raw_rigid=clear))
    clear=all(s['initial_open_hand_clear'] and s['full_open_hand_and_fingers_descent_clear'] and s['geometric_closure_opportunity'] for s in shapes)
    ikpass=all(p['IK_tolerance_pass'] and p['all_raw_rigid_finite'] and p['minimum_raw_rigid_m']>=.008 for p in poses)
    row=dict(index=i,local_reference_m=local,tracking_radius_m=radius,tracking_roi_count=int(len(roi)),shapes=shapes,robot_poses=poses,geometry_candidate=bool(clear and ikpass),physical_grasp_success=False)
    if row['geometry_candidate']:
        new=copy.deepcopy(json.loads((RUN/'config.json').read_text()))
        new['task']['bear_grasp_local']=local;new['task']['bear_grasp_rotation_xyzw']=rotation.as_quat().tolist()
        new['task']['bear_finger_force_limit']=3.;new['task']['bear_finger_target']=0.;new['task']['grasp_probe_hold_seconds']=2.
        p=OUT/f'config_lower_leg_geometry_candidate_{i}.json';p.write_text(json.dumps(new,indent=2)+'\n')
        row['provisional_probe_config']=dict(path=str(p),sha256=sha(p),task_argument='grasp',not_authorized_for_full_task=True)
    results.append(row)
    print(json.dumps(dict(index=i,roi=len(roi),geometry_candidate=row['geometry_candidate'],shapes=[dict(source=s['source_state'],initial_clear=s['initial_open_hand_clear'],sweep_clear=s['full_open_hand_and_fingers_descent_clear'],pad_rays=s['central_two_sided_intervals_within_80mm'],corridor_half_m=s['full_descent_central_pad_corridor_required_half_opening_m']) for s in shapes],IK=[dict(label=p['label'],pass_=p['IK_tolerance_pass'],rigid_m=p['minimum_raw_rigid_m']) for p in poses])),flush=True)

out=dict(kind='explicitly_authorized_upper_head_grasp_geometry_only' if upper_only else 'single_declared_lower_leg_grasp_geometry_only',physical_success=False,native_started=False,bear_scale=.85,maximum_total_finger_opening_m=.08,rotation_definition='Default DOWN; hand local Y closing axis maps to world -Y',rotation_xyzw=rotation.as_quat(),closing_axis_world=R[:,1],candidate_count=len(candidates),left_initial_IK=dict(position_error_m=el,rotation_error_rad=al,q=ql,initial_finger_half_gap_m=leftfinger),candidates=results,all_protected_inputs_unchanged=all(sha(p)==h for p,h in before.items()),input_sha256=before,limitations=['3N is only a proposed finite motor force setting, not a holding guarantee.','Both rest/initial and one saved ungrasped settled shape are frozen geometry cases; the future grasp probe has its own fresh soft state.','Robot hand collision meshes are individually convexified. A blocked convex component can overestimate its nonconvex native surface; no surface-body collision law is changed.','The 90mm straight Cartesian hand sweep is a continuous frozen solid envelope; robot IK is checked only at declared poses, not the entire arm trajectory.','Pad ray sections are geometric closure opportunities, not actual contact area, force closure or stability evidence.','This candidate is a stored fallback only, not adopted. It makes no claim about dynamic holding, natural hanging pose or fitting the current bag depth.'])
path=OUT/'review.json';path.write_text(json.dumps(out,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item(),allow_nan=False)+'\n')
print(json.dumps(dict(path=str(path),sha256=sha(path),candidate_indices=[r['index'] for r in results if r['geometry_candidate']]),indent=2))
