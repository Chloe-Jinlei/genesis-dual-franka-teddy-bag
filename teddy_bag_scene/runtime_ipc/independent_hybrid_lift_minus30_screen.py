"""One authorized -30 degree frozen geometry screen; no IK, trajectory or native run."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_handoff_upright_offset_01';OLD=ROOT/'runs/two_panel_full_21'
OUT=RUN/'independent-hybrid-lift-worldz-minus30';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
from contact_insertion import _state_context,_policy,_mouth,_section_envelopes,_evaluate_offset
from insertion_candidates import polygon_centroid,projected_inside
from validate import signed_volumes,tetra_boundary,mesh_edges

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def load(path):return dict(np.load(path,allow_pickle=False))
def read(path):return json.loads(path.read_text())
cfg=read(RUN/'config.json');oldcfg=read(OLD/'config.json')
bag=load(RUN/'settle_handoff.npz');bear=load(OLD/'lift_teddy.npz');rest=load(RUN/'rest.npz');oldrest=load(OLD/'rest.npz')
inputs=[RUN/'settle_handoff.npz',RUN/'rest.npz',RUN/'config.json',OLD/'lift_teddy.npz',OLD/'rest.npz',OLD/'config.json',
        RUN/'source/assets'/cfg['bag']['npz_file'],*[RUN/'source'/n for n in ('motion.py','contact_insertion.py','contact_robot_geometry.py','contact_slice_geometry.py','insertion_candidates.py','validate.py')],Path(__file__)]
hashes={str(p):sha(p) for p in inputs}
production={str(p):sha(p) for p in ROOT.glob('*.py')}
production.update({str(p):sha(p) for p in ROOT.glob('config*.json')})
checks=dict(same_material_robots_environment=all(cfg[k]==oldcfg[k] for k in ('robots','bag','bear','table','simulation')),
            same_rest_geometry_and_topology=all(np.array_equal(v,oldrest[k]) for k,v in rest.items()),
            expected_frozen_motion=sha(RUN/'source/motion.py')=='544d6bed1aedfe1603db860af0cd4d230641714827ff23ef27c93d0a12e32a31')
assert all(checks.values())
state=dict(bag=bag['bag'].copy(),bear=bear['bear'].copy(),q_left=bag['q_left'].copy(),tcp_left=bag['tcp_left'].copy(),
           q_right=bear['q_right'].copy(),tcp_right=bear['tcp_right'].copy(),time=np.array(max(float(bag['time']),float(bear['time']))))
np.savez_compressed(OUT/'hybrid_before_rotation.npz',**state)
params=copy.deepcopy(cfg);params['task']['contact_trial']['center_offsets_xy_m']=[[0.,0.]]
(OUT/'diagnostic_config.json').write_text(json.dumps(params,indent=2)+'\n')
params['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
for key in ('npz_file','metadata'):params['bag'][key]=str(RUN/'source/assets'/cfg['bag'][key])
policy=_policy(params);context=_state_context(rest,state,params,policy)
A=context['rotation'];G=Rotation.from_euler('z',-30.,degrees=True);target=G*A;tcp=state['tcp_right'];turned=G.apply(state['bear']-tcp)+tcp
np.savez_compressed(OUT/'ideal_rotated_geometry_only.npz',bear=turned,target_full_R_xyzw=target.as_quat(),pivot_tcp=tcp)
report=dict(kind='single_minus30_from_actual_lift_section_then_two_handle_geometry_screen',physical_success=False,native_execution=False,
    not_a_joint_native_snapshot=True,robot_pose_after_rotation_solved=False,rotation_path_assessed=False,transport_path_assessed=False,
    source_actual_bag_and_left=dict(path=str(RUN/'settle_handoff.npz'),time_s=float(bag['time'])),
    source_actual_bear_and_right=dict(path=str(OLD/'lift_teddy.npz'),time_s=float(bear['time']),uses_postyaw_bear=False),
    synthetic_bookkeeping_time_s=float(state['time']),total_world_z_angle_degrees=-30.,only_COM_offset_m=[0.,0.],
    rotation_formula='target_R=Rz(-30deg)*actual_lift_FK_R; ideal_bear=Rz(-30deg)*(actual_lift_bear-actual_lift_TCP)+actual_lift_TCP',
    actual_source_full_R_xyzw=A.as_quat(),target_full_R_xyzw=target.as_quat(),relative_target_rotvec_rad=(target*A.inv()).as_rotvec(),
    ideal_rotated_dimensions_m=np.ptp(turned,axis=0),source_dimensions_m=np.ptp(state['bear'],axis=0),
    current_production_postlift_angle_policy_admissible=False,policy=policy,checks=checks,
    hybrid_before_rotation_context_rejections=context['reasons'],hybrid_before_rotation_physical_metrics=context['physical'],
    input_sha256=hashes,production_sha256_before=production,static_handle_checks=[],
    limitations=[
        'Hybrid combines two separate native observations; joint reachability, shared force equilibrium and dynamic timing are unproven.',
        'The ideal rotated bear has no solved corresponding robot q. Context applies to the pre-rotation hybrid only. Target preserves full actual FK orientation using world-left multiplication.',
        'Only COM offset [0,0] is considered. No turn, transport, release, or insertion motion is executed or planned here.',
        'Production postlift helper restricts |angle|<=15 degrees; -30 degrees is a distinct offline geometric question, not an admissible production helper configuration.',
        'Strict interior handle-edge to whole-bear-triangle intersection proves crossing only in a frozen hypothetical static pose. No-hit does not certify disjointness or continuous clearance; tangencies/coplanar/reverse-edge cases remain outside this check.',
        'Bag geometry and all original physical and section policies are preserved. Real cloth deformation, force/grasp retention, insertion and final containment remain unknown.'])
out=OUT/'review.json'
def save():
    report['inputs_unchanged']=all(sha(Path(p))==h for p,h in hashes.items())
    report['production_unchanged']=all(sha(Path(p))==h for p,h in production.items())
    out.write_text(json.dumps(report,indent=2,default=serial,allow_nan=False)+'\n')

polygon,triangles,mouth=_mouth(rest,state,params)
weights=np.abs(signed_volumes(rest['bear'],rest['bear_tets']));com=np.average(turned[rest['bear_tets']].mean(1),axis=0,weights=weights)
anchor=polygon_centroid(polygon);inside,distance=projected_inside(np.array([anchor]),polygon)
sections,envelopes=_section_envelopes(turned,rest['bear_tets'])
evaluation=_evaluate_offset(envelopes,anchor-com[:2],polygon,triangles,mouth,policy)
height_margin=mouth['rim_z_min_m']-policy['endpoint_below_lowest_rim_m']-(cfg['table']['height']+policy['bottom_clearance_m'])-np.ptp(turned[:,2])
report['mouth']={k:mouth[k] for k in ('rim_z_min_m','rim_z_max_m','projected_area_m2')}
report['original_height_margin_m']=float(height_margin)
report['nominal_rotated_section_screen']=dict(target_COM_xy_m=anchor,rotated_rest_mass_COM_m=com,
    translation_xy_m=anchor-com[:2],COM_projected_inside=bool(inside[0]),COM_boundary_distance_m=float(distance[0]),
    section_event_count=len(sections),slab_count=len(envelopes),evaluation={k:v for k,v in evaluation.items() if k!='sections'})
reasons=list(context['reasons'])+list(evaluation['section_guard_rejections'])
if height_margin<0:reasons.append('original frozen height gate failed')
if not inside[0] or distance[0]<policy['com_boundary_clearance_m']:reasons.append('original COM margin failed')
clearances=context['physical']['actual_rigid_clearances']
if not all(np.isfinite(x['distance']) for x in clearances.values()):reasons.append('nonfinite raw pre-rotation rigid distance')
report['initial_rejections']=reasons
if reasons:
    report.update(status='rejected_original_context_or_section',handle_static_poses_assessed=False)
    save();print(json.dumps(dict(output=str(out),sha256=sha(out),status=report['status'],reasons=reasons,section=report['nominal_rotated_section_screen']),indent=2,default=serial));raise SystemExit(0)

source=load(RUN/'source/assets'/cfg['bag']['npz_file'])
error,mapping=cKDTree(rest['bag']).query(source['points']+cfg['bag']['position'])
assert len(mapping)==len(rest['bag'])==len(np.unique(mapping)) and error.max()<1e-7
assert np.array_equal(mapping[source['faces']],rest['bag_faces'])
bf=tetra_boundary(rest['bear'],rest['bear_tets']);tri=turned[bf];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
handles={}
for name in ('front','back'):
    ids=np.flatnonzero(np.isin(source['faces'],source['handle_'+name]).all(axis=1));faces=mapping[source['faces'][ids]]
    assert len(faces)==352
    edges,_,_=mesh_edges(faces);handles[name]=dict(original_face_ids=ids,native_faces=faces,native_edges=edges)
report['topology']=dict(full_bear_nodes=len(turned),full_bear_tets=len(rest['bear_tets']),whole_bear_boundary_faces=len(bf),
    source_to_rest_error_m=float(error.max()),handle_face_counts={k:len(v['native_faces']) for k,v in handles.items()})

def crossings(delta,h):
    edges=h['native_edges'];origins=state['bag'][edges[:,0]]-delta
    directions=state['bag'][edges[:,1]]-state['bag'][edges[:,0]];hits=[];count=0
    for first in range(0,len(origins),64):
        origin=origins[first:first+64];direction=directions[first:first+64]
        H=np.cross(direction[:,None,:],e2[None]);det=np.einsum('eji,ji->ej',H,e1)
        good=np.abs(det)>1e-12;inverse=np.divide(1.,det,out=np.zeros_like(det),where=good)
        S=origin[:,None,:]-tri[None,:,0];u=np.einsum('eji,eji->ej',S,H)*inverse;Q=np.cross(S,e1[None])
        v=np.einsum('ei,eji->ej',direction,Q)*inverse;t=np.einsum('ji,eji->ej',e2,Q)*inverse
        assert all(np.isfinite(x).all() for x in (det,u,v,t))
        valid=good&(u>1e-8)&(v>1e-8)&(u+v<1-1e-8)&(t>1e-8)&(t<1-1e-8)
        indices=np.argwhere(valid);count+=len(indices)
        for ee,ff in indices[:max(0,3-len(hits))]:
            edge=edges[first+ee];bary=np.array([1-u[ee,ff]-v[ee,ff],u[ee,ff],v[ee,ff]])
            point=state['bag'][edge[0]]+t[ee,ff]*directions[first+ee]
            residual=float(np.linalg.norm(point-(bary@tri[ff]+delta)));assert residual<1e-10
            hits.append(dict(handle_material_edge=edge,whole_bear_surface_face=int(ff),whole_bear_face_vertices=bf[ff],
                segment_fraction=float(t[ee,ff]),bear_triangle_barycentric=bary,world_point_m=point,reconstruction_residual_m=residual))
    return dict(proper_edge_triangle_crossing_count=count,first_strict_crossing_witnesses=hits)

outside=tcp.copy();outside[:2]+=anchor-com[:2]
outside[2]=max(float(tcp[2]),mouth['rim_z_max_m']+policy['planned_outside_clearance_m']-float((turned-tcp)[:,2].min()))
report['ideal_transport_endpoint_TCP_m']=outside
report['transport_endpoint_definition']='Original targets_for: COM to projected-rim area centroid; z=max(actual TCP z, actual rim max z+45mm-rotated bear relative minimum z). Not all-cloth maximum height.'
report['handle_static_poses_assessed']=True
for depth in [0.,.005,.025,.050]:
    destination=outside-np.array([0.,0.,depth]);delta=destination-tcp
    row=dict(depth_below_ideal_transport_endpoint_m=depth,ideal_TCP_m=destination,translation_of_turned_bear_m=delta,
             handles={k:crossings(delta,h) for k,h in handles.items()})
    report['static_handle_checks'].append(row)
    print('depth',depth,{k:v['proper_edge_triangle_crossing_count'] for k,v in row['handles'].items()},flush=True)
    if any(v['proper_edge_triangle_crossing_count'] for v in row['handles'].values()):
        report.update(status='rejected_strict_frozen_handle_bear_intersection',first_rejected_depth_m=depth,remaining_depths_not_assessed_due_to_early_rejection=True)
        break
else:
    report.update(status='no_strict_crossing_found_at_four_declared_static_poses',requires_separate_authorization_before_full_IK_or_route=True)
save();print(json.dumps(dict(output=str(out),sha256=sha(out),status=report['status']),indent=2))
