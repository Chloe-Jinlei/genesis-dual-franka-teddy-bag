"""One signed -45deg yaw from actual pre-yaw lift, with new actual held bag; CPU-only."""
from pathlib import Path
import copy,hashlib,json,sys,types,time
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_handoff_upright_offset_01';OLD=ROOT/'runs/two_panel_full_21';OUT=RUN/'independent-hybrid-lift-worldz-minus45';OUT.mkdir(exist_ok=True);sys.path.insert(0,str(RUN/'source'))
import postlift_yaw as original
from contact_insertion import plan_guarded_contact_trial,_state_context,_policy,_mouth,_serial,_section_envelopes,_evaluate_offset
from insertion_candidates import polygon_centroid,projected_inside
from validate import signed_volumes,tetra_boundary,mesh_edges
from scipy.spatial import cKDTree
from motion import Planner
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[RUN/'settle_handoff.npz',RUN/'rest.npz',RUN/'config.json',OLD/'lift_teddy.npz',OLD/'rest.npz',OLD/'config.json',*[RUN/'source'/n for n in ('motion.py','postlift_yaw.py','contact_insertion.py','contact_robot_geometry.py')],Path(__file__)]
hashes={str(p):sha(p) for p in inputs};production={str(ROOT/n):sha(ROOT/n) for n in ('run.py','motion.py','postlift_yaw.py','config_upright_offset_handle_probe.json')}
cfg=json.loads((RUN/'config.json').read_text());oldcfg=json.loads((OLD/'config.json').read_text());bag=dict(np.load(RUN/'settle_handoff.npz'));bear=dict(np.load(OLD/'lift_teddy.npz'));nr=dict(np.load(RUN/'rest.npz'));r=dict(np.load(OLD/'rest.npz'))
assert all(cfg[k]==oldcfg[k] for k in ('robots','bag','bear','table','simulation'));assert all(np.array_equal(nr[k],r[k]) for k in r)
rest={k:v.copy() for k,v in r.items()};state=dict(bag=bag['bag'].copy(),bear=bear['bear'].copy(),q_left=bag['q_left'].copy(),tcp_left=bag['tcp_left'].copy(),q_right=bear['q_right'].copy(),tcp_right=bear['tcp_right'].copy(),time=np.array(max(float(bag['time']),float(bear['time']))))
np.savez_compressed(OUT/'hybrid_before_yaw.npz',**state);np.savez_compressed(OUT/'rest.npz',**rest)
params=copy.deepcopy(cfg);params['task']['contact_trial']['postlift_yaw']={'world_z_degrees':-45.,'duration_s':6.};params['task']['contact_trial']['center_offsets_xy_m']=[[0.,0.]]
(OUT/'diagnostic_config.json').write_text(json.dumps(params,indent=2)+'\n')
params['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
for key in ('npz_file','metadata'):params['bag'][key]=str(RUN/'source/assets'/cfg['bag'][key])
pl=Planner(params);A=pl.fk('right',state['q_right'][:7])[1];G=Rotation.from_rotvec([0.,0.,-np.pi/4]);target=G*A;tcp=state['tcp_right'];turned=G.apply(state['bear']-tcp)+tcp
report=dict(kind='one_hybrid_worldZ_minus45_from_actual_pre_yaw_lift',physical_success=False,native_execution=False,not_a_joint_native_snapshot=True,current_production_yaw_policy_admissible=False,
 sources=dict(actual_bag_and_left_path=str(RUN/'settle_handoff.npz'),actual_bag_and_left_time_s=float(bag['time']),actual_bear_and_right_path=str(OLD/'lift_teddy.npz'),actual_bear_and_right_time_s=float(bear['time']),uses_postyaw_bear=False),
 total_world_z_rotation_degrees=-45.,rotation_formula='target_full_R = R_worldZ(-45deg) * actual_lift_FK_R; bear_nominal = R_worldZ(-45deg)*(actual_lift_bear-measured_TCP)+measured_TCP',
 actual_source_full_R_xyzw=A.as_quat().tolist(),target_full_R_xyzw=target.as_quat().tolist(),relative_target_rotvec_rad=(target*A.inv()).as_rotvec().tolist(),
 source_actual_bear_dimensions_m=np.ptp(state['bear'],axis=0).tolist(),ideal_rotated_dimensions_m=np.ptp(turned,axis=0).tolist(),z_height_invariant_error_m=float(abs(np.ptp(state['bear'][:,2])-np.ptp(turned[:,2]))),
 selected_only_initial_com_offset_m=[0.,0.],no_inherited_force_arrays=True,input_sha256=hashes,production_sha256_before=production,status='planning',
 limitations=['Different native observations are composed only for hypothetical geometry; their simultaneous physical reachability is unproven.','The production helper only permits |yaw|<=15deg. This diagnostic clones it in memory and changes only its magnitude bound to45deg; no production source,policy orconfig is changed.','The actual measured q values serve as hypothetical command starts. This is not an archived real command stream.','All other original physical,8mm sampled rigid,5mm whole-teddy arc,continuous position/rate,FK and finger gates remain in force.','Only one pose and unique COM offset[0,0] will be evaluated; no parameter grid.'])
def save():
 report['inputs_unchanged']=all(sha(Path(p))==v for p,v in hashes.items());report['production_unchanged']=all(sha(Path(p))==v for p,v in production.items());(OUT/'review.json').write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')
context=_state_context(rest,state,params,_policy(params));report['hybrid_context_rejections']=context['reasons'];report['hybrid_context_numeric_metrics']=context['physical'];polygon,triangles,mouth=_mouth(rest,state,params);report['mouth']={k:mouth[k] for k in ('rim_z_min_m','rim_z_max_m','projected_area_m2')};report['original_height_margin_m']=mouth['rim_z_min_m']-.002-(cfg['table']['height']+.01)-float(np.ptp(turned[:,2]))
weights=np.abs(signed_volumes(rest['bear'],rest['bear_tets']));turned_com=np.average(turned[rest['bear_tets']].mean(1),axis=0,weights=weights);anchor=polygon_centroid(polygon)
inside,distance=projected_inside(np.array([anchor]),polygon);sections,envelopes=_section_envelopes(turned,rest['bear_tets']);evaluation=_evaluate_offset(envelopes,anchor-turned_com[:2],polygon,triangles,mouth,_policy(params))
report['initial_nominal_rotated_section_screen']=dict(target_com_xy_m=anchor.tolist(),actual_rotated_mass_com_m=turned_com.tolist(),translation_xy_m=(anchor-turned_com[:2]).tolist(),com_projected_inside=bool(inside[0]),com_boundary_distance_m=float(distance[0]),section_event_count=len(sections),slab_count=len(envelopes),evaluation={k:v for k,v in evaluation.items() if k!='sections'})
initial_rejections=list(context['reasons'])+list(evaluation['section_guard_rejections'])
if report['original_height_margin_m']<0:initial_rejections.append('original frozen height gate failed')
if not bool(inside[0]) or distance[0]<.002:initial_rejections.append('original 2mm COM margin failed')
if initial_rejections:
 report['status']='rejected_initial_context_or_section';report['initial_rejections']=initial_rejections;report['rotation_path_assessed']=False;report['route_assessed']=False;save();print('INITIAL_SECTION_REJECTED',json.dumps(report['initial_nominal_rotated_section_screen']),flush=True);raise SystemExit(0)
print('INITIAL_SECTION_PASSED',json.dumps(report['initial_nominal_rotated_section_screen']),flush=True)
# Cheap declared-pose handle surface test before any yaw/route IK.
asset=dict(np.load(RUN/'source/assets'/cfg['bag']['npz_file']));error,mapping=cKDTree(rest['bag']).query(asset['points']+cfg['bag']['position']);assert error.max()<1e-7 and len(np.unique(mapping))==len(mapping)
bf=tetra_boundary(rest['bear'],rest['bear_tets']);tri=turned[bf];e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
handles={}
for name in ('front','back'):
 ids=np.flatnonzero(np.isin(asset['faces'],asset['handle_'+name]).all(axis=1));faces=mapping[asset['faces'][ids]];edges,_,_=mesh_edges(faces);assert len(faces)==352;handles[name]=edges
outside=tcp.copy();outside[:2]+=anchor-turned_com[:2];outside[2]=max(tcp[2],mouth['rim_z_max_m']+.045-(turned[:,2]-tcp[2]).min())
report['declared_ideal_transport_tcp_m']=outside.tolist();report['declared_early_handle_depths_m']=[0.,.005,.025,.050];report['early_handle_surface_checks']=[]
for depth in (0.,.005,.025,.050):
 delta=outside-tcp-[0.,0.,depth];row=dict(depth_below_transport_m=depth,translation_from_ideal_rotated_shape_m=delta.tolist(),handles={})
 for name,edges in handles.items():
  origins=state['bag'][edges[:,0]]-delta;dirs=state['bag'][edges[:,1]]-state['bag'][edges[:,0]];count=0;hits=[]
  for first in range(0,len(edges),64):
   O=origins[first:first+64];D=dirs[first:first+64];H=np.cross(D[:,None,:],e2[None]);det=np.einsum('eji,ji->ej',H,e1);good=np.abs(det)>1e-12;inv=np.divide(1.,det,out=np.zeros_like(det),where=good);S=O[:,None,:]-tri[None,:,0];u=np.einsum('eji,eji->ej',S,H)*inv;Q=np.cross(S,e1[None]);v=np.einsum('ei,eji->ej',D,Q)*inv;t=np.einsum('ji,eji->ej',e2,Q)*inv
   valid=good&(u>1e-8)&(v>1e-8)&(u+v<1-1e-8)&(t>1e-8)&(t<1-1e-8);pairs=np.argwhere(valid);count+=len(pairs)
   for ee,ff in pairs[:max(0,3-len(hits))]:
    edge=edges[first+ee];bary=np.array([1-u[ee,ff]-v[ee,ff],u[ee,ff],v[ee,ff]]);point=state['bag'][edge[0]]+t[ee,ff]*dirs[first+ee];res=float(np.linalg.norm(point-(bary@tri[ff]+delta)));assert res<1e-10
    hits.append(dict(handle_edge=edge.tolist(),bear_surface_face=int(ff),bear_face_vertices=bf[ff].tolist(),edge_fraction=float(t[ee,ff]),bear_barycentric=bary.tolist(),point_world_m=point.tolist(),residual_m=res))
  row['handles'][name]=dict(strict_crossing_count=count,first_witnesses=hits)
 report['early_handle_surface_checks'].append(row)
 if any(h['strict_crossing_count'] for h in row['handles'].values()):
  report['status']='rejected_early_frozen_handle_surface_crossing';report['rotation_path_assessed']=False;report['route_assessed']=False;report['handle_screen_scope']='Both full physical straps and whole bear boundary at stated static poses. Positive strict edge/face crossing rejects this frozen hypothesis, not a native-penetration claim. No-hit would not certify surface separation or continuous clearance.';save();print('EARLY_HANDLES_REJECTED',json.dumps(row),flush=True);raise SystemExit(0)
report['handle_screen_scope']='Only four declared static poses, both full straps and whole bear boundary. No strict edge/face crossings observed is not a disjointness or continuous clearance certificate.'
print('EARLY_FOUR_HANDLE_POSES_NO_KNOWN_CROSSING',flush=True)
q,pe,ae=pl.solve('right',tcp,target,state['q_right'][:7]);report['single_endpoint_IK']=dict(position_error_m=pe,full_rotation_error_rad=ae,q=q.tolist(),passed=pe<=.0005 and ae<=.002)
save();print('ENDPOINT',report['single_endpoint_IK'],'height_margin',report['original_height_margin_m'],flush=True)
if context['reasons'] or not report['single_endpoint_IK']['passed']:
 report['status']='rejected_before_rotation_path';save();raise SystemExit(0)
text=(RUN/'source/postlift_yaw.py').read_text();needle='if not 0 < abs(degrees) <= 15 or duration < 6:';assert text.count(needle)==1;modified=text.replace(needle,'if not 0 < abs(degrees) <= 45 or duration < 6:')
module=types.ModuleType('diagnostic_only_yaw45');module.__file__=str(RUN/'source/postlift_yaw.py');exec(compile(modified,'<diagnostic_only_yaw45>','exec'),module.__dict__)
command={s:state['q_'+s][:7].copy() for s in ('left','right')};prior,priorplan=original.plan_postlift_yaw(rest,state,params,command);assert prior['status']=='rejected' and priorplan is None
report['original_policy_rejection']=prior['rejections'];report['inmemory_diagnostic_helper_sha256']=hashlib.sha256(modified.encode()).hexdigest();t=time.monotonic();yaw,plan=module.plan_postlift_yaw(rest,state,params,command);report['yaw_preflight']=yaw;report['yaw_wall_s']=time.monotonic()-t
if plan is None:
 report['status']='rejected_rotation_path';report['route_assessed']=False;save();print('YAW_REJECTED',yaw['rejections'],flush=True);raise SystemExit(0)
np.savez_compressed(OUT/'hypothetical_yaw_plan.npz',duration=plan.duration,**{s+'_x':plan.curves[s].x for s in command},**{s+'_c':plan.curves[s].c for s in command})
end=plan.end_q();endtcp,endR=pl.fk('right',end['right']);relative=endR*A.inv();hyp={k:v.copy() for k,v in state.items()};hyp['q_right'][:7]=end['right'];hyp['q_left'][:7]=end['left'];hyp['tcp_right']=endtcp.copy();hyp['bear']=relative.apply(state['bear']-tcp)+endtcp;steps=int(np.ceil(plan.duration/cfg['simulation']['dt']));hyp['time']=np.array(float(state['time'])+steps*cfg['simulation']['dt'])
np.savez_compressed(OUT/'hybrid_hypothetical_after_yaw.npz',**hyp)
report['hypothetical_after_yaw']=dict(formula='actual endpoint FK_R * inverse(actual lift FK_R) applied about old measured TCP, then endpoint FK_TCP',endpoint_tcp_error_m=float(np.linalg.norm(endtcp-tcp)),endpoint_full_rotation_error_rad=float((endR*target.inv()).magnitude()),maximum_point_difference_from_exact_minus45_m=float(np.linalg.norm(hyp['bear']-turned,axis=1).max()),clock_increment_steps=steps,synthetic_after_time_s=float(hyp['time']),bag_unchanged=bool(np.array_equal(hyp['bag'],state['bag'])),left_q_unchanged=bool(np.array_equal(hyp['q_left'],state['q_left'])),actual_native_checkpoint=False)
report['status']='rotation_passed_planning_unique_zero_route';save();print('YAW_PASSED','rigid',yaw['robot_preflight']['minimum_sampled_rigid_clearance_m'],'soft',yaw['frozen_bear_left_rotation_sweep']['minimum_continuous_gap_lower_bound_m'],'duration',plan.duration,flush=True)
t=time.monotonic();choice=plan_guarded_contact_trial(rest,hyp,params);choice['not_a_joint_native_snapshot']=True;choice['hybrid_source_provenance_path']=str(OUT/'review.json');(OUT/'guarded_candidate.json').write_text(json.dumps(_serial(choice),indent=2,allow_nan=False)+'\n')
report['route_assessed']=True;report['route_wall_s']=time.monotonic()-t;report['route_candidate_path']=str(OUT/'guarded_candidate.json');report['route_candidate_status']=choice['status'];report['route_rejections']=choice['report'].get('rejections',[]);report['route_attempts']=[dict(center=a['center_of_mass_xy_target_m'],rejection=a['rejection'],stages_completed=a['stages_completed']) for a in choice['report'].get('robot_centering_attempts',[])];report['status']='bounded_screen_complete';save();print('DONE',json.dumps({k:report[k] for k in ('status','route_candidate_status','route_rejections','route_attempts')}),flush=True)
