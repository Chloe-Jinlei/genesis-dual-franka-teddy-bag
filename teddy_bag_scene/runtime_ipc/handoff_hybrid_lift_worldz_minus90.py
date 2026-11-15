"""One signed -90deg yaw from actual pre-yaw lift, with new actual held bag; CPU-only."""
from pathlib import Path
import copy,hashlib,json,sys,types,time
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_handoff_upright_offset_01';OLD=ROOT/'runs/two_panel_full_21';OUT=RUN/'independent-hybrid-lift-worldz-minus90';OUT.mkdir(exist_ok=True);sys.path.insert(0,str(RUN/'source'))
import postlift_yaw as original
from contact_insertion import plan_guarded_contact_trial,_state_context,_policy,_mouth,_serial
from motion import Planner
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[RUN/'settle_handoff.npz',RUN/'rest.npz',RUN/'config.json',OLD/'lift_teddy.npz',OLD/'rest.npz',OLD/'config.json',*[RUN/'source'/n for n in ('motion.py','postlift_yaw.py','contact_insertion.py','contact_robot_geometry.py')],Path(__file__)]
hashes={str(p):sha(p) for p in inputs};production={str(ROOT/n):sha(ROOT/n) for n in ('run.py','motion.py','postlift_yaw.py','config_upright_offset_handle_probe.json')}
cfg=json.loads((RUN/'config.json').read_text());oldcfg=json.loads((OLD/'config.json').read_text());bag=dict(np.load(RUN/'settle_handoff.npz'));bear=dict(np.load(OLD/'lift_teddy.npz'));nr=dict(np.load(RUN/'rest.npz'));r=dict(np.load(OLD/'rest.npz'))
assert all(cfg[k]==oldcfg[k] for k in ('robots','bag','bear','table','simulation'));assert all(np.array_equal(nr[k],r[k]) for k in r)
rest={k:v.copy() for k,v in r.items()};state=dict(bag=bag['bag'].copy(),bear=bear['bear'].copy(),q_left=bag['q_left'].copy(),tcp_left=bag['tcp_left'].copy(),q_right=bear['q_right'].copy(),tcp_right=bear['tcp_right'].copy(),time=np.array(max(float(bag['time']),float(bear['time']))))
np.savez_compressed(OUT/'hybrid_before_yaw.npz',**state);np.savez_compressed(OUT/'rest.npz',**rest)
params=copy.deepcopy(cfg);params['task']['contact_trial']['postlift_yaw']={'world_z_degrees':-90.,'duration_s':6.};params['task']['contact_trial']['center_offsets_xy_m']=[[0.,0.]]
(OUT/'diagnostic_config.json').write_text(json.dumps(params,indent=2)+'\n')
params['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
for key in ('npz_file','metadata'):params['bag'][key]=str(RUN/'source/assets'/cfg['bag'][key])
pl=Planner(params);A=pl.fk('right',state['q_right'][:7])[1];G=Rotation.from_rotvec([0.,0.,-np.pi/2]);target=G*A;tcp=state['tcp_right'];turned=G.apply(state['bear']-tcp)+tcp
report=dict(kind='one_hybrid_worldZ_minus90_from_actual_pre_yaw_lift',physical_success=False,native_execution=False,not_a_joint_native_snapshot=True,current_production_yaw_policy_admissible=False,
 sources=dict(actual_bag_and_left_path=str(RUN/'settle_handoff.npz'),actual_bag_and_left_time_s=float(bag['time']),actual_bear_and_right_path=str(OLD/'lift_teddy.npz'),actual_bear_and_right_time_s=float(bear['time']),uses_postyaw_bear=False),
 total_world_z_rotation_degrees=-90.,rotation_formula='target_full_R = R_worldZ(-90deg) * actual_lift_FK_R; bear_nominal = R_worldZ(-90deg)*(actual_lift_bear-measured_TCP)+measured_TCP',
 actual_source_full_R_xyzw=A.as_quat().tolist(),target_full_R_xyzw=target.as_quat().tolist(),relative_target_rotvec_rad=(target*A.inv()).as_rotvec().tolist(),
 source_actual_bear_dimensions_m=np.ptp(state['bear'],axis=0).tolist(),ideal_rotated_dimensions_m=np.ptp(turned,axis=0).tolist(),z_height_invariant_error_m=float(abs(np.ptp(state['bear'][:,2])-np.ptp(turned[:,2]))),
 selected_only_initial_com_offset_m=[0.,0.],no_inherited_force_arrays=True,input_sha256=hashes,production_sha256_before=production,status='planning',
 limitations=['Different native observations are composed only for hypothetical geometry; their simultaneous physical reachability is unproven.','The production helper only permits |yaw|<=15deg. This diagnostic clones it in memory and changes only its magnitude bound to90deg; no production source,policy orconfig is changed.','The actual measured q values serve as hypothetical command starts. This is not an archived real command stream.','All other original physical,8mm sampled rigid,5mm whole-teddy arc,continuous position/rate,FK and finger gates remain in force.','Only one pose and unique COM offset[0,0] will be evaluated; no parameter grid.'])
def save():
 report['inputs_unchanged']=all(sha(Path(p))==v for p,v in hashes.items());report['production_unchanged']=all(sha(Path(p))==v for p,v in production.items());(OUT/'review.json').write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')
context=_state_context(rest,state,params,_policy(params));report['hybrid_context_rejections']=context['reasons'];report['hybrid_context_numeric_metrics']=context['physical'];_,_,mouth=_mouth(rest,state,params);report['mouth']={k:mouth[k] for k in ('rim_z_min_m','rim_z_max_m','projected_area_m2')};report['original_height_margin_m']=mouth['rim_z_min_m']-.002-(cfg['table']['height']+.01)-float(np.ptp(turned[:,2]))
q,pe,ae=pl.solve('right',tcp,target,state['q_right'][:7]);report['single_endpoint_IK']=dict(position_error_m=pe,full_rotation_error_rad=ae,q=q.tolist(),passed=pe<=.0005 and ae<=.002)
save();print('ENDPOINT',report['single_endpoint_IK'],'height_margin',report['original_height_margin_m'],flush=True)
if context['reasons'] or not report['single_endpoint_IK']['passed']:
 report['status']='rejected_before_rotation_path';save();raise SystemExit(0)
text=(RUN/'source/postlift_yaw.py').read_text();needle='if not 0 < abs(degrees) <= 15 or duration < 6:';assert text.count(needle)==1;modified=text.replace(needle,'if not 0 < abs(degrees) <= 90 or duration < 6:')
module=types.ModuleType('diagnostic_only_yaw90');module.__file__=str(RUN/'source/postlift_yaw.py');exec(compile(modified,'<diagnostic_only_yaw90>','exec'),module.__dict__)
command={s:state['q_'+s][:7].copy() for s in ('left','right')};prior,priorplan=original.plan_postlift_yaw(rest,state,params,command);assert prior['status']=='rejected' and priorplan is None
report['original_policy_rejection']=prior['rejections'];report['inmemory_diagnostic_helper_sha256']=hashlib.sha256(modified.encode()).hexdigest();t=time.monotonic();yaw,plan=module.plan_postlift_yaw(rest,state,params,command);report['yaw_preflight']=yaw;report['yaw_wall_s']=time.monotonic()-t
if plan is None:
 report['status']='rejected_rotation_path';report['route_assessed']=False;save();print('YAW_REJECTED',yaw['rejections'],flush=True);raise SystemExit(0)
np.savez_compressed(OUT/'hypothetical_yaw_plan.npz',duration=plan.duration,**{s+'_x':plan.curves[s].x for s in command},**{s+'_c':plan.curves[s].c for s in command})
end=plan.end_q();endtcp,endR=pl.fk('right',end['right']);relative=endR*A.inv();hyp={k:v.copy() for k,v in state.items()};hyp['q_right'][:7]=end['right'];hyp['q_left'][:7]=end['left'];hyp['tcp_right']=endtcp.copy();hyp['bear']=relative.apply(state['bear']-tcp)+endtcp;steps=int(np.ceil(plan.duration/cfg['simulation']['dt']));hyp['time']=np.array(float(state['time'])+steps*cfg['simulation']['dt'])
np.savez_compressed(OUT/'hybrid_hypothetical_after_yaw.npz',**hyp)
report['hypothetical_after_yaw']=dict(formula='actual endpoint FK_R * inverse(actual lift FK_R) applied about old measured TCP, then endpoint FK_TCP',endpoint_tcp_error_m=float(np.linalg.norm(endtcp-tcp)),endpoint_full_rotation_error_rad=float((endR*target.inv()).magnitude()),maximum_point_difference_from_exact_minus90_m=float(np.linalg.norm(hyp['bear']-turned,axis=1).max()),clock_increment_steps=steps,synthetic_after_time_s=float(hyp['time']),bag_unchanged=bool(np.array_equal(hyp['bag'],state['bag'])),left_q_unchanged=bool(np.array_equal(hyp['q_left'],state['q_left'])),actual_native_checkpoint=False)
report['status']='rotation_passed_planning_unique_zero_route';save();print('YAW_PASSED','rigid',yaw['robot_preflight']['minimum_sampled_rigid_clearance_m'],'soft',yaw['frozen_bear_left_rotation_sweep']['minimum_continuous_gap_lower_bound_m'],'duration',plan.duration,flush=True)
t=time.monotonic();choice=plan_guarded_contact_trial(rest,hyp,params);choice['not_a_joint_native_snapshot']=True;choice['hybrid_source_provenance_path']=str(OUT/'review.json');(OUT/'guarded_candidate.json').write_text(json.dumps(_serial(choice),indent=2,allow_nan=False)+'\n')
report['route_assessed']=True;report['route_wall_s']=time.monotonic()-t;report['route_candidate_path']=str(OUT/'guarded_candidate.json');report['route_candidate_status']=choice['status'];report['route_rejections']=choice['report'].get('rejections',[]);report['route_attempts']=[dict(center=a['center_of_mass_xy_target_m'],rejection=a['rejection'],stages_completed=a['stages_completed']) for a in choice['report'].get('robot_centering_attempts',[])];report['status']='bounded_screen_complete';save();print('DONE',json.dumps({k:report[k] for k in ('status','route_candidate_status','route_rejections','route_attempts')}),flush=True)
