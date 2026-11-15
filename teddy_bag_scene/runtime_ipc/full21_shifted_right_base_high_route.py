"""One base, four offsets: frozen high-route CPU study, never a native plan."""
from pathlib import Path
import copy, hashlib, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/two_panel_full_21'
OUT=RUN/'independent-shifted-right-base-high-route'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
from motion import Planner,MotionPlan,CubicSpline,minimum_jerk
from contact_insertion import _state_context,_policy,_mouth,_section_envelopes,_evaluate_offset
from insertion_candidates import projected_inside,polygon_centroid
from contact_robot_geometry import check_translation_sweep
from postlift_yaw import continuous_rate_bounds,_clearance_rows,_finite_tree,_serial

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tracked=[RUN/'config.json',RUN/'rest.npz',RUN/'orient_teddy_postlift.npz']+[RUN/'source'/f for f in ['motion.py','contact_insertion.py','contact_robot_geometry.py','postlift_yaw.py','physical_monitor.py','validate.py','assets/panda_tactile.xml']]
before={str(p):sha(p) for p in tracked}
cfg=json.loads((RUN/'config.json').read_text())
bound=copy.deepcopy(cfg);bound['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
new=copy.deepcopy(bound);new['robots']['right_base']=[.28,.24,.76]
rest=dict(np.load(RUN/'rest.npz'));state=dict(np.load(RUN/'orient_teddy_postlift.npz'))
state_before={k:v.copy() for k,v in state.items()}
policy=_policy(cfg)
report=dict(kind='full21_single_shifted_base_four_offset_frozen_high_route',physical_success=False,
    native_execution_performed=False,gpu_used=False,production_files_edited=False,
    actual_source_state=str(RUN/'orient_teddy_postlift.npz'),actual_source_time_s=float(state['time']),
    previous_right_base_m=cfg['robots']['right_base'],candidate_right_base_m=new['robots']['right_base'],
    right_base_yaw_rad=new['robots']['right_yaw'], original_policy=policy,
    changed_physics_fields=['robots.right_base'],frozen_xml_explicitly_bound=True,
    four_com_offsets_m=[[0.,0.],[0.,-.02],[-.01,-.025],[.005,-.03]],cases=[],
    limits=dict(rigid_m=.008,whole_teddy_left_m=.005,all_cloth_high_gap_m=.01,
        velocity_rad_s=.35,acceleration_rad_s2=.5,fk_position_m=.0005,fk_rotation_rad=.002),
    limitations=['Changing the base is not simulated or executable from the old joint state. A fresh native run must validate every initial phase and grasp.',
        'New-base source q is solved for the original actual TCP and full rotation; this is an explicit frozen hypothetical robot arrangement, not a saved actual state.',
        'All soft points, left measured q and both individual finger positions are frozen. Native forces, deformation, slip and evolving cloth are not predicted.',
        'The 5mm sweep covers every frozen tetrahedron along ideal straight Cartesian translation against original MuJoCo convex left collision solids. It is not a continuous joint-FK error or native contact certificate.',
        'Robot clearances and Cartesian FK are 401 sampled checks. Joint limits and conservative velocity/acceleration bounds cover the continuous reference curve.',
        '10mm above every cloth vertex certifies only the frozen high traverse. Descent through actual cloth/body/straps and whole final containment remain unproved.'],input_sha256=before)

def save():
    report['input_hashes_unchanged']=before=={str(p):sha(p) for p in tracked}
    report['actual_input_arrays_unchanged']=all(np.array_equal(v,state_before[k]) for k,v in state.items())
    (OUT/'review.json').write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')

def clearance_individual(planner,qs):
    for side in ('left','right'):
        for j,value in enumerate(state['q_'+side][7:],1):
            planner.d.qpos[planner.m.joint(f'{side}_finger_joint{j}').qposadr[0]]=value
    rows=planner.clearance(qs,None)
    required={'left_self','left_table','left_mounts','right_self','right_table','right_mounts','arms'}
    if set(rows)!=required or not _finite_tree(rows): raise ValueError('Malformed/nonfinite original rigid categories')
    return rows

context=_state_context(rest,state,bound,policy)
report['actual_source_original_physical_state']=context['physical']
if context['reasons']: raise ValueError('Original actual source violates physical gates: '+str(context['reasons']))
target_rotation=context['rotation'];tcp=state['tcp_right'].copy()
planner=Planner(new)
q,pe,re=planner.solve('right',tcp,target_rotation,context['qs']['right'],max_nfev=250)
initial_q=dict(left=context['qs']['left'].copy(),right=q)
initial_rows=clearance_individual(planner,initial_q)
initial_cat,initial_min=min(initial_rows.items(),key=lambda x:x[1]['distance'])
report['new_base_same_pose_ik']=dict(q_right=q,position_error_m=pe,rotation_error_rad=re,
    target_tcp_m=tcp,target_rotation_xyzw=target_rotation.as_quat(),raw_rigid_clearances=initial_rows,
    minimum_rigid_clearance_m=initial_min['distance'],within_original_tolerances=pe<=.0005 and re<=.002,
    no_motion_from_old_base_q_is_assumed=True)
save()
if not (np.isfinite(q).all() and pe<=.0005 and re<=.002 and initial_min['distance']>=.008):
    report['status']='rejected_at_new_base_source_arrangement';save();print(report['status'],flush=True);sys.exit(0)
hyp_state={k:v.copy() for k,v in state.items()};hyp_state['q_right']=np.r_[q,state['q_right'][7:]]
new_context=_state_context(rest,hyp_state,new,policy)
report['new_base_hypothetical_source_physical_check']=new_context['physical']
if new_context['reasons']: raise ValueError('New-base hypothetical source violates original gates: '+str(new_context['reasons']))
np.savez(OUT/'explicit_hypothetical_source_state.npz',**hyp_state)
polygon,triangles,mouth=_mouth(rest,state,bound)
anchor=polygon_centroid(polygon);com=context['com'];bear=state['bear'];rel=bear-tcp
height=max(float(tcp[2]),float(state['bag'][:,2].max())+.010-float(rel[:,2].min()))
bottom=float(cfg['table']['height'])+policy['bottom_clearance_m']
end_z=bottom-float(rel[:,2].min())
raised=tcp.copy();raised[2]=height
report['geometry_targets']=dict(mouth_area_centroid_xy_m=anchor,actual_mass_com_m=com,
    whole_cloth_maximum_z_m=float(state['bag'][:,2].max()),actual_bear_minimum_z_m=float(bear[:,2].min()),
    high_tcp_z_m=height,raise_distance_m=height-tcp[2],frozen_high_bear_min_z_m=height+rel[:,2].min(),
    final_tcp_z_m=end_z,frozen_final_bear_min_z_m=bottom,frozen_final_bear_max_z_m=end_z+rel[:,2].max(),
    original_body_rim_min_z_m=mouth['rim_z_min_m'],whole_final_bear_below_rim_by_2mm=bool(end_z+rel[:,2].max()<=mouth['rim_z_min_m']-.002))
fingers=context['fingers']
cache={}

def stage(name,qs,start,target,duration):
    row=dict(stage=name,target_tcp_m=target,declared_start_tcp_m=start,status='not_assessed',physical_success=False)
    qend,e,a=planner.solve('right',target,target_rotation,qs['right'],max_nfev=250)
    row['endpoint_single_seed_ik']=dict(q=qend,position_error_m=e,rotation_error_rad=a,passes_original_tolerances=e<=.0005 and a<=.002)
    soft=check_translation_sweep(bear,rest['bear_tets'],state['q_left'],new,
        translation_m=start-tcp,displacement_m=target-start,required_gap_m=.005,planner=planner)
    row['whole_tetra_left_sweep']=soft
    try:
        paths=planner.plan_cartesian(qs,{'right':target},rotations={'right':target_rotation},samples=45)
        curves={}
        for side in ('left','right'):
            values=np.asarray([p['q'].get(side,qs[side]) for p in paths]);values[0]=qs[side]
            curves[side]=CubicSpline([p['fraction'] for p in paths],values,axis=0)
        plan=MotionPlan(curves,duration,fingers,fingers)
        ext=plan.validate_joint_limits(planner.limits)
        plan.limit_rate(.35,.5)
        rates=continuous_rate_bounds(plan)
        v=max(float(np.max(r['velocity_upper_bound_rad_s'])) for r in rates.values())
        acc=max(float(np.max(r['acceleration_upper_bound_rad_s2'])) for r in rates.values())
        plan.duration*=max(1.,v/.35,np.sqrt(acc/.5))*1.002
        rates=continuous_rate_bounds(plan)
        v=max(float(np.max(r['velocity_upper_bound_rad_s'])) for r in rates.values())
        acc=max(float(np.max(r['acceleration_upper_bound_rad_s2'])) for r in rates.values())
        if v>.35 or acc>.5: raise ValueError('Continuous reference rate bound failed')
        minimum=float('inf');worst=None;pmax=rmax=0.;samples=[]
        for tt in np.linspace(0.,plan.duration,401):
            sample=plan.sample(float(tt))
            if not _finite_tree(sample): raise ValueError('Nonfinite raw spline sample')
            checks=clearance_individual(planner,sample['q'])
            category,value=min(checks.items(),key=lambda x:x[1]['distance'])
            pp,rr=planner.fk('right',sample['q']['right'])
            fraction=float(minimum_jerk(tt/plan.duration)[0])
            p_err=float(np.linalg.norm(pp-(start+fraction*(target-start))))
            r_err=float((rr*target_rotation.inv()).magnitude())
            if not np.isfinite([p_err,r_err]).all(): raise ValueError('Nonfinite FK error')
            item=dict(time_s=float(tt),fraction=fraction,rigid_clearances=checks,tcp_m=pp,
                position_error_m=p_err,full_rotation_error_rad=r_err)
            samples.append(item)
            if value['distance']<minimum:minimum=value['distance'];worst=dict(time_s=float(tt),category=category,**value)
            pmax=max(pmax,p_err);rmax=max(rmax,r_err)
        row['robot_path']=dict(status='passed' if minimum>=.008 and pmax<=.0005 and rmax<=.002 else 'rejected',
            sample_count=401,minimum_rigid_clearance_m=minimum,worst_rigid_sample=worst,
            maximum_fk_position_error_m=pmax,maximum_fk_full_rotation_error_rad=rmax,
            duration_s=plan.duration,continuous_velocity_bound_rad_s=v,continuous_acceleration_bound_rad_s2=acc,
            continuous_rate_bounds=rates,analytic_joint_extrema=ext,q_end=plan.end_q())
        np.savez(OUT/(name+'_plan.npz'),duration=plan.duration,left_x=curves['left'].x,left_c=curves['left'].c,
            right_x=curves['right'].x,right_c=curves['right'].c,declared_start_tcp=start,target_tcp=target,target_rotation_xyzw=target_rotation.as_quat())
        (OUT/(name+'_samples.json')).write_text(json.dumps(_serial(samples),indent=2,allow_nan=False)+'\n')
        row['status']='passed_frozen_route_stage' if row['robot_path']['status']=='passed' and soft['status']=='clear' else 'rejected'
        return row,plan.end_q()
    except Exception as error:
        row.update(status='rejected',robot_path_error=type(error).__name__+': '+str(error))
        return row,None

common,common_q=stage('raise_at_actual_xy',initial_q,tcp,raised,3.)
report['common_raise']=common
save();print('common raise',common['status'],common.get('robot_path_error',''),flush=True)
if common['status']=='passed_frozen_route_stage':
    sections,envelopes=_section_envelopes(bear,rest['bear_tets'])
else:
    sections=envelopes=None
for index,offset in enumerate(report['four_com_offsets_m']):
    desired_com=anchor+offset;delta=desired_com-com[:2]
    outside=raised.copy();outside[:2]+=delta
    along_y=raised.copy();along_y[1]=outside[1]
    insert=outside.copy();insert[2]=end_z
    targets=[('high_y',along_y,3.),('high_x',outside,4.),('vertical_descent',insert,6.)]
    case=dict(index=index,com_offset_xy_m=offset,desired_com_xy_m=desired_com,translation_xy_m=delta,
        stage_targets={name:target for name,target,_ in targets},stages=[],all_route_gates_pass=False)
    inside,dist=projected_inside(np.asarray([desired_com]),polygon)
    case['original_com_margin']=dict(inside=bool(inside[0]),distance_m=float(dist[0]),passes=bool(inside[0] and dist[0]>=.002))
    if envelopes is not None:
        section=_evaluate_offset(envelopes,delta,polygon,triangles,mouth,policy)
        case['original_section_policy']={k:v for k,v in section.items() if k!='sections'}
    current=raised.copy();qs=common_q;prior_ok=common['status']=='passed_frozen_route_stage'
    for name,target,duration in targets:
        if prior_ok:
            value,next_q=stage(f'offset{index}_{name}',qs,current,target,duration)
            case['stages'].append(value);prior_ok=value['status']=='passed_frozen_route_stage'
            if prior_ok:qs=next_q;current=target.copy()
            print('case',index,name,value['status'],value.get('robot_path_error',''),flush=True)
        else:
            case['stages'].append(dict(stage=f'offset{index}_{name}',target_tcp_m=target,status='not_assessed_after_prior_rejection'))
    case['all_route_gates_pass']=bool(prior_ok)
    case['original_shape_policy_also_passes']=bool(prior_ok and case['original_com_margin']['passes'] and
        not case.get('original_section_policy',{}).get('section_guard_rejections',['not assessed']) and
        report['geometry_targets']['whole_final_bear_below_rim_by_2mm'])
    report['cases'].append(case);save()
report['status']='bounded_frozen_study_complete'
report['any_full_high_route_passes']=any(c['all_route_gates_pass'] for c in report['cases'])
report['any_route_and_original_shape_policy_passes']=any(c['original_shape_policy_also_passes'] for c in report['cases'])
save();print('DONE',str(OUT/'review.json'),report['any_full_high_route_passes'],flush=True)
