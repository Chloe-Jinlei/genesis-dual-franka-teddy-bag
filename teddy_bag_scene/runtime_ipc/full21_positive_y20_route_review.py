"""Read-only two-route reconstruction: original planner vs frozen runner proxy."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_21'
OUT=RUN/'independent-positive-y20-route-review';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
from motion import Planner,MotionPlan,CubicSpline,minimum_jerk
from contact_insertion import _state_context,_policy
from contact_robot_geometry import check_translation_sweep
from postlift_yaw import _serial,_finite_tree

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((RUN/'config.json').read_text());cfg['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
rest=dict(np.load(RUN/'rest.npz'));state=dict(np.load(RUN/'orient_teddy_postlift.npz'))
context=_state_context(rest,state,cfg,_policy(cfg));assert not context['reasons']
planner=context['planner'];fingers=context['fingers'];tcp=state['tcp_right'];rotation=context['rotation']
yaw=np.load(RUN/'postlift_yaw_plan.npz')
command={s:PPoly(yaw[s+'_c'],yaw[s+'_x'])(1.) for s in ['left','right']}
report=dict(kind='full21_0_vs_positive_y20_independent_frozen_route_review',physical_success=False,
    native_executed=False,source_or_config_modified=False,source_time_s=float(state['time']),
    source_actual_physical=context['physical'],source_full_rotation_xyzw=rotation.as_quat(),
    original_base_m=cfg['robots']['right_base'],cases=[],
    input_sha256={str(p):sha(p) for p in [RUN/'config.json',RUN/'rest.npz',RUN/'orient_teddy_postlift.npz',RUN/'guarded_contact_candidate.json',RUN/'independent-positive-y20-original-base.json',RUN/'postlift_yaw_plan.npz',RUN/'source/motion.py',RUN/'source/run.py',Path(__file__)]},
    reconstruction_scope=['Planner-report reconstruction uses measured actual post-yaw q and each report duration; no second retiming is applied to the reported curve.',
        'Runner proxy starts from exact saved post-yaw command curve endpoint and calls frozen Planner.trajectory with each report duration,45IK knots,201 checks and original .35/.5 retiming, matching move_guarded.',
        'Observed finger values are frozen at actual post-yaw values; a new native run will remeasure fingers at each stage. Raw401 audit includes both paired mean and original unequal fingers.',
        'The fourth downward stage is provisional only. Real runtime requires fresh actual transport reobservation and may reject/recenter before insertion.',
        'Native contact forces, future cloth/bear deformation, retention and continuous FK collision are not established by this reconstruction.'])

def save():
    report['inputs_unchanged']=all(sha(Path(p))==h for p,h in report['input_sha256'].items())
    (OUT/'review.json').write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')

def raw_checks(plan,start,target):
    minimum=float('inf');worst=None;per_category={};raw_count=0;pmax=rmax=vmax=amax=0.;samples=[]
    required={'left_self','left_table','left_mounts','right_self','right_table','right_mounts','arms'}
    for time in np.linspace(0.,plan.duration,401):
        sample=plan.sample(float(time));assert _finite_tree(sample)
        all_rows={}
        for finger_kind in ['paired_mean','actual_individual']:
            if finger_kind=='actual_individual':
                for side in ['left','right']:
                    for j,v in enumerate(state['q_'+side][7:],1):planner.d.qpos[planner.m.joint(f'{side}_finger_joint{j}').qposadr[0]]=v
            rows=planner.clearance(sample['q'],sample['finger'] if finger_kind=='paired_mean' else None)
            assert set(rows)==required and _finite_tree(rows)
            for key,value in rows.items():
                raw_count+=1
                if value['distance']<per_category.get((finger_kind,key),float('inf')):per_category[(finger_kind,key)]=value['distance']
                if value['distance']<minimum:minimum=value['distance'];worst=dict(time_s=float(time),finger_kind=finger_kind,category=key,**value)
            all_rows[finger_kind]=rows
        pos,rot=planner.fk('right',sample['q']['right']);fraction=float(minimum_jerk(time/plan.duration)[0])
        pe=float(np.linalg.norm(pos-(start+fraction*(target-start))));re=float((rot*rotation.inv()).magnitude())
        assert np.isfinite([pe,re]).all();pmax=max(pmax,pe);rmax=max(rmax,re)
        vmax=max(vmax,*[float(np.abs(sample['qd'][s]).max()) for s in ['left','right']]);amax=max(amax,*[float(np.abs(sample['qdd'][s]).max()) for s in ['left','right']])
        samples.append(dict(time_s=float(time),path_fraction=fraction,clearances=all_rows,position_error_m=pe,full_rotation_error_rad=re))
    return dict(sample_count=401,raw_distance_count=raw_count,all_raw_values_finite=True,minimum_rigid_clearance_m=minimum,
        worst_rigid_sample=worst,rigid_8mm_pass=minimum>=.008,maximum_fk_position_error_m=pmax,maximum_fk_full_rotation_error_rad=rmax,
        sampled_fk_original_tolerances_pass=pmax<=.0005 and rmax<=.002,sampled_peak_velocity_rad_s=vmax,sampled_peak_acceleration_rad_s2=amax,
        duration_s=plan.duration,analytic_joint_limits=plan.validate_joint_limits(planner.limits)),samples

for name,file in [('zero','guarded_contact_candidate.json'),('positive_y20','independent-positive-y20-original-base.json')]:
    choice=json.loads((RUN/file).read_text());assert choice['status']=='candidate' and abs(choice['observed_time_s']-float(state['time']))<1e-9
    assert (rotation*Rotation.from_quat(choice['route']['rotation_xyzw']).inv()).magnitude()<1e-12
    case=dict(name=name,source_candidate=str(RUN/file),reported_route=choice['route'],stages=[])
    q_report={k:v.copy() for k,v in context['qs'].items()};q_native={k:v.copy() for k,v in command.items()};start=tcp.copy()
    for index,stage in enumerate(choice['route']['stages']):
        target=np.asarray(stage['right_tcp_m']);entry=dict(index=index,name=stage['name'],target_tcp_m=target,declared_start_tcp_m=start,provisional_only=index==3)
        paths=planner.plan_cartesian(q_report,{'right':target},rotations={'right':rotation},samples=45)
        curves={s:CubicSpline([p['fraction'] for p in paths],np.asarray([p['q'].get(s,q_report[s]) for p in paths]),axis=0) for s in ['left','right']}
        original=MotionPlan(curves,float(stage['duration_s']),fingers,fingers)
        entry['report_reconstruction_q_end_max_error_rad']=float(np.max(np.abs(original.end_q()['right']-stage['right_q_end'])))
        assert entry['report_reconstruction_q_end_max_error_rad']<1e-10
        native=planner.trajectory(q_native,{'right':target},duration=float(stage['duration_s']),rotations={'right':rotation},
            finger_start=fingers,finger_end=fingers,clearance_margin=.008,validate_samples=201,max_joint_velocity=.35,max_joint_acceleration=.5)
        for kind,plan in [('planner_report',original),('runner_proxy',native)]:
            audit,samples=raw_checks(plan,start,target);entry[kind]=audit
            prefix=f'{name}_{index}_{kind}'
            np.savez(OUT/(prefix+'.npz'),duration=plan.duration,left_x=plan.curves['left'].x,left_c=plan.curves['left'].c,
                right_x=plan.curves['right'].x,right_c=plan.curves['right'].c,declared_start_tcp=start,target_tcp=target,
                full_rotation_xyzw=rotation.as_quat(),left_fingers_measured=state['q_left'][7:],right_fingers_measured=state['q_right'][7:])
            (OUT/(prefix+'_raw401.json')).write_text(json.dumps(_serial(samples),indent=2,allow_nan=False)+'\n')
            audit['archived_plan_path']=str(OUT/(prefix+'.npz'));audit['archived_plan_sha256']=sha(OUT/(prefix+'.npz'))
        entry['runner_proxy_start_command_q_difference_rad']={s:float(np.max(np.abs(native.sample(0.)['q'][s]-q_native[s]))) for s in ['left','right']}
        entry['whole_bear_left_sweeps']={}
        for label,left in [('actual_left',state['q_left']),('commanded_left_with_measured_fingers',np.r_[command['left'],state['q_left'][7:]])]:
            entry['whole_bear_left_sweeps'][label]=check_translation_sweep(state['bear'],rest['bear_tets'],left,cfg,
                translation_m=start-tcp,displacement_m=target-start,required_gap_m=.005,planner=planner)
        q_report=original.end_q();q_native=native.end_q();start=target.copy();case['stages'].append(entry)
        print(name,index,stage['name'],'rigid',entry['runner_proxy']['minimum_rigid_clearance_m'],'soft',[v['status'] for v in entry['whole_bear_left_sweeps'].values()],flush=True)
    case['all401rigid_fk_and_wholebear_left_gates_pass']=all(v['runner_proxy']['rigid_8mm_pass'] and v['runner_proxy']['sampled_fk_original_tolerances_pass'] and all(s['status']=='clear' for s in v['whole_bear_left_sweeps'].values()) for v in case['stages'])
    report['cases'].append(case);save()
report['status']='geometric_reconstruction_complete_continuous_rate_audit_separate';save();print('ARCHIVES_READY',str(OUT),flush=True)
