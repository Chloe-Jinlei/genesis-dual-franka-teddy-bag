"""One accepted hybrid route, frozen-source robot replay only; no native execution."""
from pathlib import Path
import copy
import hashlib
import json
import math
import sys
import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/two_panel_handoff_upright_offset_01'
OLD=ROOT/'runs/two_panel_full_21'
BASE=RUN/'independent-hybrid-minus30-positive-y34p5'
PRIOR=RUN/'independent-hybrid-lift-worldz-minus30-path'
OUT=BASE/'independent_robot_review';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
from motion import Planner,MotionPlan,CubicSpline,minimum_jerk

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(path,d):path.write_text(json.dumps(d,indent=2,default=serial,allow_nan=False)+'\n')
checks={}
def check(name,good):
    checks[name]=bool(good)
    if not good:raise AssertionError(name)

def bernstein_bound(p):
    n=p.degree()
    return max(abs(sum(p.coef[i]*math.comb(k,i)/math.comb(n,i) for i in range(k+1))) for k in range(n+1))

def continuous_outer_rates(plan):
    velocity=acceleration=0.
    inv=lambda s:0. if s<=0 else 1. if s>=1 else brentq(lambda u:float(minimum_jerk(u)[0])-s,0.,1.,xtol=5e-15)
    for curve in plan.curves.values():
        for j in range(len(curve.x)-1):
            ua,ub=inv(curve.x[j]),inv(curve.x[j+1])
            edges=np.linspace(ua,ub,9)
            for a,b in zip(edges[:-1],edges[1:]):
                u=Poly([a,b-a]);s=10*u**3-15*u**4+6*u**5-curve.x[j]
                for k in range(7):
                    c=curve.c[:,j,k];q=c[0]*s**3+c[1]*s**2+c[2]*s+c[3]
                    velocity=max(velocity,bernstein_bound(q.deriv())/((b-a)*plan.duration))
                    acceleration=max(acceleration,bernstein_bound(q.deriv(2))/((b-a)*plan.duration)**2)
    return dict(velocity_rad_s=velocity+1e-9,acceleration_rad_s2=acceleration+1e-9,
        method='Independent degree<=15 composition on eight subintervals per cubic, derivative Bernstein convex hull, 1e-9 numerical pad; not directed-rounding arithmetic.')

paths=[BASE/'review.json',BASE/'guarded_candidate.json',BASE/'hybrid_hypothetical_observation.npz',BASE/'rest.npz',BASE/'diagnostic_config.json',
       PRIOR/'review.json',PRIOR/'hybrid_hypothetical_after_yaw.npz',PRIOR/'hypothetical_yaw_plan.npz',
       RUN/'settle_handoff.npz',RUN/'rest.npz',RUN/'config.json',OLD/'lift_teddy.npz',OLD/'rest.npz',OLD/'config.json',
       RUN/'source/motion.py',RUN/'source/assets/panda_tactile.xml',Path(__file__)]
before={str(p):sha(p) for p in paths}
cfg=read(BASE/'diagnostic_config.json');basecfg=read(RUN/'config.json');oldcfg=read(OLD/'config.json')
provenance=read(PRIOR/'review.json');screen=read(BASE/'review.json');choice=read(BASE/'guarded_candidate.json')
state=dict(np.load(BASE/'hybrid_hypothetical_observation.npz'));rest=dict(np.load(BASE/'rest.npz'))
bag=dict(np.load(RUN/'settle_handoff.npz'));bear=dict(np.load(OLD/'lift_teddy.npz'))
after=dict(np.load(PRIOR/'hybrid_hypothetical_after_yaw.npz'));yaw=np.load(PRIOR/'hypothetical_yaw_plan.npz')
check('original_provenance_hashes_match',all(sha(Path(p))==h for p,h in provenance['input_sha256'].items()))
check('screen_input_hashes_match',all(sha(Path(p))==h for p,h in screen['input_sha256'].items()))
check('frozen_motion_is_reviewed_continuous_rate_fix',sha(RUN/'source/motion.py')=='544d6bed1aedfe1603db860af0cd4d230641714827ff23ef27c93d0a12e32a31')
check('hybrid_only_declared_arrays',set(state)=={'time','bag','bear','q_left','q_right','tcp_left','tcp_right'})
check('hybrid_exact_saved_after_yaw',all(np.array_equal(state[k],after[k]) for k in state))
check('hybrid_bag_and_left_exact',all(np.array_equal(state[k],bag[k]) for k in ['bag','q_left','tcp_left']))
check('both_native_rests_identical',all(np.array_equal(v,np.load(RUN/'rest.npz')[k]) and np.array_equal(v,np.load(OLD/'rest.npz')[k]) for k,v in rest.items()))
check('same_robots_materials_environment',all(basecfg[k]==oldcfg[k] for k in ['robots','bag','bear','simulation','table']))
expected=copy.deepcopy(basecfg);expected['task']['contact_trial']['center_offsets_xy_m']=[[0.,.0345]];expected['task']['contact_trial']['postlift_yaw']={'world_z_degrees':-30.,'duration_s':6.}
check('diagnostic_config_exact_declared_changes',cfg==expected)
check('only_one_accepted_route_under_review',choice['status']=='candidate' and screen['only_com_offset_m']==[0.,.0345] and choice['not_a_joint_native_snapshot'])
cfg['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
planner=Planner(cfg)
qs={s:state['q_'+s][:7].copy() for s in ['left','right']}
fingers={s:float(np.mean(state['q_'+s][7:])) for s in ['left','right']}
actual_tcp,rotation=planner.fk('right',qs['right'])
check('route_full_R_from_actual_FK',(rotation*Rotation.from_quat(choice['route']['rotation_xyzw']).inv()).magnitude()<1e-12)
check('actual_both_TCP_FK_match',all(np.linalg.norm(planner.fk(s,qs[s])[0]-state['tcp_'+s])<1e-12 for s in ['left','right']))
check('candidate_clock_matches_hybrid',choice['observed_time_s']==float(state['time']))

from scipy.interpolate import PPoly
source_tcp,source_R=planner.fk('right',bear['q_right'][:7])
endpoint_q={side:PPoly(yaw[side+'_c'],yaw[side+'_x'])(1.) for side in ['left','right']}
check('yaw_endpoint_q_exact',all(np.array_equal(endpoint_q[side],state['q_'+side][:7]) for side in endpoint_q))
check('right_fingers_inherited_exact_actual_lift',np.array_equal(state['q_right'][7:],bear['q_right'][7:]))
expected_bear=(rotation*source_R.inv()).apply(bear['bear']-bear['tcp_right'])+actual_tcp
check('bear_rigid_transform_from_actual_FK_exact',np.array_equal(expected_bear,state['bear']))
check('total_world_Z_minus30_from_lift',(rotation*(Rotation.from_euler('z',-30.,degrees=True)*source_R).inv()).magnitude()<.002)
steps=int(np.ceil(float(yaw['duration'])/cfg['simulation']['dt']))
check('synthetic_clock_integer_steps',float(state['time'])==float(bear['time'])+steps*cfg['simulation']['dt'])
required={'left_self','left_table','left_mounts','right_self','right_table','right_mounts','arms'}
base_durations={'height_at_current_xy':2.5,'transport_y_at_current_x':3.,'translate_outside_mouth':3.,'insert_candidate':6.,'withdraw_after_release':3.}
rows=[];start=actual_tcp.copy();stages=[*choice['route']['stages'],choice['route']['withdraw_stage']]
report=dict(kind='independent_single_minus30_positive_y34p5_hybrid_robot_route',physical_success=False,native_execution=False,
    hybrid_state_not_simultaneously_observed=True,checks=checks,stages=rows,input_sha256=before,
    reconstruction_scope=[
        'Exact accepted route replay from synthetic hybrid measured q, full actual right FK rotation, 45 IK knots. Original planner duration uses .55/1.0 defaults.',
        'Each report curve is then retimed under .35/.5 using the same coefficients. This is not a native command stream, command-origin replay, or fresh future observation.',
        '401 time samples check every raw result in the original seven rigid categories, paired mean fingers and actual unequal fingers. Release has 401 finger-opening samples.',
        'Continuous joint positions and time-derivative extrema are numerical polynomial calculations; independent Bernstein bounds cross-check continuous rates. FK and rigid separation are sampled only.',
        'No cloth/teddy dynamics, front-handle collision, grasp force, future body containment, or task success follows. Other agent separately assesses frozen soft geometry.',
        'Bear/right state derives from actual full21 pre-yaw lift through the recorded hypothetical -30-degree yaw endpoint; bag/left derives from the independent handoff probe. No simultaneous native reachability is implied.'])
for index,stage in enumerate(stages):
    name=stage['name'];target=np.asarray(stage['right_tcp_m'],float)
    is_withdraw=name=='withdraw_after_release'
    if is_withdraw:
        release=[];release_min=float('inf');release_worst=None
        for fraction in np.linspace(0.,1.,401):
            finger={**fingers,'right':float((1-fraction)*fingers['right']+fraction*.04)}
            raws={}
            for kind in ['paired_mean','actual_individual']:
                if kind=='actual_individual':
                    for side in qs:
                        for j,v in enumerate(state['q_'+side][7:],1):
                            value=(1-fraction)*v+fraction*.04 if side=='right' else v
                            planner.d.qpos[planner.m.joint(f'{side}_finger_joint{j}').qposadr[0]]=value
                raw=planner.clearance(qs,finger if kind=='paired_mean' else None)
                check('release_all_seven_categories',set(raw)==required)
                check('release_each_raw_distance_finite',all(np.isfinite(v['distance']) for v in raw.values()))
                for cat,v in raw.items():
                    if v['distance']<release_min:release_min=v['distance'];release_worst=dict(fraction=float(fraction),finger_kind=kind,category=cat,**v)
                raws[kind]=raw
            release.append(dict(fraction=float(fraction),raw=raws))
        check('release_rigid_8mm',release_min>=.008)
        report['release']=dict(samples=401,raw_distance_count=401*2*7,minimum_rigid_clearance_m=release_min,worst=release_worst)
        save(OUT/'release_raw401.json',release)
        fingers={**fingers,'right':.04}
    path=planner.plan_cartesian(qs,{'right':target},rotations={'right':rotation},samples=45)
    curves={s:CubicSpline([p['fraction'] for p in path],np.asarray([p['q'].get(s,qs[s]) for p in path]),axis=0) for s in qs}
    original=MotionPlan(curves,base_durations[name],fingers,fingers);original.validate_joint_limits(planner.limits);original.limit_rate(.55,1.)
    check(name+'_reported_duration_replay',abs(original.duration-stage['duration_s'])<1e-12)
    check(name+'_reported_right_endpoint_replay',np.max(np.abs(original.end_q()['right']-stage['right_q_end']))<1e-12)
    retimed=MotionPlan(curves,float(stage['duration_s']),fingers,fingers);retimed.limit_rate(.35,.5)
    analytic=retimed.validate_joint_limits(planner.limits);rates=retimed.continuous_rate_extrema();outer=continuous_outer_rates(retimed)
    check(name+'_continuous_rates_under035_05',outer['velocity_rad_s']<=.35 and outer['acceleration_rad_s2']<=.5)
    check(name+'_same_curves_retimed',all(np.array_equal(original.curves[s].c,retimed.curves[s].c) for s in qs))
    minimum=float('inf');worst=None;pmax=rmax=0.;rawrows=[]
    for t in np.linspace(0.,retimed.duration,401):
        sample=retimed.sample(float(t));progress=float(minimum_jerk(t/retimed.duration)[0]);raws={}
        for kind in ['paired_mean','actual_individual']:
            if kind=='actual_individual':
                for side in qs:
                    for j,v in enumerate(state['q_'+side][7:],1):
                        value=.04 if is_withdraw and side=='right' else v
                        planner.d.qpos[planner.m.joint(f'{side}_finger_joint{j}').qposadr[0]]=value
            raw=planner.clearance(sample['q'],sample['finger'] if kind=='paired_mean' else None)
            check(name+'_all_seven_categories',set(raw)==required)
            check(name+'_all_raw_distances_finite',all(np.isfinite(v['distance']) for v in raw.values()))
            for cat,v in raw.items():
                if v['distance']<minimum:minimum=v['distance'];worst=dict(time_s=float(t),fraction=progress,finger_kind=kind,category=cat,**v)
            raws[kind]=raw
        pos,R=planner.fk('right',sample['q']['right'])
        pe=float(np.linalg.norm(pos-(start+progress*(target-start))));re=float((R*rotation.inv()).magnitude())
        check(name+'_raw_FK_errors_finite',np.isfinite([pe,re]).all())
        pmax=max(pmax,pe);rmax=max(rmax,re)
        rawrows.append(dict(time_s=float(t),fraction=progress,clearances=raws,position_error_m=pe,full_rotation_error_rad=re))
    check(name+'_401rigid_8mm',minimum>=.008)
    check(name+'_401FK_original_tolerances',pmax<=.0005 and rmax<=.002)
    check(name+'_left_q_frozen',np.array_equal(retimed.curves['left'](0.),state['q_left'][:7]) and np.all(retimed.curves['left'].c[:3]==0.))
    prefix=f'{index}_{name}'
    archive=OUT/(prefix+'.npz')
    np.savez(archive,original_duration=original.duration,duration=retimed.duration,
        left_x=curves['left'].x,left_c=curves['left'].c,right_x=curves['right'].x,right_c=curves['right'].c,
        start_tcp=start,target_tcp=target,full_rotation_xyzw=rotation.as_quat(),
        left_fingers_actual=state['q_left'][7:],right_fingers_actual=np.full(2,.04) if is_withdraw else state['q_right'][7:])
    save(OUT/(prefix+'_raw401.json'),rawrows)
    rows.append(dict(name=name,original_duration_s=original.duration,retimed035_05_duration_s=retimed.duration,
        analytic_joint_extrema=analytic,continuous_extrema=rates,independent_continuous_rate_outer_bounds=outer,
        raw_rigid_samples=401,raw_distance_count=401*2*7,minimum_rigid_clearance_m=minimum,worst_rigid_sample=worst,
        maximum_sampled_fk_position_error_m=pmax,maximum_sampled_fk_full_R_error_rad=rmax,
        archive=dict(path=str(archive),sha256=sha(archive))))
    qs=original.end_q();start=target.copy()
    print(name,'min rigid',minimum,'rate bounds',outer['velocity_rad_s'],outer['acceleration_rad_s2'],flush=True)
    save(OUT/'review.json',report)
check('all_inputs_unchanged',all(sha(Path(p))==h for p,h in before.items()))
report.update(status='single_hybrid_robot_route_reconstruction_passed',all_checks_passed=all(checks.values()),failed_checks=[k for k,v in checks.items() if not v])
save(OUT/'review.json',report)
print(json.dumps(dict(output=str(OUT/'review.json'),sha256=sha(OUT/'review.json'),checks=len(checks),passed=report['all_checks_passed']),indent=2))
